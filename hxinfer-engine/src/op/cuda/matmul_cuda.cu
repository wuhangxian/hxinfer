#include "tensor/tensor.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "iostream"

namespace hxinfer{

__global__ void fp32_to_fp16_kernel(const float* in, __half* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) out[idx] = __float2half(in[idx]);
}

// 通用 matmul：根据输入和权重的数据类型自动选择路径
void matmul_cuda(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& weight,
                 std::shared_ptr<Tensor>& output){
    static cublasHandle_t handle = [](){
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        return h;
    }();

    int K = weight->tensor_shapes()[1];
    int M = (int)(input->tensor_total_elements() / K);
    int N = (int)(output->tensor_total_elements() / M);

    float alpha = 1.0f, beta = 0.0f;

    bool weight_fp16 = (weight->tensor_data_type() == DataType::kDataTypeFP16);
    bool input_fp16  = (input->tensor_data_type()  == DataType::kDataTypeFP16);
    bool output_fp16 = (output->tensor_data_type() == DataType::kDataTypeFP16);

    if(weight_fp16){
        const __half* d_B = weight->tensor_data_ptr<__half>();

        if(input_fp16){
            // ===== FP16 × FP16 → FP16/FP32（无需转换，全程 FP16） =====
            const __half* d_A = input->tensor_data_ptr<__half>();

            if(output_fp16){
                __half* d_C = output->tensor_data_ptr<__half>();
                cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                    &alpha,
                    d_B, CUDA_R_16F, K,
                    d_A, CUDA_R_16F, K,
                    &beta,
                    d_C, CUDA_R_16F, N,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            } else {
                float* d_C = output->tensor_data_ptr<float>();
                cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                    &alpha,
                    d_B, CUDA_R_16F, K,
                    d_A, CUDA_R_16F, K,
                    &beta,
                    d_C, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
        } else {
            // ===== FP32 输入 → 转 FP16 → FP16 × FP16 → FP32（旧路径，兼容 15M） =====
            const float* d_A = input->tensor_data_ptr<float>();
            float* d_C = output->tensor_data_ptr<float>();
            int n_input = (int)input->tensor_total_elements();

            static __half* ws = nullptr;
            static int     ws_size = 0;
            if(ws_size < n_input){
                if(ws) cudaFree(ws);
                cudaMalloc(&ws, n_input * sizeof(__half));
                ws_size = n_input;
            }

            int threads = 256;
            int blocks  = (n_input + threads - 1) / threads;
            fp32_to_fp16_kernel<<<blocks, threads>>>(d_A, ws, n_input);

            cublasGemmEx(handle,
                CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                &alpha,
                d_B, CUDA_R_16F, K,
                ws, CUDA_R_16F, K,
                &beta,
                d_C, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    } else {
        // ===== FP32 × FP32 → FP32 =====
        const float* d_A = input->tensor_data_ptr<float>();
        const float* d_B = weight->tensor_data_ptr<float>();
        float*       d_C = output->tensor_data_ptr<float>();

        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
            &alpha, d_B, K, d_A, K,
            &beta,  d_C, N);
    }
}

// ======================== 融合 QKV 投影 ========================
// FP16 激活值 × FP16 权重：不需要 fp32→fp16 转换
void matmul_qkv_cuda(const std::shared_ptr<Tensor>& input,
                      const std::shared_ptr<Tensor>& wq,
                      const std::shared_ptr<Tensor>& wk,
                      const std::shared_ptr<Tensor>& wv,
                      std::shared_ptr<Tensor>& q_out,
                      std::shared_ptr<Tensor>& k_out,
                      std::shared_ptr<Tensor>& v_out)
{
    static cublasHandle_t handle_qkv = [](){
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        return h;
    }();

    int K = wq->tensor_shapes()[1];
    int M = 1;

    float alpha = 1.0f, beta = 0.0f;

    bool input_fp16 = (input->tensor_data_type() == DataType::kDataTypeFP16);

    const __half* d_input_fp16 = nullptr;
    static __half* ws = nullptr;
    static int     ws_size = 0;

    if(input_fp16){
        d_input_fp16 = input->tensor_data_ptr<__half>();
    } else {
        int n_input = K;
        if(ws_size < n_input){
            if(ws) cudaFree(ws);
            cudaMalloc(&ws, n_input * sizeof(__half));
            ws_size = n_input;
        }
        const float* d_input = input->tensor_data_ptr<float>();
        int threads = 256;
        int blocks  = (n_input + threads - 1) / threads;
        fp32_to_fp16_kernel<<<blocks, threads>>>(d_input, ws, n_input);
        d_input_fp16 = ws;
    }

    const __half* d_B_arr[3] = {
        wq->tensor_data_ptr<__half>(),
        wk->tensor_data_ptr<__half>(),
        wv->tensor_data_ptr<__half>(),
    };

    bool q_out_fp16 = (q_out->tensor_data_type() == DataType::kDataTypeFP16);

    for(int i = 0; i < 3; i++){
        if(q_out_fp16){
            __half* d_C = (i==0) ? q_out->tensor_data_ptr<__half>()
                        : (i==1) ? k_out->tensor_data_ptr<__half>()
                                 : v_out->tensor_data_ptr<__half>();
            int N_i = (i==0) ? (int)q_out->tensor_total_elements()
                     : (i==1) ? (int)k_out->tensor_total_elements()
                              : (int)v_out->tensor_total_elements();
            cublasGemmEx(handle_qkv,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N_i, M, K,
                &alpha,
                d_B_arr[i], CUDA_R_16F, K,
                d_input_fp16, CUDA_R_16F, K,
                &beta,
                d_C, CUDA_R_16F, N_i,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            float* d_C = (i==0) ? q_out->tensor_data_ptr<float>()
                        : (i==1) ? k_out->tensor_data_ptr<float>()
                                 : v_out->tensor_data_ptr<float>();
            int N_i = (i==0) ? (int)q_out->tensor_total_elements()
                     : (i==1) ? (int)k_out->tensor_total_elements()
                              : (int)v_out->tensor_total_elements();
            cublasGemmEx(handle_qkv,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N_i, M, K,
                &alpha,
                d_B_arr[i], CUDA_R_16F, K,
                d_input_fp16, CUDA_R_16F, K,
                &beta,
                d_C, CUDA_R_32F, N_i,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
}

// ======================== 融合 Gate+Up 投影 ========================
// FP16 激活值 × FP16 权重：不需要 fp32→fp16 转换
void matmul_gate_up_cuda(const std::shared_ptr<Tensor>& input,
                          const std::shared_ptr<Tensor>& w_gate,
                          const std::shared_ptr<Tensor>& w_up,
                          std::shared_ptr<Tensor>& gate_out,
                          std::shared_ptr<Tensor>& up_out)
{
    static cublasHandle_t handle_gu = [](){
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        return h;
    }();

    int K = w_gate->tensor_shapes()[1];
    int M = 1;

    float alpha = 1.0f, beta = 0.0f;

    bool input_fp16 = (input->tensor_data_type() == DataType::kDataTypeFP16);

    const __half* d_input_fp16 = nullptr;
    static __half* ws = nullptr;
    static int     ws_size = 0;

    if(input_fp16){
        d_input_fp16 = input->tensor_data_ptr<__half>();
    } else {
        int n_input = K;
        if(ws_size < n_input){
            if(ws) cudaFree(ws);
            cudaMalloc(&ws, n_input * sizeof(__half));
            ws_size = n_input;
        }
        const float* d_input = input->tensor_data_ptr<float>();
        int threads = 256;
        int blocks  = (n_input + threads - 1) / threads;
        fp32_to_fp16_kernel<<<blocks, threads>>>(d_input, ws, n_input);
        d_input_fp16 = ws;
    }

    bool out_fp16 = (gate_out->tensor_data_type() == DataType::kDataTypeFP16);

    // gate_proj
    {
        int N_gate = (int)gate_out->tensor_total_elements();
        if(out_fp16){
            cublasGemmEx(handle_gu,
                CUBLAS_OP_T, CUBLAS_OP_N, N_gate, M, K,
                &alpha,
                w_gate->tensor_data_ptr<__half>(), CUDA_R_16F, K,
                d_input_fp16, CUDA_R_16F, K,
                &beta,
                gate_out->tensor_data_ptr<__half>(), CUDA_R_16F, N_gate,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasGemmEx(handle_gu,
                CUBLAS_OP_T, CUBLAS_OP_N, N_gate, M, K,
                &alpha,
                w_gate->tensor_data_ptr<__half>(), CUDA_R_16F, K,
                d_input_fp16, CUDA_R_16F, K,
                &beta,
                gate_out->tensor_data_ptr<float>(), CUDA_R_32F, N_gate,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }

    // up_proj
    {
        int N_up = (int)up_out->tensor_total_elements();
        if(out_fp16){
            cublasGemmEx(handle_gu,
                CUBLAS_OP_T, CUBLAS_OP_N, N_up, M, K,
                &alpha,
                w_up->tensor_data_ptr<__half>(), CUDA_R_16F, K,
                d_input_fp16, CUDA_R_16F, K,
                &beta,
                up_out->tensor_data_ptr<__half>(), CUDA_R_16F, N_up,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasGemmEx(handle_gu,
                CUBLAS_OP_T, CUBLAS_OP_N, N_up, M, K,
                &alpha,
                w_up->tensor_data_ptr<__half>(), CUDA_R_16F, K,
                d_input_fp16, CUDA_R_16F, K,
                &beta,
                up_out->tensor_data_ptr<float>(), CUDA_R_32F, N_up,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
}

} // namespace hxinfer
