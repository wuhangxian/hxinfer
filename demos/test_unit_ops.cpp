// hxinfer unit tests for all CUDA operators
// Uses mathematical identities — no PyTorch dependency
// Each test: create known input -> run operator -> verify against expected values
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "base/allocator.h"
#include "base/config.h"
#include "tensor/tensor.h"
#include "op/math_ops.h"

using namespace hxinfer;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK_NEAR(actual, expected, tol, name) do { \
    if (fabs((float)(actual) - (float)(expected)) < (tol)) { \
        std::cout << "  PASS: " << name << " (got=" << (actual) << ", exp=" << (expected) << ")" << std::endl; \
        g_pass++; \
    } else { \
        std::cout << "  FAIL: " << name << " (got=" << (actual) << ", exp=" << (expected) << ", diff=" << fabs((float)(actual)-(float)(expected)) << ")" << std::endl; \
        g_fail++; \
    } \
} while(0)

#define CHECK_EQ(actual, expected, name) do {     if ((actual) == (expected)) {         std::cout << "  PASS: " << name << " (got=" << (actual) << ")" << std::endl;         g_pass++;     } else {         std::cout << "  FAIL: " << name << " (got=" << (actual) << ", exp=" << (expected) << ")" << std::endl;         g_fail++;     } } while(0)

#define CHECK_TRUE(cond, name) do { \
    if (cond) { std::cout << "  PASS: " << name << std::endl; g_pass++; } \
    else { std::cout << "  FAIL: " << name << std::endl; g_fail++; } \
} while(0)
#define CHECK_EQ(actual, expected, name) do { \
    if ((actual) == (expected)) { \
        std::cout << "  PASS: " << name << " (got=" << (actual) << ")" << std::endl; \
        g_pass++; \
    } else { \
        std::cout << "  FAIL: " << name << " (got=" << (actual) << ", exp=" << (expected) << ")" << std::endl; \
        g_fail++; \
    } \
} while(0)

static std::shared_ptr<CPUAllocator> cpu_alloc;
static std::shared_ptr<CUDAAllocator> cuda_alloc;

// Helper: create FP32 CUDA tensor from host data
static std::shared_ptr<Tensor> make_cuda_tensor_f32(const std::vector<float>& data, std::vector<int> shape) {
    auto t = std::make_shared<Tensor>(cuda_alloc, shape, DataType::kDataTypeFP32);
    t->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(t->raw_data_ptr(), data.data(), data.size()*sizeof(float), cudaMemcpyHostToDevice);
    return t;
}

static std::shared_ptr<Tensor> make_cuda_tensor_f16(const std::vector<float>& data, std::vector<int> shape) {
    std::vector<__half> h(data.size());
    for (size_t i = 0; i < data.size(); i++) h[i] = __float2half(data[i]);
    auto t = std::make_shared<Tensor>(cuda_alloc, shape, DataType::kDataTypeFP16);
    t->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(t->raw_data_ptr(), h.data(), h.size()*sizeof(__half), cudaMemcpyHostToDevice);
    return t;
}

static std::shared_ptr<Tensor> make_cuda_int_tensor(const std::vector<int>& data, std::vector<int> shape) {
    auto t = std::make_shared<Tensor>(cuda_alloc, shape, DataType::kDataTypeFP32);
    t->tensor_set_device_type(DeviceType::kDeviceCUDA);
    cudaMemcpy(t->raw_data_ptr(), data.data(), data.size()*sizeof(int), cudaMemcpyHostToDevice);
    return t;
}

static std::vector<float> read_cuda_tensor_f32(std::shared_ptr<Tensor>& t) {
    std::vector<float> h(t->tensor_total_elements());
    cudaMemcpy(h.data(), t->raw_data_ptr(), h.size()*sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

static std::vector<float> read_cuda_tensor_f16_as_f32(std::shared_ptr<Tensor>& t) {
    std::vector<__half> h(t->tensor_total_elements());
    cudaMemcpy(h.data(), t->raw_data_ptr(), h.size()*sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); i++) f[i] = __half2float(h[i]);
    return f;
}

// ===== Test 1: add =====
void test_add() {
    std::cout << "\n=== test_add ===" << std::endl;
    // FP32
    auto a = make_cuda_tensor_f32({1, 2, 3, 4, 5}, {5});
    auto b = make_cuda_tensor_f32({10, 20, 30, 40, 50}, {5});
    auto out = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{5}, DataType::kDataTypeFP32);
    out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    add_tensor(a, b, out);
    auto r = read_cuda_tensor_f32(out);
    CHECK_NEAR(r[0], 11, 1e-5, "add FP32[0]");
    CHECK_NEAR(r[4], 55, 1e-5, "add FP32[4]");
    // FP16
    auto a16 = make_cuda_tensor_f16({1, 2, 3, 4, 5}, {5});
    auto b16 = make_cuda_tensor_f16({10, 20, 30, 40, 50}, {5});
    auto out16 = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{5}, DataType::kDataTypeFP16);
    out16->tensor_set_device_type(DeviceType::kDeviceCUDA);
    add_tensor(a16, b16, out16);
    auto r16 = read_cuda_tensor_f16_as_f32(out16);
    CHECK_NEAR(r16[0], 11, 0.01, "add FP16[0]");
    CHECK_NEAR(r16[4], 55, 0.01, "add FP16[4]");
}

// ===== Test 2: mul =====
void test_mul() {
    std::cout << "\n=== test_mul ===" << std::endl;
    auto a = make_cuda_tensor_f32({2, 3, 4, 5}, {4});
    auto b = make_cuda_tensor_f32({1, 2, 3, 4}, {4});
    auto out = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{4}, DataType::kDataTypeFP32);
    out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    mul_tensor(a, b, out);
    auto r = read_cuda_tensor_f32(out);
    CHECK_NEAR(r[0], 2, 1e-5, "mul FP32[0]");
    CHECK_NEAR(r[1], 6, 1e-5, "mul FP32[1]");
    CHECK_NEAR(r[3], 20, 1e-5, "mul FP32[3]");
}

// ===== Test 3: silu =====
void test_silu() {
    std::cout << "\n=== test_silu ===" << std::endl;
    // silu(0) = 0, silu(1) = 1/(1+e^-1) ≈ 0.7311, silu(-1) = -1/(1+e^1) ≈ -0.2689
    auto in = make_cuda_tensor_f32({0, 1, -1, 2, -2}, {5});
    auto out = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{5}, DataType::kDataTypeFP32);
    out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    silu_tensor(in, out);
    auto r = read_cuda_tensor_f32(out);
    CHECK_NEAR(r[0], 0.0f, 1e-5, "silu(0)=0");
    CHECK_NEAR(r[1], 0.7311f, 1e-4, "silu(1)=0.7311");
    CHECK_NEAR(r[2], -0.2689f, 1e-4, "silu(-1)=-0.2689");
    CHECK_NEAR(r[3], 1.7616f, 1e-4, "silu(2)=1.7616");
}

// ===== Test 4: softmax =====
void test_softmax() {
    std::cout << "\n=== test_softmax ===" << std::endl;
    // softmax([1,2,3]) — sum should be 1, max element has highest prob
    auto in = make_cuda_tensor_f32({1, 2, 3}, {3});
    auto out = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{3}, DataType::kDataTypeFP32);
    out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    softmax_tensor(in, out);
    auto r = read_cuda_tensor_f32(out);
    // Check sum ≈ 1
    float sum = r[0] + r[1] + r[2];
    CHECK_NEAR(sum, 1.0f, 1e-4, "softmax sum=1");
    // Check ordering: r[2] > r[1] > r[0]
    CHECK_TRUE(r[2] > r[1] && r[1] > r[0], "softmax ordering preserved");
    // Check values: softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
    CHECK_NEAR(r[2], 0.6652f, 1e-3, "softmax[2]=0.6652");
}

// ===== Test 5: argmax =====
void test_argmax() {
    std::cout << "\n=== test_argmax ===" << std::endl;
    // argmax([1,5,3,2,4]) = 1 (index of max value 5)
    auto in = make_cuda_tensor_f32({1, 5, 3, 2, 4}, {5});
    int idx = argmax_tensor(in);
    CHECK_EQ(idx, 1, "argmax([1,5,3,2,4])=1");
    // argmax with max at end
    auto in2 = make_cuda_tensor_f32({1, 2, 3, 9}, {4});
    int idx2 = argmax_tensor(in2);
    CHECK_EQ(idx2, 3, "argmax max-at-end=3");
    // argmax with all same values (should return 0)
    auto in3 = make_cuda_tensor_f32({7, 7, 7}, {3});
    int idx3 = argmax_tensor(in3);
    CHECK_EQ(idx3, 0, "argmax all-same=0");
}

// ===== Test 6: rmsnorm =====
void test_rmsnorm() {
    std::cout << "\n=== test_rmsnorm ===" << std::endl;
    // rmsnorm: x_normed = x / sqrt(mean(x^2) + eps) * weight
    // With weight=1, output RMS should be ~sqrt(d)
    // Input: [3, 4], mean_sq = (9+16)/2 = 12.5, rms = sqrt(12.5) ≈ 3.5355
    // output = [3/3.5355, 4/3.5355] = [0.8485, 1.1314]
    auto in = make_cuda_tensor_f32({3, 4}, {1, 2});
    auto weight = make_cuda_tensor_f32({1, 1}, {2});  // weight=1 for all
    auto out = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, 2}, DataType::kDataTypeFP32);
    out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    rmsnorm_tensor(in, weight, out);
    auto r = read_cuda_tensor_f32(out);
    CHECK_NEAR(r[0], 0.8485f, 1e-3, "rmsnorm[0]=0.8485");
    CHECK_NEAR(r[1], 1.1314f, 1e-3, "rmsnorm[1]=1.1314");
    // Check RMS of output ≈ 1 (when weight=1)
    float rms = sqrt((r[0]*r[0] + r[1]*r[1]) / 2);
    CHECK_NEAR(rms, 1.0f, 1e-3, "rmsnorm output RMS=1");
}

// ===== Test 7: rope =====
void test_rope() {
    std::cout << "\n=== test_rope ===" << std::endl;
    // At pos=0: angle=0, cos=1, sin=0, so output should EQUAL input
    ModelConfig config;
    config.dim = 4;
    config.head = 1;
    config.kv_head = 1;
    config.rope_theta = 10000.0f;

    std::vector<float> q_data = {1.0, 2.0, 3.0, 4.0};  // 1 head, head_dim=4
    auto q = make_cuda_tensor_f32(q_data, {4});
    auto k = make_cuda_tensor_f32(q_data, {4});
    rope_tensor(q, k, config, 0, 10000.0f);
    auto q_out = read_cuda_tensor_f32(q);
    auto k_out = read_cuda_tensor_f32(k);
    // pos=0: no rotation, output == input
    CHECK_NEAR(q_out[0], 1.0f, 1e-5, "rope pos=0 Q[0]=1.0 (identity)");
    CHECK_NEAR(q_out[1], 2.0f, 1e-5, "rope pos=0 Q[1]=2.0 (identity)");
    CHECK_NEAR(q_out[2], 3.0f, 1e-5, "rope pos=0 Q[2]=3.0 (identity)");
    CHECK_NEAR(q_out[3], 4.0f, 1e-5, "rope pos=0 Q[3]=4.0 (identity)");

    // At pos=1, d=0: freq=1/base^0=1, angle=1 rad
    // cos(1)=0.5403, sin(1)=0.8415
    // NeoX: half_dim=2, d=0 pairs with d=2, d=1 pairs with d=3
    // Q[0]' = Q[0]*cos - Q[2]*sin = 1*0.5403 - 3*0.8415 = 0.5403 - 2.5244 = -1.9841
    // Q[2]' = Q[2]*cos + Q[0]*sin = 3*0.5403 + 1*0.8415 = 1.6209 + 0.8415 = 2.4624
    auto q2 = make_cuda_tensor_f32(q_data, {4});
    auto k2 = make_cuda_tensor_f32(q_data, {4});
    rope_tensor(q2, k2, config, 1, 10000.0f);
    auto q2_out = read_cuda_tensor_f32(q2);
    CHECK_NEAR(q2_out[0], -1.9841f, 1e-3, "rope pos=1 Q[0]=-1.9841");
    CHECK_NEAR(q2_out[2], 2.4624f, 1e-3, "rope pos=1 Q[2]=2.4624");
}

// ===== Test 8: matmul =====
void test_matmul() {
    std::cout << "\n=== test_matmul ===" << std::endl;
    // input: [1, 3], weight: [4, 3] (transposed), output: [1, 4]
    // weight stored as [N, K] = [4, 3], matmul does output = input * weight^T
    // input = [1, 2, 3]
    // weight = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]  (4x3 identity+ones)
    // output = [1*1, 2*1, 3*1, 1+2+3] = [1, 2, 3, 6]
    auto in = make_cuda_tensor_f32({1, 2, 3}, {1, 3});
    auto w = make_cuda_tensor_f32({1,0,0, 0,1,0, 0,0,1, 1,1,1}, {4, 3});
    auto out = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, 4}, DataType::kDataTypeFP32);
    out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    matmul_tensor(in, w, out);
    auto r = read_cuda_tensor_f32(out);
    CHECK_NEAR(r[0], 1, 1e-4, "matmul[0]=1");
    CHECK_NEAR(r[1], 2, 1e-4, "matmul[1]=2");
    CHECK_NEAR(r[2], 3, 1e-4, "matmul[2]=3");
    CHECK_NEAR(r[3], 6, 1e-4, "matmul[3]=6");
}

// ===== Test 9: embedding =====
void test_embedding() {
    std::cout << "\n=== test_embedding ===" << std::endl;
    // token_ids=[0,1], weight=[[10,20,30],[40,50,60]] (vocab=2, dim=3)
    // output should be [[10,20,30],[40,50,60]]
    auto ids = make_cuda_int_tensor({0, 1}, {2});
    auto weight = make_cuda_tensor_f32({10,20,30, 40,50,60}, {2, 3});
    auto out = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{2, 3}, DataType::kDataTypeFP32);
    out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    embedding_tensor(ids, weight, out);
    auto r = read_cuda_tensor_f32(out);
    CHECK_NEAR(r[0], 10, 1e-5, "embed[0,0]=10");
    CHECK_NEAR(r[2], 30, 1e-5, "embed[0,2]=30");
    CHECK_NEAR(r[3], 40, 1e-5, "embed[1,0]=40");
    CHECK_NEAR(r[5], 60, 1e-5, "embed[1,2]=60");
}

// ===== Test 10: fused_add_rmsnorm =====
void test_fused_add_rmsnorm() {
    std::cout << "\n=== test_fused_add_rmsnorm ===" << std::endl;
    // residual=[1,2], hidden=[3,4], weight=[1,1], eps=1e-5
    // residual_out = [1+3, 2+4] = [4, 6]
    // mean_sq = (16+36)/2 = 26, rms = sqrt(26) ≈ 5.0990
    // hidden_out = [4/5.099, 6/5.099] = [0.7845, 1.1767]
    auto hidden = make_cuda_tensor_f32({3, 4}, {1, 2});
    auto residual = make_cuda_tensor_f32({1, 2}, {1, 2});
    auto weight = make_cuda_tensor_f32({1, 1}, {2});
    auto h_out = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, 2}, DataType::kDataTypeFP32);
    auto r_out = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, 2}, DataType::kDataTypeFP32);
    h_out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    r_out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    fused_add_rmsnorm_tensor(hidden, residual, h_out, r_out, weight, 1e-5f);
    auto r = read_cuda_tensor_f32(r_out);
    auto h = read_cuda_tensor_f32(h_out);
    CHECK_NEAR(r[0], 4.0f, 1e-3, "fused residual_out[0]=4");
    CHECK_NEAR(r[1], 6.0f, 1e-3, "fused residual_out[1]=6");
    CHECK_NEAR(h[0], 0.7845f, 1e-3, "fused hidden_out[0]=0.7845");
    CHECK_NEAR(h[1], 1.1767f, 1e-3, "fused hidden_out[1]=1.1767");
}

// ===== Test 11: attention_score =====
void test_attention() {
    std::cout << "\n=== test_attention_score ===" << std::endl;
    // Simple attention: 1 head, head_dim=2, single position (pos=0)
    // Q=[1,0], K=[1,0], V=[5,10]
    // score = Q.K / sqrt(2) = 1/sqrt(2) ≈ 0.7071
    // softmax([0.7071]) = [1.0] (single element)
    // output = 1.0 * V = [5, 10]
    ModelConfig config;
    config.dim = 2;
    config.head = 1;
    config.kv_head = 1;
    config.seq_len = 10;

    auto q = make_cuda_tensor_f32({1, 0}, {1, 2});
    auto curr_k = make_cuda_tensor_f32({1, 0}, {1, 2});
    auto curr_v = make_cuda_tensor_f32({5, 10}, {1, 2});
    auto k_cache = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{10, 2}, DataType::kDataTypeFP32);
    auto v_cache = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{10, 2}, DataType::kDataTypeFP32);
    auto out = std::make_shared<Tensor>(cuda_alloc, std::vector<int>{1, 2}, DataType::kDataTypeFP32);
    k_cache->tensor_set_device_type(DeviceType::kDeviceCUDA);
    v_cache->tensor_set_device_type(DeviceType::kDeviceCUDA);
    out->tensor_set_device_type(DeviceType::kDeviceCUDA);
    attention_score_cuda(q, curr_k, curr_v, k_cache, v_cache, out, config, 0);
    auto r = read_cuda_tensor_f32(out);
    // Single position: softmax of single element = 1, output = V
    CHECK_NEAR(r[0], 5.0f, 1e-3, "attn pos=0 out[0]=5");
    CHECK_NEAR(r[1], 10.0f, 1e-3, "attn pos=0 out[1]=10");
}

// ===== Test 12: sample (greedy = temperature=0) =====
void test_sample() {
    std::cout << "\n=== test_sample ===" << std::endl;
    // With temperature very low, sample should pick argmax
    auto logits = make_cuda_tensor_f32({1, 5, 3, 9, 2}, {5});
    int token = sample_tensor(logits, 0.01f, 1.0f);  // near-greedy
    CHECK_EQ(token, 3, "sample greedy picks argmax=3");
}

int main() {
    cpu_alloc = std::make_shared<CPUAllocator>();
    cuda_alloc = std::make_shared<CUDAAllocator>();

    std::cout << "============================================" << std::endl;
    std::cout << "  hxinfer Operator Unit Tests" << std::endl;
    std::cout << "============================================" << std::endl;

    test_add();
    test_mul();
    test_silu();
    test_softmax();
    test_argmax();
    test_rmsnorm();
    test_rope();
    test_matmul();
    test_embedding();
    test_fused_add_rmsnorm();
    test_attention();
    test_sample();

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Results: " << g_pass << " passed, " << g_fail << " failed" << std::endl;
    std::cout << "============================================" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
