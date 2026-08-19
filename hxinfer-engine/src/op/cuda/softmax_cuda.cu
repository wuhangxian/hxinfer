#include "tensor/tensor.h"
#include "cuda_runtime.h"
#include "iostream"
namespace hxinfer{

    // __global__ 表示这是一个在 GPU 上执行，但是可以被 CPU 端调用的 Kernel 函数
    __global__ void softmax_kernel_cuda(const float* in_data, float* out_data, size_t total_elements){
        
        // 声明一个动态大小的共享内存（Shared Memory）。
        // 共享内存存在于 GPU 的 Block 内部，同一个 Block 内的所有线程读写它极快，用来做线程间的数据交换。
        // 大小在 host 端启动 kernel 时（<<<>>>的第三个参数）指定。
        extern __shared__ float shared[]; 
        
        int tid = threadIdx.x;        // 当前线程在 Block 中的 ID（0 到 block_size-1）
        int block_size = blockDim.x;  // 当前 Block 一共有多少个线程（在这个例子里是 256）

        // ==========================================
        // Pass 1: 寻找整个数组的全局最大值 (Global Max)
        // ==========================================
        
        // 1. 初始化一个极小值，用来打底。
        float local_max = -3.402823466e+38f; 
        
        // 2. 网格跨步循环 (Grid-Stride Loop / Block-Stride Loop)
        // 为什么不用 i++？因为数据量可能远大于线程数（比如 10000个数据，但只有 256个线程）。
        // 这样写，线程 0 会处理索引 0, 256, 512... 的数据。
        // 每个线程都在自己负责的那些数据里，找出一个“局部最大值”。
        for(size_t i = tid; i < total_elements; i += block_size){
            float val = in_data[i];
            if(val > local_max) local_max = val;
        }
        
        // 3. 把每个线程找到的局部最大值，写到共享内存里对应自己的位置。
        shared[tid] = local_max; 
        
        // 4. 线程同步屏障。必须等所有 256 个线程都把自己找到的 local_max 写完，才能进行下一步的归约。
        __syncthreads();

        // 5. 树形归约 (Tree Reduction) 求全局最大值
        // 初始 stride = 128。
        // 第一轮：线程 0~127 会把 shared[0~127] 和 shared[128~255] 比较，大的留在 shared[0~127] 里。
        // 第二轮：stride = 64。线程 0~63 继续两两比较...
        // 最终，整个数组的最大值会被推到 shared[0] 的位置。
        for(int stride = block_size / 2; stride > 0; stride >>= 1){
            if(tid < stride){
                if(shared[tid + stride] > shared[tid]){
                    shared[tid] = shared[tid + stride];
                }
            }
            __syncthreads(); // 每一层树算完，都要同步一次，防止数据读写混乱
        }
        
        // 6. 把存在 shared[0] 的全局最大值拿出来，每个线程都存一份到自己的寄存器 max_val 里。
        float max_val = shared[0];
        __syncthreads(); // 再次同步，准备进入下一阶段

        // ==========================================
        // Pass 2: 计算 exp(x_i - max) 并求全局和 (Global Sum)
        // ==========================================
        float local_sum = 0.0f;
        
        // 1. 依然是跨步循环，每个线程处理自己负责的数据
        for(size_t i = tid; i < total_elements; i += block_size){
            // 核心公式：减去最大值，防止数值溢出 (Safe Softmax)
            float exp_val = expf(in_data[i] - max_val); 
            
            // 把分子结果直接写进输出数组里（注意，这里 out_data 存的还不是最终结果，只是分子）
            out_data[i] = exp_val; 
            
            // 顺手把当前线程计算出的 exp 值累加起来，得到一个局部和
            local_sum += exp_val; 
        }
        
        // 2. 把局部和写入共享内存
        shared[tid] = local_sum;
        __syncthreads();

        // 3. 再次使用树形归约，这次是求和（加法）。
        // 结束后，全局的 exp 总和会存在 shared[0] 里。
        for(int stride = block_size / 2; stride > 0; stride >>= 1){
            if(tid < stride){
                shared[tid] += shared[tid + stride];
            }
            __syncthreads();
        }
        
        // 4. 每个线程获取全局总和
        float sum_val = shared[0];
        __syncthreads();

        // ==========================================
        // Pass 3: 归一化 (Normalization)
        // ==========================================
        // 每个线程再次遍历自己负责的数据，用刚刚存入 out_data 的分子，除以全局总和 sum_val
        for(size_t i = tid; i < total_elements; i += block_size){
            out_data[i] = out_data[i] / sum_val;
        }
    }

    void softmax_cuda(const std::shared_ptr<Tensor>& input, std::shared_ptr<Tensor>& output){
        // 类型检查：确保输入输出都在 GPU 上
        if(input->tensor_device_type() != DeviceType::kDeviceCUDA ||
            output->tensor_device_type() != DeviceType::kDeviceCUDA){
            std::cerr << "[Fatal Error]softmax_cuda expects CUDA Tensors!" << std::endl;
            return;
        }
        
        size_t total_elements = input->tensor_total_elements();
        if(total_elements == 0){
            return;
        }
        const float *d_in = input->tensor_data_ptr<float>();
        float *d_out = output->tensor_data_ptr<float>();

        // 配置 CUDA 启动参数
        int threads_per_block = 256; 
        
        // 计算需要的共享内存大小：256 个线程，每个线程存一个 float，所以是 256 * 4 bytes
        size_t shared_mem_bytes = threads_per_block * sizeof(float);

        // <<<1, threads_per_block, shared_mem_bytes>>>
        // 1 表示只启动 1 个 Block (网格大小为 1)
        // threads_per_block = 256 表示这个 Block 里面有 256 个线程
        // shared_mem_bytes 动态分配共享内存大小
        softmax_kernel_cuda<<<1, threads_per_block, shared_mem_bytes>>>(d_in, d_out, total_elements);
    }
}