#include <iostream>
// 引入 CUDA 运行时核心头文件
#include <cuda_runtime.h>

// ==========================================================
// 1. 编写 Kernel（核函数）：这是要在 GPU 上千军万马一起跑的代码
// __global__ 修饰符告诉编译器：这个函数从 CPU 启动，但在 GPU 执行
// ==========================================================
__global__ void array_multiply_kernel(float* d_data, int size, float factor) {
    // 核心奥义：计算当前线程在全局网格里的唯一编号 (ID)
    // 想象你有成千上万个打工人，每个打工人只处理数组里的一个元素！
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 防止线程数超过了数组大小，越界访问
    if (idx < size) {
        d_data[idx] = d_data[idx] * factor; // 显卡上的每一个小核心，独立执行这一句
    }
}

int main() {
    int size = 1000;
    int bytes = size * sizeof(float);

    // ==========================================================
    // 2. 在 CPU (Host) 上准备数据
    // ==========================================================
    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < size; ++i) {
        h_data[i] = i * 1.0f; // 赋初值: 0, 1, 2, 3...
    }

    std::cout << "Before CUDA:" << std::endl;
    for (int i = 0; i < size; ++i) std::cout << h_data[i] << " ";
    std::cout << std::endl;

    // ==========================================================
    // 3. 在 GPU (Device) 上分配显存
    // ==========================================================
    float* d_data = nullptr;
    // cudaMalloc 负责在显卡上开辟空间，注意第一个参数是指针的地址
    cudaMalloc((void**)&d_data, bytes);

    // ==========================================================
    // 4. 把 CPU 的数据“搬运”到 GPU
    // 参数流向：目标地址 d_data, 源地址 h_data, 大小, 搬运方向 Host -> Device
    // ==========================================================
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // ==========================================================
    // 5. 启动 GPU 计算 (Launcher 发射器)
    // ==========================================================
    int threads_per_block = 256; // 设定一个 Block 里有 256 个线程（经验值）
    int blocks = (size + threads_per_block - 1) / threads_per_block; // 计算需要多少个 Block

    // <<<blocks, threads>>> 是 CUDA 独有的点火语法！
    array_multiply_kernel<<<blocks, threads_per_block>>>(d_data, size, 2.0f);

    // 强制 CPU 等待 GPU 计算完成（因为 GPU 发射是异步的，不等待 CPU 就直接跑下面代码了）
    cudaDeviceSynchronize();

    // ==========================================================
    // 6. 把 GPU 算好的结果“搬回” CPU
    // 方向变成了 Device -> Host
    // ==========================================================
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    std::cout << "After CUDA (Multiplied by 2):" << std::endl;
    for (int i = 0; i < size; ++i) std::cout << h_data[i] << " ";
    std::cout << std::endl;

    // ==========================================================
    // 7. 打扫战场释放内存
    // ==========================================================
    cudaFree(d_data); // GPU 显存释放
    free(h_data);     // CPU 内存释放

    return 0;
}