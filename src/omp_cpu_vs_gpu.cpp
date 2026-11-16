#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#endif

int main() {
    #ifdef _WIN32
    SetConsoleOutputCP(65001);
    #endif

    int N = 50000000;
    std::cout << "==================================================" << std::endl;
    std::cout << "      OpenMP CPU vs GPU Target Performance PK     " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Data Size N = " << N << " elements" << std::endl;

    std::vector<double> A(N, 1.5);
    std::vector<double> B(N, 2.5);
    std::vector<double> C(N, 0.0);

    // 1. CPU OpenMP Multi-Core Parallel
    auto start_cpu = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * B[i] + A[i];
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
    std::cout << "\n[1] CPU OpenMP Multi-Core Time: " << diff_cpu.count() << " s" << std::endl;

    std::fill(C.begin(), C.end(), 0.0);

    // 2. GPU OpenMP Target Offload
    double* pA = A.data();
    double* pB = B.data();
    double* pC = C.data();

    // Warmup
    #pragma omp target map(to: pA[0:100], pB[0:100]) map(from: pC[0:100])
    #pragma omp teams distribute parallel for
    for (int i = 0; i < 100; i++) {
        pC[i] = pA[i] * pB[i] + pA[i];
    }

    auto start_gpu = std::chrono::high_resolution_clock::now();

    #pragma omp target map(to: pA[0:N], pB[0:N]) map(from: pC[0:N])
    #pragma omp teams distribute parallel for
    for (int i = 0; i < N; i++) {
        pC[i] = pA[i] * pB[i] + pA[i];
    }

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_gpu = end_gpu - start_gpu;
    std::cout << "[2] GPU OpenMP Target Time    : " << diff_gpu.count() << " s" << std::endl;

    // 3. Speedup
    double speedup = diff_cpu.count() / diff_gpu.count();
    std::cout << "\n==================================================" << std::endl;
    std::cout << "GPU Speedup vs CPU: " << speedup << " x" << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
