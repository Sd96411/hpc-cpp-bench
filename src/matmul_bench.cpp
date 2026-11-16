#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#endif

int main() {
    // Windows 控制台自動切換為 UTF-8 碼頁 (65001)，徹底解決繁體中文亂碼問題
    #ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    #endif

    int N = 2000; // 矩陣大小 2000 x 2000
    std::cout << "=== 本機 CPU GFLOPS 壓測程式 ===" << std::endl;
    std::cout << "矩陣大小: " << N << " x " << N << std::endl;
    std::cout << "使用核心數: " << omp_get_max_threads() << " threads" << std::endl;
    std::cout << "正在進行運算，請聽風扇聲音..." << std::endl;

    // 配置記憶體
    std::vector<double> A(N * N, 1.0);
    std::vector<double> B(N * N, 2.0);
    std::vector<double> C(N * N, 0.0);

    auto start = std::chrono::high_resolution_clock::now();

    // OpenMP 多執行緒平行矩陣相乘
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double seconds = diff.count();

    // 計算 2 * N^3 次浮點運算 (FLOPs)
    double flops = 2.0 * N * N * N;
    double gflops = (flops / seconds) / 1e9;

    std::cout << "\n=== 測試完成！===" << std::endl;
    std::cout << "耗時: " << seconds << " 秒" << std::endl;
    std::cout << "效能: " << gflops << " GFLOPS" << std::endl;

    return 0;
}
