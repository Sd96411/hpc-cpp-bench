#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#endif

struct Result {
    int nb;
    int threads;
    double time_sec;
    double gflops;
};

int main() {
    #ifdef _WIN32
    SetConsoleOutputCP(65001);
    #endif

    int N = 2000; // 用小矩陣進行快速網格搜尋 (Fast Grid Search)
    std::cout << "==================================================" << std::endl;
    std::cout << "       自動化 HPL 網格搜尋器 (Auto Grid Search)   " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "測試固定矩陣大小 N = " << N << " x " << N << std::endl;
    std::cout << "正在自動測試不同的 [NB] 與 [Threads] 組合...\n" << std::endl;

    std::vector<int> nb_candidates = {16, 32, 64, 128, 256};
    std::vector<int> thread_candidates = {1, 2, 4, 8, 12};
    std::vector<Result> results;

    std::vector<double> A(N * N, 1.0);
    std::vector<double> B(N * N, 2.0);

    for (int threads : thread_candidates) {
        omp_set_num_threads(threads);
        for (int NB : nb_candidates) {
            std::vector<double> C(N * N, 0.0);

            auto start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for collapse(2) schedule(static)
            for (int sj = 0; sj < N; sj += NB) {
                for (int si = 0; si < N; si += NB) {
                    for (int sk = 0; sk < N; sk += NB) {
                        int i_end = std::min(si + NB, N);
                        int j_end = std::min(sj + NB, N);
                        int k_end = std::min(sk + NB, N);

                        for (int i = si; i < i_end; i++) {
                            for (int k = sk; k < k_end; k++) {
                                double r = A[i * N + k];
                                for (int j = sj; j < j_end; j++) {
                                    C[i * N + j] += r * B[k * N + j];
                                }
                            }
                        }
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            double seconds = diff.count();
            double flops = 2.0 * std::pow(N, 3);
            double gflops = (flops / seconds) / 1e9;

            results.push_back({NB, threads, seconds, gflops});
        }
    }

    // 依照 GFLOPS 由大到小排序
    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
        return a.gflops > b.gflops;
    });

    std::cout << "==================================================================" << std::endl;
    std::cout << "                    最佳參數排行榜 (Top 5 Results)               " << std::endl;
    std::cout << "==================================================================" << std::endl;
    std::cout << std::setw(10) << "排名" 
              << std::setw(12) << "NB (區塊)" 
              << std::setw(15) << "Threads (核心)" 
              << std::setw(15) << "耗時 (秒)" 
              << std::setw(16) << "效能 (GFLOPS)" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;

    for (size_t i = 0; i < std::min(size_t(5), results.size()); i++) {
        std::cout << std::setw(8) << "Rank " << (i + 1)
                  << std::setw(12) << results[i].nb
                  << std::setw(15) << results[i].threads
                  << std::setw(15) << std::fixed << std::setprecision(4) << results[i].time_sec
                  << std::setw(16) << std::fixed << std::setprecision(2) << results[i].gflops << std::endl;
    }
    std::cout << "==================================================================" << std::endl;

    return 0;
}
