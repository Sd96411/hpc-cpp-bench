# hpc-cpp-bench

**English summary.** Three small C++/OpenMP benchmarks on an Intel i7-9750H: a naive FP64 DGEMM baseline, a blocked i-k-j DGEMM with a grid search over block size and thread count (a proxy for tuning HPL's `NB`), and an OpenMP `target` offload test. Blocking plus loop reordering took 2000² DGEMM from **1.52 to 15.66 GFLOPS (~10×)**; 8 threads beat 12 on this 6-core CPU. The offload test showed that this MinGW g++ build silently falls back to the host (0.96×), so it is CPU vs. CPU, and the README says so. Raw outputs are in `results/`. The detailed write-up below is in Traditional Chinese.

---

三支 C++/OpenMP 小型 benchmark，用來量本機 CPU 的 FP64 矩陣乘法表現、為 HPL 的 `NB` 和執行緒數做網格搜尋，並測試 OpenMP target offload。在 i7-9750H 上，只是把迴圈順序改成 i-k-j 並分塊，2000² DGEMM 就從 **1.52 GFLOPS 提升到 15.66 GFLOPS（約 10×）**；OpenMP target 在 MinGW g++ 下沒有 offload 到 GPU，實際上是在 CPU 上執行（0.96×）。

所有數字都來自 2026-10-22 的單次實跑，原始輸出放在 [`results/2026-10-22_i7-9750H/`](results/2026-10-22_i7-9750H/)。

## 測試環境

| 項目 | 數值 |
|---|---|
| CPU | Intel Core i7-9750H @ 2.60 GHz，6 核 12 緒 |
| RAM | 約 24 GB |
| OS | Windows 11 |
| 編譯器 | g++ 15.2.0（MinGW-Builds, x86_64-posix-seh），`-O3 -fopenmp` |
| GPU | GeForce GTX 1650（有裝，但這三支程式都沒有實際用到，見下面說明） |

## 程式與結果

### `src/matmul_bench.cpp`：naive DGEMM baseline

- 2000×2000 FP64，`C = A·B`，三層迴圈 i-j-k，`#pragma omp parallel for collapse(2)`，使用全部 12 個 thread。
- 最內層 `B[k*N + j]` 每次跨 N 個 double 讀取，cache 和 SIMD 幾乎都用不上。

| 耗時 | 效能 |
|---:|---:|
| 10.51 s | **1.52 GFLOPS** |

### `src/hpl_local_tuner.cpp`：分塊 DGEMM 網格搜尋（幫 HPL 挑 NB）

- 同樣是 2000² FP64，但改成 **NB×NB 分塊 + i-k-j 迴圈順序**（最內層連續讀 `B` 和 `C`，可以向量化）。OpenMP 用 `collapse(2)` 平行化 (sj, si) 兩層 tile 迴圈，每個 thread 負責自己的 C tile，所以不會有 race。
- 掃描 NB ∈ {16, 32, 64, 128, 256} × threads ∈ {1, 2, 4, 8, 12}，共 25 組，輸出前 5 名。
- 注意：這支程式**不是 HPL 本身**（沒有做 LU 分解），只是用分塊矩陣乘法當代理指標，幫 `HPL.dat` 挑 NB 和執行緒數。

| 排名 | NB | Threads | 耗時 (s) | GFLOPS |
|---:|---:|---:|---:|---:|
| 1 | 128 | 8 | 1.0216 | **15.66** |
| 2 | 64 | 8 | 1.0300 | 15.53 |
| 3 | 32 | 8 | 1.0475 | 15.27 |
| 4 | 256 | 12 | 1.0783 | 14.84 |
| 5 | 16 | 12 | 1.0784 | 14.84 |

觀察：

- 比 naive 快約 10.3×。這個差距同時來自迴圈順序和分塊，沒有做 ablation，所以**兩者各自貢獻多少未確認**。
- NB 從 32 到 128 的差距不到 3%，在單次量測的雜訊範圍內，不能說 128 一定比較好。
- 8 threads 比 12 threads 好。推測是 6 顆實體核心上開 hyper-threading 後，兩個 thread 共用 FPU 和 L1/L2，或是 thermal 降頻（**推測，未驗證**）。
- 輸出只列前 5 名，完整的 25 組結果沒有存下來。

### `src/omp_cpu_vs_gpu.cpp`：OpenMP CPU 平行 vs `omp target` offload

- 對 5,000 萬個 double 做 `C[i] = A[i]*B[i] + A[i]`（3 個陣列共約 1.2 GB，memory-bound）。先跑 CPU `parallel for`，再跑 `#pragma omp target teams distribute parallel for`（含 `map` 資料搬移），計算兩者的比值。

| 項目 | 耗時 (s) |
|---|---:|
| CPU OpenMP | 0.1208 |
| `omp target` | 0.1262 |
| 比值 | **0.96×** |

- **這不是 GPU 的結果**。這版 MinGW g++ 沒有編進 offload target（`g++ -v` 裡沒有 `OFFLOAD_TARGET_NAMES`），所以 target 區塊退回 host 執行，等於是 CPU 對 CPU。
- 原始資料夾裡舊的 `.exe` 是用 MSVC 編的；MSVC 的 `/openmp` 不支援 `target`，所以那個執行檔同樣量不到 GPU。
- CPU 的部分：讀 2 個陣列、寫 1 個陣列，約 1.2 GB / 0.1208 s ≈ 9.9 GB/s（沒有計入 write-allocate）。
- 要真的量到 GPU，需要支援 nvptx offload 的編譯器（例如 Linux 上的 `nvc++ -mp=gpu` 或 clang `-fopenmp-targets=nvptx64`）。即使 offload 成功，計時區間也包含約 1.2 GB 的 PCIe 傳輸，對這種只做 2 FLOP/element 的 kernel 很可能比 CPU 慢（**推測，未實測**）。

## 編譯與執行

```powershell
g++ -O3 -fopenmp src/matmul_bench.cpp    -o matmul_bench.exe
g++ -O3 -fopenmp src/hpl_local_tuner.cpp -o hpl_local_tuner.exe
g++ -O3 -fopenmp src/omp_cpu_vs_gpu.cpp  -o omp_cpu_vs_gpu.exe
.\matmul_bench.exe; .\hpl_local_tuner.exe; .\omp_cpu_vs_gpu.exe
```

用 `OMP_NUM_THREADS` 可以控制 `matmul_bench` 的 thread 數；`hpl_local_tuner` 會自己掃 thread 數。Windows 下程式會切到 UTF-8 console（`SetConsoleOutputCP(65001)`），Linux 不受影響。

## 已知限制

- 每支都只跑一次，沒有暖機、沒有重複取中位數，也沒有做結果正確性檢查。
- `-O3` 沒有加 `-march=native`，AVX2/FMA 可能沒有被完整利用。
- 這三支都沒有做 microkernel 和 packing，跟 OpenBLAS/MKL 這類 BLAS 實作的差距很大（實際差距沒有量過）。
