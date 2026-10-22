// Phase 7, file 2a: how fast this card can actually move memory.
//
// Every "% of peak" figure in the lab so far divides by 624 GB/s - the
// datasheet number, which no kernel reaches. This measures the real ceiling.
//
// It is also the smallest useful HIP program, and reads almost exactly like
// CUDA: __global__, blockIdx/threadIdx, hipMalloc, hipMemcpy, hipEvent. Swap
// the "hip" prefixes for "cuda" and it compiles with nvcc. That is the point
// of HIP - CUDA-shaped source that targets AMD (and, through a thin header,
// NVIDIA). llama.cpp's ROCm backend is built exactly this way.
//
// Probes, each over a range of buffer sizes:
//   vram_read    read every byte once, write nothing. What decode does to the
//                weights, so the most relevant ceiling for this lab.
//   vram_copy    read + write. The classic STREAM "copy"; counts both bytes.
//   d2d_memcpy   hipMemcpy device -> device: the runtime's own copy kernel.
//   h2d_pinned   host -> device over PCIe, from pinned (page-locked) memory.
//   d2h_pinned   device -> host. Both matter for CPU/GPU offload, where
//                activations cross the bus every layer boundary.
//
// Small buffers fit in the 4 MB L2 or the 64 MB Infinity Cache and report
// speeds above VRAM. That is real, and it is why the size sweep exists - only
// the large-buffer plateau is the VRAM figure.
//
// Build and run (WSL, inside ~/rocm-env):
//   hipcc -O3 --offload-arch=gfx1101 -o bandwidth 02_bandwidth.hip
//   ./bandwidth            # one JSON object per line
//
// Driven by 02_ceilings.py, which builds it, runs it and stores the results.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                        \
    do {                                                                   \
        hipError_t err_ = (call);                                          \
        if (err_ != hipSuccess) {                                          \
            fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorString(err_)); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// Grid-stride loops: a fixed grid walks any buffer size. float4 = 16-byte
// loads, the widest a thread can issue, so the memory system rather than the
// instruction count is the limit.
__global__ void read_kernel(const float4* __restrict__ in, float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; i < n; i += stride) {
        float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // The buffer is zeroed, so this never writes - but the compiler cannot know
    // that, so it cannot delete the loads as dead code.
    if (acc == 1234.5f) out[0] = acc;
}

__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out,
                            size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = in[i];
}

static const int REPEATS = 7;
static const int THREADS = 256;
static const int BLOCKS = 30 * 64;   // 30 WGPs (60 CUs) x plenty of waves each

// Time `iters` launches of fn as one batch; return GB/s for each of REPEATS.
template <typename F>
static std::vector<double> time_gbs(F fn, size_t bytes_per_iter, int iters) {
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    for (int w = 0; w < 3; w++) fn();   // warm-up: clocks up, caches primed
    CHECK(hipDeviceSynchronize());
    std::vector<double> out;
    for (int r = 0; r < REPEATS; r++) {
        CHECK(hipEventRecord(a));
        for (int k = 0; k < iters; k++) fn();
        CHECK(hipEventRecord(b));
        CHECK(hipEventSynchronize(b));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, a, b));
        out.push_back((double)bytes_per_iter * iters / (ms / 1e3) / 1e9);
    }
    CHECK(hipEventDestroy(a));
    CHECK(hipEventDestroy(b));
    return out;
}

static void report(const char* probe, size_t bytes, std::vector<double> v) {
    std::sort(v.begin(), v.end());
    printf("{\"probe\": \"%s\", \"bytes\": %zu, \"best\": %.2f, \"median\": %.2f, "
           "\"repeats\": %zu}\n", probe, bytes, v.back(), v[v.size() / 2], v.size());
    fflush(stdout);
}

// Enough iterations that each timed batch moves ~4 GB: long enough to hold
// boost clocks, short enough to keep the whole sweep under a minute.
static int iters_for(size_t bytes) {
    return (int)std::max<size_t>(10, (4ull << 30) / bytes);
}

int main() {
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    printf("{\"device\": \"%s\", \"arch\": \"%s\", \"multiprocessors\": %d, "
           "\"l2_bytes\": %d, \"bus_bits\": %d, \"mem_clock_khz\": %d}\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount, prop.l2CacheSize,
           prop.memoryBusWidth, prop.memoryClockRate);

    const size_t MB = 1ull << 20;
    const size_t sizes[] = {1 * MB, 4 * MB, 16 * MB, 32 * MB, 64 * MB, 128 * MB,
                            256 * MB, 512 * MB, 1024 * MB, 2048 * MB};
    const size_t max_bytes = 2048 * MB;

    float4 *a, *b;
    float* sink;
    CHECK(hipMalloc(&a, max_bytes));
    CHECK(hipMalloc(&b, max_bytes));
    CHECK(hipMalloc(&sink, sizeof(float)));
    CHECK(hipMemset(a, 0, max_bytes));
    CHECK(hipMemset(b, 0, max_bytes));

    for (size_t bytes : sizes) {
        size_t n = bytes / sizeof(float4);
        int iters = iters_for(bytes);
        report("vram_read", bytes, time_gbs([&] {
            read_kernel<<<BLOCKS, THREADS>>>(a, sink, n);
        }, bytes, iters));
        // Copy moves each byte twice (read + write), and is counted that way.
        report("vram_copy", bytes, time_gbs([&] {
            copy_kernel<<<BLOCKS, THREADS>>>(a, b, n);
        }, 2 * bytes, iters));
        report("d2d_memcpy", bytes, time_gbs([&] {
            CHECK(hipMemcpyAsync(b, a, bytes, hipMemcpyDeviceToDevice));
        }, 2 * bytes, iters));
    }

    // PCIe. Pinned host memory, because pageable memory adds a CPU-side staging
    // copy and measures that instead of the bus.
    const size_t pcie_sizes[] = {16 * MB, 256 * MB, 1024 * MB};
    void* host;
    CHECK(hipHostMalloc(&host, 1024 * MB, hipHostMallocDefault));
    for (size_t bytes : pcie_sizes) {
        int iters = std::max(5, (int)((2ull << 30) / bytes));
        report("h2d_pinned", bytes, time_gbs([&] {
            CHECK(hipMemcpyAsync(a, host, bytes, hipMemcpyHostToDevice));
        }, bytes, iters));
        report("d2h_pinned", bytes, time_gbs([&] {
            CHECK(hipMemcpyAsync(host, a, bytes, hipMemcpyDeviceToHost));
        }, bytes, iters));
    }

    CHECK(hipHostFree(host));
    CHECK(hipFree(a));
    CHECK(hipFree(b));
    CHECK(hipFree(sink));
    return 0;
}
