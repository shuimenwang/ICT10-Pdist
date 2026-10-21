// pdist_driver.cpp —— 自定义算子 Pdist 的 aclnn 调用驱动
// 用途：① 正确性（读入 x.bin，输出 y.bin 供 Python 与 PyTorch 对拍）
//       ② 性能（warmup + 多次迭代，报告 kernel 平均耗时与端到端平均耗时）
// 用法见 tests/run_custom.py
#include <acl/acl.h>
#include <aclnn_pdist.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define ACL_CHECK(expr)                                                                          \
    do {                                                                                         \
        aclError ret = (expr);                                                                    \
        if (ret != ACL_SUCCESS) {                                                                 \
            std::fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, \
                         static_cast<int>(ret));                                                  \
            return 1;                                                                             \
        }                                                                                         \
    } while (0)

namespace {
struct Args {
    std::string xPath;
    std::string yPath;
    int64_t n = 0;
    int64_t m = 0;
    double p = 2.0;
    bool fp16 = false;
    int iters = 20;
    int warmup = 5;
};

bool ParseArgs(int argc, char** argv, Args* a)
{
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (k == "--x") {
            const char* v = next(); if (!v) return false; a->xPath = v;
        } else if (k == "--y") {
            const char* v = next(); if (!v) return false; a->yPath = v;
        } else if (k == "--n") {
            const char* v = next(); if (!v) return false; a->n = std::atoll(v);
        } else if (k == "--m") {
            const char* v = next(); if (!v) return false; a->m = std::atoll(v);
        } else if (k == "--p") {
            const char* v = next(); if (!v) return false; a->p = std::atof(v);
        } else if (k == "--dtype") {
            const char* v = next(); if (!v) return false; a->fp16 = (std::string(v) == "fp16");
        } else if (k == "--iters") {
            const char* v = next(); if (!v) return false; a->iters = std::atoi(v);
        } else if (k == "--warmup") {
            const char* v = next(); if (!v) return false; a->warmup = std::atoi(v);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", k.c_str());
            return false;
        }
    }
    return a->n > 0 && a->m > 0 && !a->xPath.empty() && !a->yPath.empty();
}

template <typename T>
bool ReadBin(const std::string& path, std::vector<T>* buf, size_t count)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[ERROR] open %s failed\n", path.c_str());
        return false;
    }
    buf->resize(count);
    f.read(reinterpret_cast<char*>(buf->data()), static_cast<std::streamsize>(count * sizeof(T)));
    return f.gcount() == static_cast<std::streamsize>(count * sizeof(T));
}

bool WriteBin(const std::string& path, const void* data, size_t bytes)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return true;
}
}  // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!ParseArgs(argc, argv, &args)) {
        std::fprintf(stderr, "usage: %s --x in.bin --y out.bin --n N --m M [--p P] "
                             "[--dtype fp32|fp16] [--iters K] [--warmup W]\n", argv[0]);
        return 2;
    }
    const int64_t totalPairs = args.n * (args.n - 1) / 2;
    const size_t elemSize = args.fp16 ? 2 : 4;
    const size_t inBytes = static_cast<size_t>(args.n * args.m) * elemSize;
    const size_t outBytes = static_cast<size_t>(totalPairs) * elemSize;
    const aclDataType dtype = args.fp16 ? ACL_FLOAT16 : ACL_FLOAT;

    std::vector<uint8_t> hostIn;
    hostIn.resize(inBytes);
    {
        std::ifstream f(args.xPath, std::ios::binary);
        if (!f) { std::fprintf(stderr, "[ERROR] open %s failed\n", args.xPath.c_str()); return 1; }
        f.read(reinterpret_cast<char*>(hostIn.data()), static_cast<std::streamsize>(inBytes));
        if (f.gcount() != static_cast<std::streamsize>(inBytes)) {
            std::fprintf(stderr, "[ERROR] input size mismatch, need %zu bytes\n", inBytes);
            return 1;
        }
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    void* devIn = nullptr;
    void* devOut = nullptr;
    ACL_CHECK(aclrtMalloc(&devIn, inBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&devOut, outBytes > 0 ? outBytes : 32, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(devIn, inBytes, hostIn.data(), inBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemset(devOut, outBytes > 0 ? outBytes : 32, 0, outBytes > 0 ? outBytes : 32));

    int64_t xDims[2] = {args.n, args.m};
    int64_t xStrides[2] = {args.m, 1};
    int64_t yDims[1] = {totalPairs > 0 ? totalPairs : 1};
    int64_t yStrides[1] = {1};
    aclTensor* xTensor = aclCreateTensor(xDims, 2, dtype, xStrides, 0, ACL_FORMAT_ND, xDims, 2, devIn);
    aclTensor* yTensor = aclCreateTensor(yDims, 1, dtype, yStrides, 0, ACL_FORMAT_ND, yDims, 1, devOut);
    if (xTensor == nullptr || yTensor == nullptr) {
        std::fprintf(stderr, "[ERROR] aclCreateTensor failed\n");
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ACL_CHECK(aclnnPdistGetWorkspaceSize(xTensor, args.p, yTensor, &workspaceSize, &executor));
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // 预热
    for (int i = 0; i < args.warmup; ++i) {
        ACL_CHECK(aclnnPdist(workspace, workspaceSize, executor, stream));
    }
    ACL_CHECK(aclnnSynchronizeStream(stream));

    // 端到端：重复下发（含 Host 侧 aclnn 组装开销），仅 sync 一次
    std::vector<double> samples;
    samples.reserve(args.iters);
    for (int i = 0; i < args.iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACL_CHECK(aclnnPdist(workspace, workspaceSize, executor, stream));
        ACL_CHECK(aclnnSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::vector<uint8_t> hostOut(outBytes > 0 ? outBytes : 1);
    ACL_CHECK(aclrtMemcpy(hostOut.data(), outBytes, devOut, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteBin(args.yPath, hostOut.data(), outBytes)) {
        std::fprintf(stderr, "[ERROR] write %s failed\n", args.yPath.c_str());
        return 1;
    }

    double sum = 0.0;
    double best = 1e18;
    for (double v : samples) { sum += v; best = std::fmin(best, v); }
    const double mean = sum / static_cast<double>(samples.size());
    // 机器可读输出，供 Python 汇总
    std::printf("RESULT shape=%ldx%ld p=%g dtype=%s iters=%d mean_ms=%.6f best_ms=%.6f\n",
                static_cast<long>(args.n), static_cast<long>(args.m), args.p,
                args.fp16 ? "fp16" : "fp32", args.iters, mean, best);

    if (workspace != nullptr) { aclrtFree(workspace); }
    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    aclrtFree(devIn);
    aclrtFree(devOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
