#include "core/runtime.h"
#include "core/blob.h"
#include "core/kernel.h"
#include "core/graph.h"
#include "core/kernel.h"
#include <chrono>
#include <cstring>
#include <memory>
namespace infini
{
    void NativeCpuRuntimeObj::run(const Graph &graph) const
    {
        const auto &kernelRegistry = KernelRegistry::getInstance();

        for (auto &op : graph->getOperators())
        {
            auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
            Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
            kernel->compute(op, this);
        }
    }

    string NativeCpuRuntimeObj::toString() const { return "CPU Runtime"; }

    void NativeCpuRuntimeObj::dealloc(void *ptr)
    {
        return free(ptr);
    }

// alloc 分配了一段内存，返回指向这段内存的指针，allocator 会用这个指针来管理内存
    void *NativeCpuRuntimeObj::alloc(size_t size)
    {
        return calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t),
                      sizeof(uint64_t));
    }

} // namespace infini
