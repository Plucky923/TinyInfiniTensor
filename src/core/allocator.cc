#include "core/allocator.h"

#include <utility>

namespace infini
{
Allocator::Allocator(Runtime runtime) : runtime(runtime)
{
    used = 0;
    peak = 0;
    ptr = nullptr;

    // 'alignment' defaults to sizeof(uint64_t), because it is the length of
    // the longest data type currently supported by the DataType field of
    // the tensor
    alignment = sizeof(uint64_t);
    free_blocks[0] = 0;
}

Allocator::~Allocator()
{
    if (this->ptr != nullptr)
    {
        runtime->dealloc(this->ptr);
    }
}

size_t Allocator::alloc(size_t size)
{
    IT_ASSERT(this->ptr == nullptr);
    // pad the size to the multiple of alignment
    size = this->getAlignedSize(size);

    // =================================== 作业
    // ===================================
    // TODO: 设计一个算法来分配内存，返回起始地址偏移量
    // =================================== 作业
    // ===================================
    
    // 因为要返回的是起始地址偏移量，所以要先将 used 加上 size
    this->used += size;

    for(auto it = free_blocks.begin(); it != free_blocks.end(); it++)
    {
        if(it->second >= size)
        {
            size_t addr = it->first;
            size_t block_size = it->second;
            free_blocks.erase(it);
            if(block_size > size)
            {
                free_blocks[addr + size] = block_size - size;
            }
            return addr;
        }
    }
    // 因为一开始的free_blocks 里面啥都没存，所以是通过peak来确定起始地址偏移量
    // 因为一开始就是从 0 开始分配的，所以这里直接返回 used - size
    this->peak = std::max(this->peak, this->used);
    return peak - size;
}

void Allocator::free(size_t addr, size_t size)
{
    IT_ASSERT(this->ptr == nullptr);
    size = getAlignedSize(size);

    // =================================== 作业
    // ===================================
    // TODO: 设计一个算法来回收内存
    // =================================== 作业
    // ===================================

    // 在回收内存之后，会产生新的free_blocks，所以要更新free_blocks
    this->used -= size; // 回收内存，所以used要减去size

    // 如果有free block，那么就要合并free block
    for(auto it = free_blocks.begin(); it != free_blocks.end(); it++)
    {
        // 如果找到了一个free_block，它的起始地址加上大小等于要回收的地址，那么就可以合并
        // 把这个free_block的大小加上要回收的大小，也就是把要释放得内存和原有的free_block合并
        // 要释放的内存在后，free_block在前
        if(it->first + it->second == addr)
        {
            free_blocks[it->first] += size;
            return;
        }
        // 如果找到了一个free_block，它的起始地址等于要回收的地址加上大小，那么也可以合并
        // 也就是把要释放得内存和原有的free_block合并，要释放的内存在前，free_block在后
        else if(it->first == addr + size)
        {
            free_blocks[addr] = size + it->second;
            free_blocks.erase(it);
            return;
        }
    }
    // 如果没有找到可以合并的free block，那么直接插入新的free block
    free_blocks[addr] = size;
}

void *Allocator::getPtr()
{
    if (this->ptr == nullptr)
    {
        this->ptr = runtime->alloc(this->peak);
        printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
    }
    return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size)
{
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info()
{
    std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak << std::endl;
}
} // namespace infini
