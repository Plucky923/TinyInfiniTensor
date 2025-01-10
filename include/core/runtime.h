#pragma once
#include "core/common.h"
#include "core/op_type.h"
#include "core/ref.h"

namespace infini
{
  class TensorObj;
  class OperatorObj;
  class GraphObj;
  class RuntimeObj;
  class BlobObj;

  using Tensor = Ref<TensorObj>;
  using Operator = Ref<OperatorObj>;
  using Graph = Ref<GraphObj>;
  using Runtime = Ref<RuntimeObj>;
  using Blob = Ref<BlobObj>;

  using TensorVec = vector<Tensor>;
  using OpVec = vector<Operator>;

  enum class Device
  {
    CPU = 1
  };

// 基类
  class RuntimeObj : public std::enable_shared_from_this<RuntimeObj>
  {
  protected:
    Device device;

  public:
    explicit RuntimeObj(Device device)
        : device(device) {}
    RuntimeObj(RuntimeObj &other) = delete;
    RuntimeObj &operator=(RuntimeObj const &) = delete;
    virtual ~RuntimeObj() {}

    virtual void run(const Graph &graph) const = 0;
    virtual void *alloc(size_t size) = 0;
    virtual void dealloc(void *ptr) = 0;

    bool isCpu() const
    {
      return true;
    }

    virtual string toString() const = 0;
  };

// 继承Runtime类，并传递Device::CPU作为参数（关联的设备）
  class NativeCpuRuntimeObj : public RuntimeObj
  {
  public:
    NativeCpuRuntimeObj() : RuntimeObj(Device::CPU) {}

    static Ref<NativeCpuRuntimeObj> &getInstance()
    {
      static Ref<NativeCpuRuntimeObj> instance =
          make_ref<NativeCpuRuntimeObj>();
      return instance;
    }
    void dealloc(void *ptr) override;
    void run(const Graph &graph) const override;
    void *alloc(size_t size) override; // 重写对应的虚函数，代码在对应的cpp文件中
    string toString() const override;
  };

} // namespace infini
