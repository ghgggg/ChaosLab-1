#include "core/vulkan/vk_allocator.hpp"
#include "core/vulkan/vk_tensor.hpp"

namespace chaos
{
    VkTensor::VkTensor() {}
    VkTensor::VkTensor(const Shape& shape, const Depth& depth, const Packing& packing, VkAllocator* allocator)
    {
        Create(shape, depth, packing, allocator);
    }
    VkTensor::~VkTensor() { Release(); }

    VkTensor::VkTensor(const VkTensor& t) :
        data(t.data), allocator(t.allocator), ref_cnt(t.ref_cnt), shape(t.shape), depth(t.depth), packing(t.packing), steps(t.steps)
    {
        if (ref_cnt) CHAOS_XADD(ref_cnt, 1);
    }
    VkTensor& VkTensor::operator=(const VkTensor& t)
    {
        if (this == &t)  return *this;

        if (t.ref_cnt) CHAOS_XADD(t.ref_cnt, 1);

        Release();

        data = t.data;
        ref_cnt = t.ref_cnt;
        allocator = t.allocator;

        shape = t.shape;
        depth = t.depth;
        packing = t.packing;
        steps = t.steps;

        return *this;
    }

	void VkTensor::Create(const Shape& _shape, const Depth& _depth, const Packing& _packing, VkAllocator* _allocator)
	{
		if (_shape == shape && _depth == depth && _packing == packing && _allocator == allocator) return;

		Release();

        shape = _shape;
        depth = _depth;
        packing = _packing;
        allocator = _allocator;

        steps = _shape.steps(); //Steps(shape, 1 * depth * packing);

        size_t total = (size_t)steps[0] * shape[0];
        if (total > 0)
        {
            size_t size = AlignSize(total * depth * packing, 4);

            data = allocator->FastMalloc(size);

            ref_cnt = (int*)((unsigned char*)data + offsetof(VkBufferMemory, ref_cnt));
            *ref_cnt = 1;
        }
	}

    void VkTensor::CreateLike(const Tensor& t, VkAllocator* allocator)
    {
        Create(t.shape, t.depth, t.packing, allocator);
    }

    void VkTensor::CreateLike(const VkTensor& t, VkAllocator* allocator)
    {
        Create(t.shape, t.depth, t.packing, allocator);
    }

	void VkTensor::Release()
	{
        if (ref_cnt && CHAOS_XADD(ref_cnt, -1) == 1)
        {
            if (allocator && data)
            {
                allocator->FastFree(data);
            }
        }

        data = nullptr;
        ref_cnt = nullptr;
	}

    Tensor VkTensor::Mapped() const
    {
        if (!allocator->mappable) return Tensor();
        return Tensor(shape, depth, packing, mapped_data(), steps);
    }
    void* VkTensor::mapped_data() const
    {
        if (!allocator->mappable) return nullptr;
        return (unsigned char*)data->mapped_data + data->offset;
    }
}