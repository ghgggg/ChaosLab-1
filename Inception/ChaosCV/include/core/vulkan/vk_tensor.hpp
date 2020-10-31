#pragma once

#include "core/tensor.hpp"

#include "vk_allocator.hpp"

#include <vulkan/vulkan.hpp>

namespace chaos
{
	class VkBufferMemory;
	class VkAllocator;
	class CHAOS_API VkTensor
	{
	public:
		VkTensor();
		VkTensor(const Shape& shape, const Depth& depth, const Packing& packing, VkAllocator* allocator);
		~VkTensor();

		VkTensor(const VkTensor& t);
		VkTensor& operator=(const VkTensor& t);

		void Create(const Shape& shape, const Depth& depth, const Packing& packing, VkAllocator* allocator);
		// allocate like
		void CreateLike(const Tensor& t, VkAllocator* allocator);
		// allocate like
		void CreateLike(const VkTensor& t, VkAllocator* allocator);

		void Release();

		// mapped
		Tensor Mapped() const;
		void* mapped_data() const;

		bool empty() const noexcept { return data == nullptr || shape.vol() == 0; }
		bool continua() const noexcept { return shape.vol() == ((size_t)shape[0] * steps[0]); }

		/// <summary>ref_cnt++</summary>
		void AddRef() noexcept { if (ref_cnt) CHAOS_XADD(ref_cnt, 1); }

		// low-level reference
		VkBuffer buffer() const noexcept { return data->buffer; }
		size_t buffer_offset() const noexcept { return data->offset; }
		size_t buffer_capacity() const noexcept { return data->capacity; }

		// device buffer
		VkBufferMemory* data = nullptr;

		// pointer to the reference counter
		// when points to user-allocated data, the pointer is NULL
		int* ref_cnt = nullptr;

		// the allocator
		VkAllocator* allocator = nullptr;

		Depth depth = Depth::D1;
		Packing packing = Packing::CHW;

		Shape shape;
		Steps steps;
	};
}