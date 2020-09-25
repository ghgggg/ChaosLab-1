#pragma once

#include "core.hpp"
#include "vec.hpp"

namespace chaos
{
	class CHAOS_API Tensor
	{
	public:
		Tensor() = default;
		Tensor(const Shape& shape, const Depth& depth, const Packing& packing = Packing::CHW, Allocator* allocator = nullptr);
		Tensor(const Shape& shape, const Depth& depth, const Packing& packing, void* data, Allocator* allocator = nullptr);

		~Tensor();

		Tensor(const Tensor& t);
		Tensor& operator=(const Tensor& t);

		void Create(const Shape& _shape, const Depth& _depth, const Packing& _packing, Allocator* _allocator);

		void CopyTo(Tensor& t) const;

		/// <summary>Release the tensor, ref_cnt--</summary>
		void Release();

		bool empty() const noexcept { return data == nullptr || shape.vol() == 0; }

		/// <summary>ref_cnt++</summary>
		void AddRef() noexcept { if(ref_cnt) CHAOS_XADD(ref_cnt, 1); }

		template<class Type, std::enable_if_t<std::is_arithmetic_v<Type> or std::is_void_v<Type>, bool> = true> 
		operator const Type* () const { return (const Type*)data; }
		template<class Type, std::enable_if_t<std::is_arithmetic_v<Type> or std::is_void_v<Type>, bool> = true> 
		operator Type* () { return (Type*)data; }

		const float& operator[](size_t idx) const noexcept { return ((float*)data)[idx]; }
		float& operator[](size_t idx) noexcept { return ((float*)data)[idx]; }

		void* data = nullptr;
		Allocator* allocator = nullptr;

		// pointer to the reference counter
		// when points to user-allocated data, the pointer is NULL
		int* ref_cnt = nullptr;

		Depth depth = Depth::D1;
		Packing packing = Packing::CHW;
		Shape shape;

		// element_size = depth * packing
	};
}