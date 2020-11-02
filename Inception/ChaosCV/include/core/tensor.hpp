#pragma once

#include "core.hpp"
#include "vec.hpp"

namespace chaos
{
	class OutputArray;
	class VkTensor;
	class CHAOS_API Tensor
	{
	public:
		Tensor() = default;
		Tensor(const Shape& shape, const Depth& depth, const Packing& packing = Packing::CHW, Allocator* allocator = nullptr);
		Tensor(const Shape& shape, const Depth& depth, const Packing& packing, void* data, const Steps& steps = Steps());

		~Tensor();

		Tensor(const Tensor& t);
		Tensor& operator=(const Tensor& t);

		void Create(const Shape& _shape, const Steps& steps, const Depth& _depth, const Packing& _packing, Allocator* _allocator);
		void CreateLike(const VkTensor& t, Allocator* allocator);

		//void CopyTo(Tensor& t) const;
		void CopyTo(const OutputArray& arr, Allocator* allocator = nullptr) const;
		Tensor Clone(Allocator* allocator = nullptr) const;

		/// <summary>Release the tensor, ref_cnt--</summary>
		void Release();

		bool empty() const noexcept { return data == nullptr || shape.vol() == 0; }
		bool continua() const noexcept { return shape.vol() == ((size_t)shape[0] * steps[0]); }

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
		Steps steps;
	};


	class CHAOS_API InputArray
	{
	public:
		enum KindFlag
		{
			NONE,
			VEC,
			TENSOR,
			VECTOR_TENSOR,
		};

		InputArray();
		InputArray(const Tensor& data);

		Tensor GetTensor() const;

		bool IsTensor() const;
	protected:
		int flag;
		void* obj;

		void Init(int _flag, const void* _obj);
	};

	class CHAOS_API OutputArray : public InputArray
	{
	public:
		OutputArray();
		OutputArray(Tensor& data);

		bool Needed() const;
		void Create(const Shape& shape, const Depth& depth, const Packing& packing, Allocator* allocator = nullptr) const;
		void Release() const;
		Tensor& GetTensorRef() const;
	};

	class CHAOS_API InputOutputArray : public OutputArray
	{
	public:
		InputOutputArray();
		InputOutputArray(Tensor& data);
	};

	CHAOS_API const InputOutputArray& noArray();
}