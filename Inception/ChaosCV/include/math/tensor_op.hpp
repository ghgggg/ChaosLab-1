#pragma once

#include "core/core.hpp"
#include "core/vec.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	class CHAOS_API InputArray
	{
	public:
		enum KindFlag 
		{
			NONE,
			TENSOR,
		};
		InputArray();
		InputArray(const Tensor& data);
		
		Tensor GetTensor() const;

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
	};

	class CHAOS_API InputOutputArray : public OutputArray
	{
	public:
		InputOutputArray();
		InputOutputArray(Tensor& data);
	};

	CHAOS_API const InputOutputArray& noArray();

	CHAOS_API void SetIdentity(const InputOutputArray& src, double val = 1.);
	CHAOS_API void Transpose(const InputArray& src, const OutputArray& dst);
	CHAOS_API void Permute(const Tensor& src, Tensor& dst, const Vec<uint>& orders);

	CHAOS_API void Permute(const InputArray& src, const OutputArray& dst, const Vec<uint>& orders);
}