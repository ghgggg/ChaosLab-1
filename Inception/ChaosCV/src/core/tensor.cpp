#include "core/tensor.hpp"

namespace chaos
{
	

	Tensor::Tensor(const Shape& shape, const Depth& depth, const Packing& packing, Allocator* allocator)
	{
		Create(shape, depth, packing, allocator);
	}
	Tensor::Tensor(const Shape& shape, const Depth& depth, const Packing& packing, void* data, Allocator* allocator)
		: data(data), allocator(allocator), shape(shape), depth(depth), packing(packing) {}

	Tensor::~Tensor() { Release(); }

	Tensor::Tensor(const Tensor& t) :
		data(t.data), allocator(t.allocator), ref_cnt(t.ref_cnt), shape(t.shape), depth(t.depth), packing(t.packing)
	{
		if (ref_cnt) CHAOS_XADD(ref_cnt, 1);
	}
	Tensor& Tensor::operator=(const Tensor& t)
	{
		if (this == &t) return *this;

		if (t.ref_cnt) CHAOS_XADD(t.ref_cnt, 1);

		Release();

		data = t.data;
		ref_cnt = t.ref_cnt;
		allocator = t.allocator;

		shape = t.shape;
		depth = t.depth;
		packing = t.packing;

		return *this;
	}

	void Tensor::Create(const Shape& _shape, const Depth& _depth, const Packing& _packing, Allocator* _allocator)
	{
		if (_shape == shape && _depth == depth && _packing == packing && _allocator == allocator) return;

		Release();

		shape = _shape;
		depth = _depth;
		packing = _packing;
		allocator = _allocator;

		//elem_size = depth * packing;
		size_t vol = shape.vol();
		if (vol > 0)
		{
			size_t size = AlignSize(vol * depth * packing, 4);

			if (allocator)
				data = allocator->FastMalloc(size + sizeof(*ref_cnt));
			else
				data = FastMalloc(size + sizeof(*ref_cnt));

			ref_cnt = (int*)(((uchar*)data) + size);
			*ref_cnt = 1;
		}
	}

	void Tensor::Release()
	{
		if (ref_cnt && CHAOS_XADD(ref_cnt, -1) == 1)
		{
			if (allocator)
				allocator->FastFree(data);
			else
				FastFree(data);
		}

		data = nullptr;
		ref_cnt = nullptr;
	}

	void Tensor::CopyTo(Tensor& t) const
	{
		memcpy(t.data, data, shape.vol() * depth * packing);
	}
	//Tensor Tensor::channel(size_t ch) const
	//{
	//}
	//Tensor Tensor::row(size_t r) const
	//{
	//}
}