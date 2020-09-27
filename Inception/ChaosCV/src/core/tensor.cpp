#include "core/tensor.hpp"

namespace chaos
{
	

	Tensor::Tensor(const Shape& shape, const Depth& depth, const Packing& packing, Allocator* allocator)
	{
		Create(shape, depth, packing, allocator);
	}
	Tensor::Tensor(const Shape& shape, const Depth& depth, const Packing& packing, void* data, const Steps& _steps)
		: data(data), allocator(nullptr), shape(shape), depth(depth), packing(packing), steps(_steps)
	{
	}

	Tensor::~Tensor() { Release(); }

	Tensor::Tensor(const Tensor& t) :
		data(t.data), allocator(t.allocator), ref_cnt(t.ref_cnt), shape(t.shape), depth(t.depth), packing(t.packing), steps(t.steps)
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

		steps = Steps(shape.size());
		for (int64 i = shape.size() -2; i >= 0; i--)
		{
			steps[i] = shape[i + 1] * steps[i + 1];
		}
		steps = steps * (uint)(1 * depth * packing);

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
		t.Create(shape, depth, packing, allocator);
		if (IsContinue() || steps.empty())
		{
			memcpy(t.data, data, (size_t)shape[0] * steps[0]);
		}
		else
		{
			size_t dims = shape.size();
			size_t rows = shape.vol() / shape.back();
			size_t len = t.steps[dims - 2]; // shape.back() * depth * packing; // t.steps[sz - 1]
			for (size_t r = 0; r < rows; r++)
			{
				size_t offset = 0;
				size_t row = r * len;
				for (int j = 0; j < shape.size() - 1; j++)
				{
					offset += (row / t.steps[j]) * steps[j];
					row %= t.steps[j];
				}
				memcpy((uchar*)t + r * len, (uchar*)data + offset, len);
			}
		}
	}
}