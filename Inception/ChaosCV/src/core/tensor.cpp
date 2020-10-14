#include "core/tensor.hpp"

namespace chaos
{
	

	Tensor::Tensor(const Shape& shape, const Depth& depth, const Packing& packing, Allocator* allocator)
	{
		Create(shape, depth, packing, allocator);
	}
	Tensor::Tensor(const Shape& _shape, const Depth& _depth, const Packing& _packing, void* _data, const Steps& _steps)
		: data(_data), shape(_shape), depth(_depth), packing(_packing) 
	{
		steps = _steps.empty() ? Steps(shape, 1 * depth * packing) : _steps;
		CHECK_EQ(steps.size(), shape.size());
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
		steps = t.steps;

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

		steps = Steps(shape, 1 * depth * packing);

		size_t total = (size_t)steps[0] * shape[0];
		if (total > 0)
		{
			size_t size = AlignSize(total * depth * packing, 4);

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

	void Tensor::CopyTo(const OutputArray& arr) const
	{
		arr.Create(shape, depth, packing, allocator);
		Tensor t = arr.GetTensor();

		//if (t.empty()) t.Create(shape, depth, packing, allocator);
		if (IsContinue() && t.IsContinue())
		{
			memcpy(t.data, data, (size_t)shape[0] * steps[0]);
		}
		else
		{
			size_t dims = shape.size();
			size_t rows = shape.vol() / shape.back();
			size_t rstep = t.steps[dims - 2];
			size_t len = shape.back() * depth * packing; // row lenth
			for (size_t r = 0; r < rows; r++)
			{
				size_t offset = 0;
				size_t row = r * rstep;
				for (int j = 0; j < shape.size() - 1; j++)
				{
					offset += (row / t.steps[j]) * steps[j];
					row %= t.steps[j];
				}
				memcpy((uchar*)t + r * rstep, (uchar*)data + offset, len);
			}
		}
	}
	//void Tensor::CopyTo(Tensor& t) const
	//{
	//	if (t.empty()) t.Create(shape, depth, packing, allocator);
	//	if (IsContinue() && t.IsContinue())
	//	{
	//		memcpy(t.data, data, (size_t)shape[0] * steps[0]);
	//	}
	//	else
	//	{
	//		size_t dims = shape.size();
	//		size_t rows = shape.vol() / shape.back();
	//		size_t rstep = t.steps[dims - 2];
	//		size_t len = shape.back() * depth * packing; // row lenth
	//		for (size_t r = 0; r < rows; r++)
	//		{
	//			size_t offset = 0;
	//			size_t row = r * rstep;
	//			for (int j = 0; j < shape.size() - 1; j++)
	//			{
	//				offset += (row / t.steps[j]) * steps[j];
	//				row %= t.steps[j];
	//			}
	//			memcpy((uchar*)t + r * rstep, (uchar*)data + offset, len);
	//		}
	//	}
	//}


	InputArray::InputArray() { Init(NONE, nullptr); }
	InputArray::InputArray(const Tensor& data) { Init(TENSOR, &data); }
	Tensor InputArray::GetTensor() const
	{
		if (flag == TENSOR)
		{
			return *(Tensor*)obj;
		}
		LOG(FATAL) << "unknown/unsupported array type";
		return Tensor();
	}
	bool InputArray::IsTensor() const { return flag == TENSOR; }
	void InputArray::Init(int _flag, const void* _obj)
	{
		flag = _flag; obj = (void*)_obj;
	}


	OutputArray::OutputArray() { Init(NONE, nullptr); }
	OutputArray::OutputArray(Tensor& data) { Init(TENSOR, &data); }
	void OutputArray::Create(const Shape& shape, const Depth& depth, const Packing& packing, Allocator* allocator) const
	{
		if (flag == TENSOR)
		{
			return ((Tensor*)obj)->Create(shape, depth, packing, allocator);
		}
		if (flag == NONE)
		{
			LOG(FATAL) << "Create() called for the missing output array";
		}
		LOG(FATAL) << "unknown/unsupported array type";
	}
	void OutputArray::Release() const
	{
		if (flag == TENSOR)
		{
			return ((Tensor*)obj)->Release();
		}
		if (flag == NONE)
		{
			return;
		}
		LOG(FATAL) << "unknown/unsupported array type";
	}
	bool OutputArray::Needed() const
	{
		return flag != NONE;
	}
	

	InputOutputArray::InputOutputArray() { Init(NONE, nullptr); }
	InputOutputArray::InputOutputArray(Tensor& data) { Init(TENSOR, &data); }

	static InputOutputArray none;
	const InputOutputArray& noArray() { return none; }
}