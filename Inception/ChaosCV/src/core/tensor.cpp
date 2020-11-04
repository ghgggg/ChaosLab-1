#include "core/tensor.hpp"

#include "core/vulkan/vk_tensor.hpp"

namespace chaos
{
	

	Tensor::Tensor(const Shape& shape, const Depth& depth, const Packing& packing, Allocator* allocator)
	{
		Create(shape, shape.steps(), depth, packing, allocator);
	}
	Tensor::Tensor(const Shape& _shape, const Depth& _depth, const Packing& _packing, void* _data, const Steps& _steps)
		: data(_data), shape(_shape), depth(_depth), packing(_packing) 
	{
		if (_steps.empty())
		{
			steps = shape.steps();
		}
		else
		{
			steps = _steps;
			CHECK_EQ(steps.size(), shape.size());
		}
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

	void Tensor::Create(const Shape& _shape, const Steps& _steps, const Depth& _depth, const Packing& _packing, Allocator* _allocator)
	{
		if (_shape == shape && _steps == steps && _depth == depth && _packing == packing  && _allocator == allocator) return;

		Release();

		shape = _shape;
		depth = _depth;
		packing = _packing;
		allocator = _allocator;
		steps = _steps; //Steps(shape, 1 * depth * packing);

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

	void Tensor::CreateLike(const VkTensor& t, Allocator* allocator)
	{
		Create(t.shape, t.steps, t.depth, t.packing, allocator);
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

	void Tensor::CopyTo(const OutputArray& arr, Allocator* allocator) const
	{
		if (arr.empty()) arr.Create(shape, steps, depth, packing, allocator);
		Tensor& t = arr.GetTensorRef();
		CHECK_EQ(t.shape, shape);

		size_t elem_size = 1 * depth * packing;
		if (continua() && t.continua())
		{
			memcpy(t.data, data, (size_t)shape[0] * steps[0] * elem_size);
		}
		else
		{
			size_t dims = shape.size();
			size_t rows = shape.vol() / shape.back();
			size_t len = shape[dims - 1] * elem_size; // row len
			for (size_t r = 0; r < rows; r++)
			{
				size_t dst_offset = 0;
				size_t src_offset = 0;
				size_t idx = r * shape[dims - 1];
				for (int64 j = dims - 1; j >= 0; j--)
				{
					size_t k = idx % shape[j];
					dst_offset += k * t.steps[j];
					src_offset += k * steps[j];
					idx /= shape[j];
				}
				memcpy((uchar*)t + dst_offset * elem_size, (uchar*)data + src_offset * elem_size, len);
			}
		}
	}
	Tensor Tensor::Clone(Allocator* allocator) const
	{
		Tensor t;
		CopyTo(t, allocator);
		return t;
	}

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
	//void InputArray::GetVectorTensor(std::vector<Tensor>& mv) const
	//{
	//	if (flag == TENSOR)
	//	{
	//		const Tensor& t = *(Tensor*)obj;
	//		mv.emplace_back(t);
	//		//mv.resize(1);
	//		//mv[0] = t;
	//		return;
	//	}
	//	if (flag == VECTOR_TENSOR)
	//	{
	//		const std::vector<Tensor>& data = *(std::vector<Tensor>*)obj;
	//		size_t n = data.size();
	//		mv.resize(n);
	//		for (size_t i = 0; i < n; i++)
	//			mv[i] = data[i];
	//		return;
	//	}
	//	LOG(FATAL) << "unknown/unsupported array type";
	//}
	bool InputArray::IsTensor() const { return flag == TENSOR; }
	bool InputArray::empty() const
	{
		if (flag == TENSOR) return ((const Tensor*)obj)->empty();
	}
	void InputArray::Init(int _flag, const void* _obj)
	{
		flag = _flag; obj = (void*)_obj;
	}


	OutputArray::OutputArray() { Init(NONE, nullptr); }
	OutputArray::OutputArray(Tensor& data) { Init(TENSOR, &data); }
	void OutputArray::Create(const Shape& shape, const Steps& steps, const Depth& depth, const Packing& packing, Allocator* allocator) const
	{
		if (flag == TENSOR)
		{
			return ((Tensor*)obj)->Create(shape, steps.empty() ? shape.steps() : steps, depth, packing, allocator);
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
	Tensor& OutputArray::GetTensorRef() const
	{
		if (flag == TENSOR)
		{
			return *(Tensor*)obj;
		}
		LOG(FATAL) << "unknown/unsupported array type";
		return *(Tensor*)obj; // warning C4715
	}
	

	InputOutputArray::InputOutputArray() { Init(NONE, nullptr); }
	InputOutputArray::InputOutputArray(Tensor& data) { Init(TENSOR, &data); }

	static InputOutputArray none;
	const InputOutputArray& noArray() { return none; }
}