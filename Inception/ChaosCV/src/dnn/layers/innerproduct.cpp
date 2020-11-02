#include "dnn/layers/innerproduct.hpp"

namespace chaos
{
	namespace dnn
	{
		inline void CalcRowOffsets(const Shape& shape, const Steps& steps, uint* offsets)
		{
			size_t dims = shape.size();
			uint cols = shape.back();
			uint rows = (uint)shape.vol() / cols;

			for (size_t r = 1; r < rows; r++)
			{
				size_t idx = r * cols;
				size_t offset = 0;
				for (int64 j = dims - 1; j >= 0; j--)
				{
					size_t k = idx % shape[j];
					offset += k * steps[j];
					idx /= shape[j];
				}
				offsets[r] = (uint)offset;
			}
		}

		InnerProduct::InnerProduct() : Layer("InnerProduct")
		{
			one_blob_only = true;
		}

		void InnerProduct::Set(const std::string& key, const ParamValue& value)
		{
			if (key == "weight")
			{
				weight = value;
				CHECK_EQ(2, weight.shape.size());
			}
			if (key == "bias")
			{
				bias = value;
			}
		}

		void InnerProduct::Forward(const Tensor& bottom, Tensor& top, const Option& opt) const
		{
			bool use_bias = not bias.empty();
			size_t in_dims = bottom.shape.size();
			uint inw = bottom.shape.back();
			uint inh = (uint)bottom.shape.vol() / inw;
			CHECK_EQ(inw, weight.shape[1]) << Format("expect %d, but got %d)", weight.shape.back(), inw);
			
			AutoBuffer<uint, 8> in_offsets(inh);
			CalcRowOffsets(bottom.shape, bottom.steps, in_offsets.data());

			Shape out_shape = bottom.shape;
			uint outw = out_shape.back() = weight.shape[0];
			top.Create(out_shape, out_shape.steps(), Depth::D4, Packing::CHW, opt.blob_allocator);
			if (use_bias)
			{
				CHECK_EQ(top.shape, bias.shape);
				CHECK_EQ(top.steps, bias.steps);
			}

			AutoBuffer<uint, 8> out_offsets(inh);
			CalcRowOffsets(top.shape, top.steps, out_offsets.data());

			for (size_t r = 0; r < inh; r++)
			{
				const float* x = (const float*)bottom + in_offsets[r];
				for (size_t c = 0; c < outw; c++)
				{
					const float* w = (const float*)weight + c * weight.steps[0];

					size_t offset = out_offsets[r] + c; //r * outw + c;
					float b = use_bias ? bias[offset] : 0.f;
					float y = std::inner_product(x, x + inw, w, b);

					if (activation_type == 1)
					{
						y = std::max(0.f, y);
					}
					if (activation_type == 2)
					{
						float slope = activation_params[0];
						y = y > 0.f ? y : y * slope;
					}
					if (activation_type == 3)
					{
						float min = activation_params[0];
						float max = activation_params[1];
						if (y < min)
							y = min;
						if (y > max)
							y = max;
					}
					if (activation_type == 4)
					{
						y = 1.f / (1.f + std::exp(-y));
					}
					if (activation_type == 5)
					{
						y = y * std::tanh(std::log(std::exp(y) + 1.f));
					}
					
					top[offset] = y;
				}
			}
		}
	}
}