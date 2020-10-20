#include "dnn/layers/innerproduct.hpp"

namespace chaos
{
	namespace dnn
	{
		InnerProduct::InnerProduct() : Layer("InnerProduct")
		{
			one_blob_only = true;
		}

		void InnerProduct::Set(const std::string& key, const ParamValue& value)
		{
			if (key == "weight")
			{
				weight = value;
			}
			if (key == "bias")
			{
				bias = value;
			}
		}

		void InnerProduct::Forward(const Tensor& bottom, Tensor& top, const Option& opt) const
		{
			size_t in_dims = bottom.shape.size();
			uint inw = bottom.shape.back();
			uint inh = (uint)bottom.shape.vol() / inw;
			CHECK_EQ(inw, weight.shape[1]) << Format("expect %d, but got %d)", weight.shape.back(), inw);
			
			AutoBuffer<uint, 8> rsteps(inh); // 8 is enough for dims
			for (size_t r = 1; r < inh; r++)
			{
				size_t idx = r * inw;
				size_t step = 0;
				for (int64 j = in_dims - 1; j >= 0; j--)
				{
					size_t k = idx % bottom.shape[j];
					step += k * bottom.steps[j];
					idx /= bottom.shape[j];
				}
				rsteps[r] = (uint)step;
			}

			Shape out_shape = bottom.shape;
			uint outw = out_shape.back() = weight.shape[0];
			// 这个地方对形状的推理不对
			top.Create(out_shape, Depth::D4, Packing::CHW, opt.blob_allocator);
			if (not bias.empty()) CHECK_EQ(top.shape, bias.shape);

			for (size_t r = 0; r < inh; r++)
			{
				const float* x = (const float*)bottom + rsteps[r];
				for (size_t c = 0; c < outw; c++)
				{
					const float* w = (const float*)weight + c * weight.steps[0];

					size_t offset = r * outw + c;
					float b = bias.empty() ? 0.f : bias[offset];
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