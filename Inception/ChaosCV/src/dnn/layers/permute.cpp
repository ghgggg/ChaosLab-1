#include "dnn/layers/permute.hpp"

namespace chaos
{
	namespace dnn
	{
        template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
        void PermuteImpl(size_t count, const Type* src, const uint* permute_order, const uint* src_shapes, const uint* src_steps,
            const uint* dst_shapes, const uint* dst_steps, size_t num_axes, Type* dst)
        {
            for (size_t i = 0; i < count; i++)
            {
                size_t src_idx = 0;
                size_t dst_idx = 0;
                size_t idx = i;
                for (int64 j = num_axes - 1; j >= 0; j--)
                {
                    size_t order = permute_order[j];
                    size_t k = idx % dst_shapes[j];
                    dst_idx += k * dst_steps[j];
                    src_idx += k * src_steps[order];
                    idx /= dst_shapes[j];
                }
                dst[dst_idx] = src[src_idx];
            }
        }

		Permute::Permute() : Layer("Permute")
		{
			one_blob_only = true;
		}

        void Permute::Set(const std::string& key, const ParamValue& value)
        {
            if (key == "orders")
            {
                const Tensor& v = value;
                orders = Vec<uint>(v.shape.vol(), (const float*)v);
            }
        }

		void Permute::Forward(const Tensor& bottom, Tensor& top, const Option& opt) const
		{
            size_t num_axes = bottom.shape.size();
            CHECK_EQ(orders.size(), num_axes) << Format("expect %d, got %d", orders.size(), num_axes);

            bool need_permute = false;
            for (size_t i = 0; i < num_axes; i++)
            {
                if (i != orders[i])
                {
                    need_permute = true;
                    break;
                }
            }
            if (not need_permute)
            {
                bottom.CopyTo(top);
                return;
            }

            Shape shape = bottom.shape;
            for (size_t i = 0; i < num_axes; i++) shape[i] = bottom.shape[orders[i]];
            top.Create(shape, Depth::D4, Packing::CHW, opt.blob_allocator);

            PermuteImpl<float>(shape.vol(), bottom, orders.data(), bottom.shape.data(), bottom.steps.data(), 
                top.shape.data(), top.steps.data(), num_axes, top);
		}
	}
}