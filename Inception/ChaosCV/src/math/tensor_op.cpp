#include "math/tensor_op.hpp"


namespace chaos
{
    //////////////////////////////////////// set identity ////////////////////////////////////////////
    void SetIdentity(Tensor& tensor, double val)
    {
        CHECK_EQ(tensor.shape.size(), 2) << "just support 2D tensor";
        int rows = tensor.shape[0], cols = tensor.shape[1];

        if (Depth::D4 == tensor.depth)
        {
            float* data = tensor;
            for (int i = 0; i < rows; i++, data += cols)
            {
                for (int j = 0; j < cols; j++)
                    data[j] = 0.f;
                if (i < cols)
                    data[i] = (float)val;
            }
        }
        if (Depth::D8 == tensor.depth)
        {
            double* data = tensor;
            for (int i = 0; i < rows; i++, data += cols)
            {
                for (int j = 0; j < cols; j++)
                    data[j] = i == j ? val : 0.;
            }
        }
        LOG(FATAL) << "not supported yet";
    }

    ////////////////////////////////////// transpose /////////////////////////////////////////
    void Transpose(const Tensor& src, Tensor& dst)
    {
        for (size_t i = 0; i < src.shape[0]; i++)
        {
            for (size_t j = 0; j < src.shape[1]; j++)
            {
                dst[j * dst.shape[1] + i] = src[i * src.shape[1] + j];
            }
        }
    }

    ////////////////////////////////////// permute /////////////////////////////////////////
    template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
    void PermuteImpl(size_t count, const Type* src, const uint* permute_order, const uint* old_steps, const uint* new_steps, size_t num_axes, Type* dst)
    {
        for (int i = 0; i < count; i++)
        {
            int old_idx = 0;
            int idx = i;
            for (int j = 0; j < num_axes; j++)
            {
                int order = permute_order[j];
                old_idx += (idx * sizeof(Type) / new_steps[j]) * old_steps[order] / sizeof(Type);
                idx %= (new_steps[j] / sizeof(Type));
            }
            dst[i] = src[old_idx];
        }
    }

    void Permute(const Tensor& src, Tensor& dst, const Vec<uint>& orders)
    {
        CHECK_EQ(src.packing, Packing::CHW);
        CHECK_EQ(src.shape.size(), orders.size());

        size_t num_axes = src.shape.size();
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
            dst = src;
            return;
        }

        Shape shape = src.shape;
        for (size_t i = 0; i < num_axes; i++)
        {
            shape[i] = src.shape[orders[i]];
        }

        CHECK_EQ(src.shape.vol(), shape.vol());

        dst.Create(shape, src.depth, src.packing, src.allocator);

        if (Depth::D4 == src.depth)
            return PermuteImpl<float>(shape.vol(), src, orders.data(), src.steps.data(), dst.steps.data(), num_axes, dst);
        if (Depth::D8 == src.depth)
            return PermuteImpl<double>(shape.vol(), src, orders.data(), src.steps.data(), dst.steps.data(), num_axes, dst);
        LOG(FATAL) << "not supported yet";
        return;
    }

}