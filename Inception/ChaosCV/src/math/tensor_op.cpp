#include "math/tensor_op.hpp"


namespace chaos
{
    //////////////////////////////////////// set identity ////////////////////////////////////////////
    void SetIdentity(Tensor& tensor, double val)
    {
        CHECK_EQ(tensor.shape.size(), 2) << "just support 2D tensor";
        int rows = tensor.shape[0], cols = tensor.steps[0] / tensor.depth / tensor.packing;

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
            return;
        }
        if (Depth::D8 == tensor.depth)
        {
            double* data = tensor;
            for (int i = 0; i < rows; i++, data += cols)
            {
                for (int j = 0; j < cols; j++)
                    data[j] = i == j ? val : 0.;
            }
            return;
        }
        LOG(FATAL) << "not supported yet";
        return;
    }

   

    ////////////////////////////////////// permute /////////////////////////////////////////
    // 可能得改一改
    // 否则对于dst不是连续的状态，不太好处理

    template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
    void PermuteImpl(size_t count, const Type* src, const uint*permute_order, const uint* src_shapes, const uint* src_steps, 
        const uint* dst_shapes, const uint* dst_steps, size_t num_axes, Type* dst)
    {
        for (int i = 0; i < count; i++)
        {
            int src_idx = 0;
            int dst_idx = 0;
            int idx = i;
            for (int j = num_axes - 1; j >= 0; j--)
            {
                int order = permute_order[j];
                int k = idx % dst_shapes[j];
                dst_idx += k * (size_t)dst_steps[j] / sizeof(Type);
                src_idx += k * (size_t)src_steps[order] / sizeof(Type);
                idx /= dst_shapes[j];
            }
            dst[dst_idx] = src[src_idx];
        }
    }

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
            return PermuteImpl<float>(shape.vol(), src, orders.data(), src.shape.data(), src.steps.data(), shape.data(), dst.steps.data(), num_axes, dst);
        if (Depth::D8 == src.depth)
            return PermuteImpl<double>(shape.vol(), src, orders.data(), src.shape.data(), src.steps.data(), shape.data(), dst.steps.data(), num_axes, dst);
        LOG(FATAL) << "not supported yet";
        return;
    }

    ////////////////////////////////////// transpose /////////////////////////////////////////

    template<typename Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
    static void TransposeImpl(const uchar* src, size_t sstep, uchar* dst, size_t dstep, int width, int height)
    {
        int i = 0, j, m = width, n = height;

        //UNROLLED
        for (; i <= m - 4; i += 4)
        {
            Type* d0 = (Type*)(dst + dstep * i);
            Type* d1 = (Type*)(dst + dstep * (i + 1LL));
            Type* d2 = (Type*)(dst + dstep * (i + 2LL));
            Type* d3 = (Type*)(dst + dstep * (i + 3LL));

            for (j = 0; j <= n - 4; j += 4)
            {
                const Type* s0 = (const Type*)(src + i * sizeof(Type) + sstep * j);
                const Type* s1 = (const Type*)(src + i * sizeof(Type) + sstep * (j + 1LL));
                const Type* s2 = (const Type*)(src + i * sizeof(Type) + sstep * (j + 2LL));
                const Type* s3 = (const Type*)(src + i * sizeof(Type) + sstep * (j + 3LL));

                d0[j] = s0[0]; d0[j + 1] = s1[0]; d0[j + 2] = s2[0]; d0[j + 3] = s3[0];
                d1[j] = s0[1]; d1[j + 1] = s1[1]; d1[j + 2] = s2[1]; d1[j + 3] = s3[1];
                d2[j] = s0[2]; d2[j + 1] = s1[2]; d2[j + 2] = s2[2]; d2[j + 3] = s3[2];
                d3[j] = s0[3]; d3[j + 1] = s1[3]; d3[j + 2] = s2[3]; d3[j + 3] = s3[3];
            }

            for (; j < n; j++)
            {
                const Type* s0 = (const Type*)(src + i * sizeof(Type) + j * sstep);
                d0[j] = s0[0]; d1[j] = s0[1]; d2[j] = s0[2]; d3[j] = s0[3];
            }
        }
        for (; i < m; i++)
        {
            Type* d0 = (Type*)(dst + dstep * i);
            j = 0;
            //UNROLLED
            for (; j <= n - 4; j += 4)
            {
                const Type* s0 = (const Type*)(src + i * sizeof(Type) + sstep * j);
                const Type* s1 = (const Type*)(src + i * sizeof(Type) + sstep * (j + 1LL));
                const Type* s2 = (const Type*)(src + i * sizeof(Type) + sstep * (j + 2LL));
                const Type* s3 = (const Type*)(src + i * sizeof(Type) + sstep * (j + 3LL));

                d0[j] = s0[0]; d0[j + 1] = s1[0]; d0[j + 2] = s2[0]; d0[j + 3] = s3[0];
            }
            for (; j < n; j++)
            {
                const Type* s0 = (const Type*)(src + i * sizeof(Type) + j * sstep);
                d0[j] = s0[0];
            }
        }
    }

    template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
    static void TransposeInplaceImpl(uchar* data, size_t step, int n)
    {
        for (int i = 0; i < n; i++)
        {
            Type* row = (Type*)(data + step * i);
            uchar* data1 = data + i * sizeof(Type);
            for (int j = i + 1; j < n; j++)
                std::swap(row[j], *(Type*)(data1 + step * j));
        }
    }

    void Transpose(const Tensor& src, Tensor& dst)
    {
        CHECK_EQ(src.shape.size(), 2);
        CHECK_EQ(src.packing, Packing::CHW);

        if (src.data == dst.data) // inplace
        {
            CHECK_EQ(dst.shape[0], dst.shape[1]);
            TransposeInplaceImpl<float>(dst, dst.steps[0], dst.shape[0]);
        }
        else
        {
            dst.Create({ src.shape[1], src.shape[0] }, src.depth, src.packing, src.allocator);
            TransposeImpl<float>(src, src.steps[0], dst, dst.steps[0], src.shape[1], src.shape[0]);
        }
    }

}