#include "math/base.hpp"
#include "math/tensor_op.hpp"

#include "dnn/layer.hpp"
#include "dnn/layer_factory.hpp"

namespace chaos
{
    static dnn::Option opt; // default option
    void Add(const InputArray& _a, const InputArray& _b, const OutputArray& _c)
    {
        auto layer = dnn::LayerRegistry::CreateLayer("BinaryOp");

        layer->Set("op", dnn::BinOpType::ADD);

        Tensor a = _a.GetTensor();
        Tensor b = _b.GetTensor();
        _c.Create(a.shape.vol() > b.shape.vol() ? a.shape : b.shape, Steps(), Depth::D4, Packing::CHW, opt.blob_allocator);
        Tensor c = _c.GetTensor();

        std::vector<Tensor> tops = { c };
        layer->Forward({ a, b }, tops, opt);
    }
    void Sub(const InputArray& _a, const InputArray& _b, const OutputArray& _c)
    {
        auto layer = dnn::LayerRegistry::CreateLayer("BinaryOp");

        layer->Set("op", dnn::BinOpType::SUB);

        Tensor a = _a.GetTensor();
        Tensor b = _b.GetTensor();
        _c.Create(a.shape.vol() > b.shape.vol() ? a.shape : b.shape, Steps(), Depth::D4, Packing::CHW, opt.blob_allocator);
        Tensor c = _c.GetTensor();

        std::vector<Tensor> tops = { c };
        layer->Forward({ a, b }, tops, opt);
    }
    void Mul(const InputArray& _a, const InputArray& _b, const OutputArray& _c)
    {
        auto layer = dnn::LayerRegistry::CreateLayer("BinaryOp");

        layer->Set("op", dnn::BinOpType::MUL);

        Tensor a = _a.GetTensor();
        Tensor b = _b.GetTensor();
        _c.Create(a.shape.vol() > b.shape.vol() ? a.shape : b.shape, Steps(), Depth::D4, Packing::CHW, opt.blob_allocator);
        Tensor c = _c.GetTensor();

        std::vector<Tensor> tops = { c };
        layer->Forward({ a, b }, tops, opt);
    }
    void Div(const InputArray& _a, const InputArray& _b, const OutputArray& _c)
    {
        auto layer = dnn::LayerRegistry::CreateLayer("BinaryOp");

        layer->Set("op", dnn::BinOpType::DIV);

        Tensor a = _a.GetTensor();
        Tensor b = _b.GetTensor();
        _c.Create(a.shape.vol() > b.shape.vol() ? a.shape : b.shape, Steps(), Depth::D4, Packing::CHW, opt.blob_allocator);
        Tensor c = _c.GetTensor();

        std::vector<Tensor> tops = {c};
        layer->Forward({ a, b }, tops, opt);
    }


    void Dot(const InputArray& _a, const InputArray& _b, const OutputArray& _c)
    {
        Tensor a = _a.GetTensor(), b;
        Transpose(_b.GetTensor(), b);
        auto layer = dnn::LayerRegistry::CreateLayer("InnerProduct");
        layer->Set("weight", b);

        Tensor& c = _c.GetTensorRef();
        layer->Forward(a, c, opt);
    }

    //////////////////////////////////////// set identity ////////////////////////////////////////////
    void SetIdentity(const InputOutputArray& _src, double val)
    {
        Tensor src = _src.GetTensor();
        //CHECK_EQ(tensor.shape.size(), 2) << "just support 2D tensor";
        int rows = src.shape[0], cols = src.steps[0];

        if (Depth::D4 == src.depth)
        {
            float* data = src;
            for (int i = 0; i < rows; i++, data += cols)
            {
                for (int j = 0; j < cols; j++)
                    data[j] = 0.f;
                if (i < cols)
                    data[i] = (float)val;
            }
            return;
        }
        if (Depth::D8 == src.depth)
        {
            double* data = src;
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
    template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
    void PermuteImpl(size_t count, const Type* src, const uint*permute_order, const uint* src_shapes, const uint* src_steps, 
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

    //template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
    //void PermuteImpl(size_t count, const Type* src, const uint* permute_order, const uint* old_steps, const uint* new_steps, size_t num_axes, Type* dst)
    //{
    //    for (int i = 0; i < count; i++)
    //    {
    //        int old_idx = 0;
    //        int idx = i;
    //        for (int j = 0; j < num_axes; j++)
    //        {
    //            int order = permute_order[j];
    //            old_idx += (idx * sizeof(Type) / new_steps[j]) * old_steps[order] / sizeof(Type);
    //            idx %= (new_steps[j] / sizeof(Type));
    //        }
    //        dst[i] = src[old_idx];
    //    }
    //}

    void Permute(const InputArray& _src, const OutputArray& _dst, const Vec<uint>& orders)
    {

#if 0
        Tensor src = _src.GetTensor();
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
            //dst = src;
            src.CopyTo(_dst);
            return;
        }

        Shape shape = src.shape;
        for (size_t i = 0; i < num_axes; i++)
        {
            shape[i] = src.shape[orders[i]];
        }
        CHECK_EQ(src.shape.vol(), shape.vol());

        _dst.Create(shape, src.depth, src.packing, src.allocator);
        Tensor dst = _dst.GetTensor();

        if (Depth::D4 == src.depth)
            return PermuteImpl<float>(shape.vol(), src, orders.data(), src.shape.data(), src.steps.data(), shape.data(), dst.steps.data(), num_axes, dst);
        LOG(FATAL) << "not supported yet";
        return;
#endif
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

    void Transpose(const InputArray& _src, const OutputArray& _dst)
    {
        Tensor src = _src.GetTensor();

        if (_dst.empty()) _dst.Create({ src.shape[1], src.shape[0] }, Steps(), src.depth, src.packing, src.allocator);
        Tensor& dst = _dst.GetTensorRef();
        CHECK(src.shape[0], dst.shape[1]);
        CHECK(src.shape[1], dst.shape[0]);

        size_t esz = 1 * src.depth * src.packing;
        if (dst.data == src.data)
        {
            CHECK_EQ(dst.shape[0], dst.shape[1]);
            TransposeInplaceImpl<float>(dst, dst.steps[0] * esz, dst.shape[0]);
        }
        else
        {
            TransposeImpl<float>(src, src.steps[0] * esz, dst, dst.steps[0] * esz, src.shape[1], src.shape[0]);
        }
    }

}