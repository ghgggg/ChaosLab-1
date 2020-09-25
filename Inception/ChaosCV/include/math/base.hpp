#pragma once

#include "lapack.hpp"
#include "tensor_op.hpp"

namespace chaos
{
    static void SVDcompute(const Tensor& _aarr, Tensor& _w,
        Tensor& _u, Tensor& _vt, int flags)
    {

    }



    bool Invert(const Tensor& src, Tensor& dst, int method)
    {
        int m = src.shape[0], n = src.shape[1];

        CHECK_EQ(m, n);

        dst.Create(Shape(n,n), src.depth, src.packing, src.allocator);

        int elem_size = static_cast<int>(src.depth);
        AutoBuffer<uchar> buf(n * n * elem_size);
        Tensor src1(Shape(n,n), src.depth, Packing::CHW, buf.data());
        src.CopyTo(src1);

        SetIdentity(dst);

        auto ret = LUImpl<float>(src1, src1.shape[1], n, dst, dst.shape[1], n, FLT_EPSILON * 10);
        //if (method == DECOMP_LU && type == CV_32F)
        //    result = hal::LU32f(src1.ptr<float>(), src1.step, n, dst.ptr<float>(), dst.step, n) != 0;
        //else if (method == DECOMP_LU && type == CV_64F)
        //    result = hal::LU64f(src1.ptr<double>(), src1.step, n, dst.ptr<double>(), dst.step, n) != 0;
        //else if (method == DECOMP_CHOLESKY && type == CV_32F)
        //    result = hal::Cholesky32f(src1.ptr<float>(), src1.step, n, dst.ptr<float>(), dst.step, n);
        //else
        //    result = hal::Cholesky64f(src1.ptr<double>(), src1.step, n, dst.ptr<double>(), dst.step, n);
        return true;
    }
}