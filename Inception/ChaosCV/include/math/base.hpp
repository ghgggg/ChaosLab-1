#pragma once

#include "lapack.hpp"
#include "tensor_op.hpp"

namespace chaos
{
    CHAOS_API bool LU(float* A, size_t astep, int m, float* b, size_t bstep, int n);
    CHAOS_API bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n);

    CHAOS_API void JacobiSVD(float* At, size_t astep, float* W, float* Vt, size_t vstep, int m, int n, int n1 = -1);

    static void SVDcompute(const Tensor& src, Tensor& _w,
        Tensor& _u, Tensor& _vt, int flags)
    {

#if 0
        int m = src.shape[0], n = src.shape[1]; // .cols;
        //int type = src.type();
        bool compute_uv = not _u.empty() || not _vt.empty(); // _u.needed() || _vt.needed();
        //bool full_uv = (flags & SVD::FULL_UV) != 0;

        //CV_Assert(type == CV_32F || type == CV_64F);

        //if (flags & SVD::NO_UV)
        //{
        //    _u.release();
        //    _vt.release();
        //    compute_uv = full_uv = false;
        //}

        bool at = false;
        if (m < n)
        {
            std::swap(m, n);
            at = true;
        }

        int urows = m; //full_uv ? m : n;
        size_t esz = static_cast<size_t>(src.depth), astep = AlignSize(m * esz, 16) / esz, vstep = AlignSize(n * esz, 16) / esz;
        AutoBuffer<uchar> _buf(urows * astep + n * vstep + n * esz + 32);
        uchar* buf = AlignPtr(_buf.data(), 16);
        Tensor temp_a(Shape(n, m), Depth::D4, Packing::CHW, buf, astep);
        Tensor temp_w(Shape(n, 1), Depth::D4, Packing::CHW, buf + urows * astep);
        Tensor temp_u(Shape(urows, m), Depth::D4, Packing::CHW, buf, astep), temp_v;

        if (compute_uv)
            temp_v = Tensor(Shape(n, n), Depth::D4, Packing::CHW, AlignPtr(buf + urows * astep + n * esz, 16), vstep);

        if (urows > n)
            memset(temp_u, 0, temp_u.shape.vol() * esz);
            //temp_u = Scalar::all(0);

        if (!at)
            Transpose(src, temp_a);
        else
            src.CopyTo(temp_a);

        //if (type == CV_32F)
        {
            JacobiSVDImpl<float>(temp_a, temp_u.step, temp_w,
                temp_v, temp_v.  , m, n, compute_uv ? urows : 0);
        }
        //else
        //{
        //    JacobiSVD(temp_a.ptr<double>(), temp_u.step, temp_w.ptr<double>(),
        //        temp_v.ptr<double>(), temp_v.step, m, n, compute_uv ? urows : 0);
        //}
        temp_w.CopyTo(_w);
        if (compute_uv)
        {
            if (!at)
            {
                if (not _u.empty())
                    Transpose(temp_u, _u);
                if (not _vt.empty())
                    temp_v.CopyTo(_vt);
            }
            else
            {
                if (_u.needed())
                    transpose(temp_v, _u);
                if (_vt.needed())
                    temp_u.CopyTo(_vt);
            }
        }
#endif
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