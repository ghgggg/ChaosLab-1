#pragma once

#include "tensor_op.hpp"

namespace chaos
{
    CHAOS_API bool LU(float* A, size_t astep, int m, float* b, size_t bstep, int n);
    CHAOS_API bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n);

    CHAOS_API void JacobiSVD(float* At, size_t astep, float* W, float* Vt, size_t vstep, int m, int n, int n1 = -1);


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

        auto ret = LU(src1, src1.shape[1], n, dst, dst.shape[1], n);
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


    CHAOS_API void SVDcompute(const Tensor& _aarr, Tensor& _w,
        Tensor& _u, Tensor& _vt, int flags);

    class CHAOS_API SVD
    {
    public:
        enum Flags 
        {
            /** allow the algorithm to modify the decomposed matrix; it can save space and speed up
            processing. currently ignored. */
            MODIFY_A = 1,
            /** indicates that only a vector of singular values `w` is to be processed, while u and vt
            will be set to empty matrices */
            NO_UV = 2,
            /** when the matrix is not square, by default the algorithm produces u and vt matrices of
            sufficiently large size for the further A reconstruction; if, however, FULL_UV flag is
            specified, u and vt will be full-size square orthogonal matrices.*/
            FULL_UV = 4
        };

        /** @brief decomposes matrix and stores the results to user-provided matrices

        The methods/functions perform SVD of matrix. Unlike SVD::SVD constructor
        and SVD::operator(), they store the results to the user-provided
        matrices:

        @code{.cpp}
        Mat A, w, u, vt;
        SVD::compute(A, w, u, vt);
        @endcode

        @param src decomposed matrix. The depth has to be Depth::D4.
        @param w calculated singular values
        @param u calculated left singular vectors
        @param vt transposed matrix of right singular vectors
        @param flags operation flags - see SVD::Flags.
          */
        static void Compute(const Tensor& src, Tensor& w, Tensor& u, Tensor& vt, int flags = 0);


        /** @brief solves an under-determined singular linear system

        The method finds a unit-length solution x of a singular linear system
        A\*x = 0. Depending on the rank of A, there can be no solutions, a
        single solution or an infinite number of solutions. In general, the
        algorithm solves the following problem:
        \f[dst =  \arg \min _{x:  \| x \| =1}  \| src  \cdot x  \|\f]
        @param src left-hand-side matrix.
        @param dst found solution.
          */
        static void SolveZ(const Tensor& src, Tensor dst);
    };
}