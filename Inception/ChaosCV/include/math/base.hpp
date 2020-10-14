#pragma once

#include "tensor_op.hpp"

namespace chaos
{
    enum DecompTypes
    {
        /** Gaussian elimination with the optimal pivot element chosen. */
        DECOMP_LU = 0,
        /** singular value decomposition (SVD) method; the system can be over-defined and/or the matrix
        src1 can be singular */
        DECOMP_SVD = 1,
        /** eigenvalue decomposition; the matrix src1 must be symmetrical */
        DECOMP_EIG = 2,
        /** Cholesky \f$LL^T\f$ factorization; the matrix src1 must be symmetrical and positively
        defined */
        DECOMP_CHOLESKY = 3,
        /** QR factorization; the system can be over-defined and/or the matrix src1 can be singular */
        DECOMP_QR = 4,
    };

    CHAOS_API bool LU(float* A, size_t astep, int m, float* b, size_t bstep, int n);
    CHAOS_API bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n);

    /** @brief Calculates eigenvalues and eigenvectors of a symmetric matrix.

    The function cv::eigen calculates just eigenvalues, or eigenvalues and eigenvectors of the symmetric
    matrix src:
    @code
        src*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()
    @endcode

    @note Use cv::eigenNonSymmetric for calculation of real eigenvalues and eigenvectors of non-symmetric matrix.

    @param src input matrix that must have CV_32FC1 or CV_64FC1 type, square size and be symmetrical
    (src ^T^ == src).
    @param eigenvalues output vector of eigenvalues of the same type as src; the eigenvalues are stored
    in the descending order.
    @param eigenvectors output matrix of eigenvectors; it has the same size and type as src; the
    eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding
    eigenvalues.
    @sa eigenNonSymmetric, completeSymm , PCA
    */
    CHAOS_API bool Eigen(const InputArray& _src, const OutputArray& _evals, const OutputArray& _evects);

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

        @param A decomposed matrix. The depth has to be Depth::D4.
        @param w calculated singular values
        @param u calculated left singular vectors
        @param vt transposed matrix of right singular vectors
        @param flags operation flags - see SVD::Flags.
          */
        static void Compute(const InputArray& A, const OutputArray& w, const OutputArray& u, const OutputArray& vt, int flags = 0);


        /** @brief performs a singular value back substitution.

        The method calculates a back substitution for the specified right-hand
        side:

        \f[\texttt{x} =  \texttt{vt} ^T  \cdot diag( \texttt{w} )^{-1}  \cdot \texttt{u} ^T  \cdot \texttt{rhs} \sim \texttt{A} ^{-1}  \cdot \texttt{rhs}\f]

        Using this technique you can either get a very accurate solution of the
        convenient linear system, or the best (in the least-squares terms)
        pseudo-solution of an overdetermined linear system.

        @param rhs right-hand side of a linear system (u\*w\*v')\*dst = rhs to
        be solved, where A has been previously decomposed.

        @param dst found solution of the system.

        @note Explicit SVD with the further back substitution only makes sense
        if you need to solve many linear systems with the same left-hand side
        (for example, src ). If all you need is to solve a single system
        (possibly with multiple rhs immediately available), simply call solve
        add pass #DECOMP_SVD there. It does absolutely the same thing.
          */
        static void BackSubst(const InputArray& w, const InputArray& u,
            const InputArray& vt, const InputArray& rhs,
            const OutputArray& dst);

        /** @brief solves an under-determined singular linear system

        The method finds a unit-length solution x of a singular linear system
        A\*x = 0. Depending on the rank of A, there can be no solutions, a
        single solution or an infinite number of solutions. In general, the
        algorithm solves the following problem:
        \f[dst =  \arg \min _{x:  \| x \| =1}  \| src  \cdot x  \|\f]
        @param src left-hand-side matrix.
        @param dst found solution.
          */
        static void SolveZ(const InputArray& src, const OutputArray& dst);
    };

    /** @brief Finds the inverse or pseudo-inverse of a matrix.

    The function cv::invert inverts the matrix src and stores the result in dst
    . When the matrix src is singular or non-square, the function calculates
    the pseudo-inverse matrix (the dst matrix) so that norm(src\*dst - I) is
    minimal, where I is an identity matrix.

    In case of the #DECOMP_LU method, the function returns non-zero value if
    the inverse has been successfully calculated and false if src is singular.

    In case of the #DECOMP_SVD method, the function returns the inverse
    condition number of src (the ratio of the smallest singular value to the
    largest singular value) and false if src is singular. The SVD method
    calculates a pseudo-inverse matrix if src is singular.

    In case of the #DECOMP_EIG method, the function returns the inverse
    condition number of src which is a symmetrical matrix and false if src 
    is singular.

    Similarly to #DECOMP_LU, the method #DECOMP_CHOLESKY works only with
    non-singular square matrices that should also be symmetrical and
    positively defined. In this case, the function stores the inverted
    matrix in dst and returns non-zero. Otherwise, it returns false.

    @param src input floating-point M x N matrix.
    @param dst output matrix of N x M size and the same type as src.
    @param flags inversion method (cv::DecompTypes)
    @sa solve, SVD
    */
    CHAOS_API bool Invert(const InputArray& src, const OutputArray& dst, int method = DECOMP_LU);
}