#include "math/base.hpp"

namespace chaos
{
    template<typename Type>
    static inline bool LUImpl(Type* A, size_t astep, int m, Type* b, size_t bstep, int n, Type eps)
    {
        int i, j, k;
        bool p = true;
        astep /= sizeof(A[0]);
        bstep /= sizeof(b[0]);

        for (i = 0; i < m; i++)
        {
            k = i;

            for (j = i + 1; j < m; j++)
                if (std::abs(A[j * astep + i]) > std::abs(A[k * astep + i]))
                    k = j;

            if (std::abs(A[k * astep + i]) < eps)
                return 0;

            if (k != i)
            {
                for (j = i; j < m; j++)
                    std::swap(A[i * astep + j], A[k * astep + j]);
                if (b)
                    for (j = 0; j < n; j++)
                        std::swap(b[i * bstep + j], b[k * bstep + j]);
                p = !p;
            }

            Type d = -1 / A[i * astep + i];

            for (j = i + 1; j < m; j++)
            {
                Type alpha = A[j * astep + i] * d;

                for (k = i + 1; k < m; k++)
                    A[j * astep + k] += alpha * A[i * astep + k];

                if (b)
                    for (k = 0; k < n; k++)
                        b[j * bstep + k] += alpha * b[i * bstep + k];
            }
        }

        if (b)
        {
            for (i = m - 1; i >= 0; i--)
                for (j = 0; j < n; j++)
                {
                    Type s = b[i * bstep + j];
                    for (k = i + 1; k < m; k++)
                        s -= A[i * astep + k] * b[k * bstep + j];
                    b[i * bstep + j] = s / A[i * astep + i];
                }
        }

        return p;
    }

    template<typename Type> 
    static inline bool CholImpl(Type* A, size_t astep, int m, Type* b, size_t bstep, int n)
    {
        Type* L = A;
        int i, j, k;
        double s;
        astep /= sizeof(A[0]);
        bstep /= sizeof(b[0]);

        for (i = 0; i < m; i++)
        {
            for (j = 0; j < i; j++)
            {
                s = A[i * astep + j];
                for (k = 0; k < j; k++)
                    s -= L[i * astep + k] * L[j * astep + k];
                L[i * astep + j] = (Type)(s * L[j * astep + j]);
            }
            s = A[i * astep + i];
            for (k = 0; k < j; k++)
            {
                double t = L[i * astep + k];
                s -= t * t;
            }
            if (s < std::numeric_limits<Type>::epsilon())
                return false;
            L[i * astep + i] = (Type)(1. / std::sqrt(s));
        }

        if (!b)
        {
            for (i = 0; i < m; i++)
                L[i * astep + i] = 1 / L[i * astep + i];
            return true;
        }

        // LLt x = b
        // 1: L y = b
        // 2. Lt x = y

        /*
         [ L00             ]  y0   b0
         [ L10 L11         ]  y1 = b1
         [ L20 L21 L22     ]  y2   b2
         [ L30 L31 L32 L33 ]  y3   b3

         [ L00 L10 L20 L30 ]  x0   y0
         [     L11 L21 L31 ]  x1 = y1
         [         L22 L32 ]  x2   y2
         [             L33 ]  x3   y3
         */

        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                s = b[i * bstep + j];
                for (k = 0; k < i; k++)
                    s -= L[i * astep + k] * b[k * bstep + j];
                b[i * bstep + j] = (Type)(s * L[i * astep + i]);
            }
        }

        for (i = m - 1; i >= 0; i--)
        {
            for (j = 0; j < n; j++)
            {
                s = b[i * bstep + j];
                for (k = m - 1; k > i; k--)
                    s -= L[k * astep + i] * b[k * bstep + j];
                b[i * bstep + j] = (Type)(s * L[i * astep + i]);
            }
        }
        for (i = 0; i < m; i++)
            L[i * astep + i] = 1 / L[i * astep + i];

        return true;
    }

    bool LU(float* A, size_t astep, int m, float* b, size_t bstep, int n)
    {
        return LUImpl(A, astep, m, b, bstep, n, FLT_EPSILON * 10);
    }
    bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n)
    {
        return CholImpl(A, astep, m, b, bstep, n);
    }


    template<class Type>
    bool JacobiImpl(Type* A, size_t astep, Type* W, Type* V, size_t vstep, int n, uchar* buf)
    {
        const Type eps = std::numeric_limits<Type>::epsilon();
        int i, j, k, m;

        astep /= sizeof(A[0]);
        if (V)
        {
            vstep /= sizeof(V[0]);
            for (i = 0; i < n; i++)
            {
                for (j = 0; j < n; j++)
                    V[i * vstep + j] = (Type)0;
                V[i * vstep + i] = (Type)1;
            }
        }

        int iters, maxIters = n * n * 30;

        int* indR = (int*)AlignPtr(buf, sizeof(int));
        int* indC = indR + n;
        Type mv = (Type)0;

        for (k = 0; k < n; k++)
        {
            W[k] = A[(astep + 1) * k];
            if (k < n - 1)
            {
                for (m = k + 1, mv = std::abs(A[astep * k + m]), i = k + 2; i < n; i++)
                {
                    Type val = std::abs(A[astep * k + i]);
                    if (mv < val)
                        mv = val, m = i;
                }
                indR[k] = m;
            }
            if (k > 0)
            {
                for (m = 0, mv = std::abs(A[k]), i = 1; i < k; i++)
                {
                    Type val = std::abs(A[astep * i + k]);
                    if (mv < val)
                        mv = val, m = i;
                }
                indC[k] = m;
            }
        }

        if (n > 1) for (iters = 0; iters < maxIters; iters++)
        {
            // find index (k,l) of pivot p
            for (k = 0, mv = std::abs(A[indR[0]]), i = 1; i < n - 1; i++)
            {
                Type val = std::abs(A[astep * i + indR[i]]);
                if (mv < val)
                    mv = val, k = i;
            }
            int l = indR[k];
            for (i = 1; i < n; i++)
            {
                Type val = std::abs(A[astep * indC[i] + i]);
                if (mv < val)
                    mv = val, k = indC[i], l = i;
            }

            Type p = A[astep * k + l];
            if (std::abs(p) <= eps)
                break;
            Type y = (Type)((W[l] - W[k]) * 0.5);
            Type t = std::abs(y) + hypot(p, y);
            Type s = hypot(p, t);
            Type c = t / s;
            s = p / s; t = (p / t) * p;
            if (y < 0)
                s = -s, t = -t;
            A[astep * k + l] = 0;

            W[k] -= t;
            W[l] += t;

            Type a0, b0;

            //#undef rotate
            //#define rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c
            auto Rotate = [&](Type& v0, Type& v1) { a0 = v0; b0 = v1; v0 = a0 * c - b0 * s; v1 = a0 * s + b0 * c; };

            // rotate rows and columns k and l
            for (i = 0; i < k; i++)
                Rotate(A[astep * i + k], A[astep * i + l]);
            for (i = k + 1; i < l; i++)
                Rotate(A[astep * k + i], A[astep * i + l]);
            for (i = l + 1; i < n; i++)
                Rotate(A[astep * k + i], A[astep * l + i]);

            // rotate eigenvectors
            if (V)
                for (i = 0; i < n; i++)
                    Rotate(V[vstep * k + i], V[vstep * l + i]);

            //#undef rotate

            for (j = 0; j < 2; j++)
            {
                int idx = j == 0 ? k : l;
                if (idx < n - 1)
                {
                    for (m = idx + 1, mv = std::abs(A[astep * idx + m]), i = idx + 2; i < n; i++)
                    {
                        Type val = std::abs(A[astep * idx + i]);
                        if (mv < val)
                            mv = val, m = i;
                    }
                    indR[idx] = m;
                }
                if (idx > 0)
                {
                    for (m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++)
                    {
                        Type val = std::abs(A[astep * i + idx]);
                        if (mv < val)
                            mv = val, m = i;
                    }
                    indC[idx] = m;
                }
            }
        }

        // sort eigenvalues & eigenvectors
        for (k = 0; k < n - 1; k++)
        {
            m = k;
            for (i = k + 1; i < n; i++)
            {
                if (W[m] < W[i])
                    m = i;
            }
            if (k != m)
            {
                std::swap(W[m], W[k]);
                if (V)
                    for (i = 0; i < n; i++)
                        std::swap(V[vstep * m + i], V[vstep * k + i]);
            }
        }

        return true;
    }


    bool Eigen(const InputArray& _src, const OutputArray& _evals, const OutputArray& _evects)
    {
        Tensor src = _src.GetTensor();
        //int type = src.type();
        int n = src.shape[0];

        CHECK_EQ(src.shape[0], src.shape[1]);
        CHECK_EQ(src.depth, Depth::D4);

        Tensor v;
        if (_evects.Needed())
        {
            _evects.Create({ n, n }, src.depth, src.packing, src.allocator);
            v = _evects.GetTensor();
        }

        size_t esz = 1 * src.depth, astep = AlignSize(n * esz, 16);
        AutoBuffer<uchar> buf(n * astep + n * 5LL * esz + 32);
        uchar* ptr = AlignPtr(buf.data(), 16);
        Tensor a({ n, n }, src.depth, src.packing, ptr, { astep / esz,  1ULL }), w({ n, 1 }, src.depth, src.packing, ptr + astep * n);
        ptr += astep * n + esz * n;
        src.CopyTo(a);

        bool ok = JacobiImpl<float>(a, astep, w, v, v.steps[0] * esz, n, ptr);
        w.CopyTo(_evals);
        return ok;
    }

    int givens(float* a, float* b, int n, float c, float s)
    {
#if 0
        if (n < v_float32x4::nlanes)
            return 0;
        int k = 0;
        v_float32x4 c4 = vx_setall_f32(c), s4 = vx_setall_f32(s);
        for (; k <= n - v_float32x4::nlanes; k += v_float32x4::nlanes)
        {
            v_float32x4 a0 = vx_load(a + k);
            v_float32x4 b0 = vx_load(b + k);
            v_float32x4 t0 = (a0 * c4) + (b0 * s4);
            v_float32x4 t1 = (b0 * c4) - (a0 * s4);
            v_store(a + k, t0);
            v_store(b + k, t1);
        }
        vx_cleanup();
        return k;
#endif
        if (n < 4)
            return 0;
        int k = 0;
        for (; k <= n - 4; k += 4)
        {
            float& a0 = a[k];
            float& a1 = a[k+1];
            float& a2 = a[k+2];
            float& a3 = a[k+3];

            float& b0 = b[k];
            float& b1 = b[k+1];
            float& b2 = b[k+2];
            float& b3 = b[k+3];

            float t00 = a0 * c + b0 * s;
            float t01 = a1 * c + b1 * s;
            float t02 = a2 * c + b2 * s;
            float t03 = a3 * c + b3 * s;

            float t10 = b0 * c - a0 * s;
            float t11 = b1 * c - a1 * s;
            float t12 = b2 * c - a2 * s;
            float t13 = b3 * c - a3 * s;

            a0 = t00;
            a1 = t01;
            a2 = t02;
            a3 = t03;

            b0 = t10;
            b1 = t11;
            b2 = t12;
            b3 = t13;
        }
        return k;
    }

    template<typename Type>
    void JacobiSVDImpl(Type* At, size_t astep, Type* _W, Type* Vt, size_t vstep,
            int m, int n, int n1, double minval, Type eps)
    {
        //VBLAS<Type> vblas;
        AutoBuffer<double> Wbuf(n);
        double* W = Wbuf.data();
        int i, j, k, iter, max_iter = std::max(m, 30);
        Type c, s;
        double sd;
        astep /= sizeof(At[0]);
        vstep /= sizeof(Vt[0]);

        for (i = 0; i < n; i++)
        {
            for (k = 0, sd = 0; k < m; k++)
            {
                Type t = At[i * astep + k];
                sd += (double)t * t;
            }
            W[i] = sd;

            if (Vt)
            {
                for (k = 0; k < n; k++)
                    Vt[i * vstep + k] = 0;
                Vt[i * vstep + i] = 1;
            }
        }

        for (iter = 0; iter < max_iter; iter++)
        {
            bool changed = false;

            for (i = 0; i < n - 1; i++)
                for (j = i + 1; j < n; j++)
                {
                    Type* Ai = At + i * astep, * Aj = At + j * astep;
                    double a = W[i], p = 0, b = W[j];

                    for (k = 0; k < m; k++)
                        p += (double)Ai[k] * Aj[k];

                    if (std::abs(p) <= eps * std::sqrt((double)a * b))
                        continue;

                    p *= 2;
                    double beta = a - b, gamma = hypot((double)p, beta);
                    if (beta < 0)
                    {
                        double delta = (gamma - beta) * 0.5;
                        s = (Type)std::sqrt(delta / gamma);
                        c = (Type)(p / (gamma * s * 2));
                    }
                    else
                    {
                        c = (Type)std::sqrt((gamma + beta) / (gamma * 2));
                        s = (Type)(p / (gamma * c * 2));
                    }

                    a = b = 0;
                    for (k = 0; k < m; k++)
                    {
                        Type t0 = c * Ai[k] + s * Aj[k];
                        Type t1 = -s * Ai[k] + c * Aj[k];
                        Ai[k] = t0; Aj[k] = t1;

                        a += (double)t0 * t0; b += (double)t1 * t1;
                    }
                    W[i] = a; W[j] = b;

                    changed = true;

                    if (Vt)
                    {
                        Type* Vi = Vt + i * vstep, * Vj = Vt + j * vstep;
                        k = givens(Vi, Vj, n, c, s);

                        for (; k < n; k++)
                        {
                            Type t0 = c * Vi[k] + s * Vj[k];
                            Type t1 = -s * Vi[k] + c * Vj[k];
                            Vi[k] = t0; Vj[k] = t1;
                        }
                    }
                }
            if (!changed)
                break;
        }

        for (i = 0; i < n; i++)
        {
            for (k = 0, sd = 0; k < m; k++)
            {
                Type t = At[i * astep + k];
                sd += (double)t * t;
            }
            W[i] = std::sqrt(sd);
        }

        for (i = 0; i < n - 1; i++)
        {
            j = i;
            for (k = i + 1; k < n; k++)
            {
                if (W[j] < W[k])
                    j = k;
            }
            if (i != j)
            {
                std::swap(W[i], W[j]);
                if (Vt)
                {
                    for (k = 0; k < m; k++)
                        std::swap(At[i * astep + k], At[j * astep + k]);

                    for (k = 0; k < n; k++)
                        std::swap(Vt[i * vstep + k], Vt[j * vstep + k]);
                }
            }
        }

        for (i = 0; i < n; i++)
            _W[i] = (Type)W[i];

        if (!Vt)
            return;

        //RNG rng(0x12345678);
        uint64 state = 0x12345678;
        for (i = 0; i < n1; i++)
        {
            sd = i < n ? W[i] : 0;

            for (int ii = 0; ii < 100 && sd <= minval; ii++)
            {
                // if we got a zero singular value, then in order to get the corresponding left singular vector
                // we generate a random vector, project it to the previously computed left singular vectors,
                // subtract the projection and normalize the difference.
                const Type val0 = (Type)(1. / m);
                for (k = 0; k < m; k++)
                {
                    // RNG::next()
                    state = (uint64)(unsigned)state * /*CV_RNG_COEFF*/ 4164903690U + (unsigned)(state >> 32);
                    Type val = ((unsigned)state & 256) != 0 ? val0 : -val0;
                    At[i * astep + k] = val;
                }
                for (iter = 0; iter < 2; iter++)
                {
                    for (j = 0; j < i; j++)
                    {
                        sd = 0;
                        for (k = 0; k < m; k++)
                            sd += At[i * astep + k] * At[j * astep + k];
                        Type asum = 0;
                        for (k = 0; k < m; k++)
                        {
                            Type t = (Type)(At[i * astep + k] - sd * At[j * astep + k]);
                            At[i * astep + k] = t;
                            asum += std::abs(t);
                        }
                        asum = asum > eps * 100 ? 1 / asum : 0;
                        for (k = 0; k < m; k++)
                            At[i * astep + k] *= asum;
                    }
                }
                sd = 0;
                for (k = 0; k < m; k++)
                {
                    Type t = At[i * astep + k];
                    sd += (double)t * t;
                }
                sd = std::sqrt(sd);
            }

            s = (Type)(sd > minval ? 1 / sd : 0.);
            for (k = 0; k < m; k++)
                At[i * astep + k] *= s;
        }
    }

    void JacobiSVD(float* At, size_t astep, float* W, float* Vt, size_t vstep, int m, int n, int n1)
    {
        JacobiSVDImpl<float>(At, astep, W, Vt, vstep, m, n, !Vt ? 0 : n1 < 0 ? n : n1, FLT_MIN, FLT_EPSILON * 2);
    }

    /* y[0:m,0:n] += diag(a[0:1,0:m]) * x[0:m,0:n] */
    template<typename T1, typename T2, typename T3>
    static void MatrAXPY(int m, int n, const T1* x, int dx,
            const T2* a, int inca, T3* y, int dy)
    {
        int i;
        for (i = 0; i < m; i++, x += dx, y += dy)
        {
            T2 s = a[i * inca];
            int j = 0;
            // unroll
            for (; j <= n - 4; j += 4)
            {
                T3 t0 = (T3)(y[j] + s * x[j]);
                T3 t1 = (T3)(y[j + 1] + s * x[j + 1]);
                y[j] = t0;
                y[j + 1] = t1;
                t0 = (T3)(y[j + 2] + s * x[j + 2]);
                t1 = (T3)(y[j + 3] + s * x[j + 3]);
                y[j + 2] = t0;
                y[j + 3] = t1;
            }
            for (; j < n; j++)
                y[j] = (T3)(y[j] + s * x[j]);
        }
    }

    template<typename T> 
    static void SVBkSbImpl(int m, int n, const T* w, int incw,
            const T* u, int ldu, bool uT,
            const T* v, int ldv, bool vT,
            const T* b, int ldb, int nb,
            T* x, int ldx, double* buffer, T eps)
    {
        double threshold = 0;
        int udelta0 = uT ? ldu : 1, udelta1 = uT ? 1 : ldu;
        int vdelta0 = vT ? ldv : 1, vdelta1 = vT ? 1 : ldv;
        int i, j, nm = std::min(m, n);

        if (!b)
            nb = m;

        for (i = 0; i < n; i++)
            for (j = 0; j < nb; j++)
                x[i * ldx + j] = 0;

        for (i = 0; i < nm; i++)
            threshold += w[i * incw];
        threshold *= eps;

        // v * inv(w) * uT * b
        for (i = 0; i < nm; i++, u += udelta0, v += vdelta0)
        {
            double wi = w[i * incw];
            if ((double)std::abs(wi) <= threshold)
                continue;
            wi = 1 / wi;

            if (nb == 1)
            {
                double s = 0;
                if (b)
                    for (j = 0; j < m; j++)
                        s += u[j * udelta1] * b[j * ldb];
                else
                    s = u[0];
                s *= wi;

                for (j = 0; j < n; j++)
                    x[j * ldx] = (T)(x[j * ldx] + s * v[j * vdelta1]);
            }
            else
            {
                if (b)
                {
                    for (j = 0; j < nb; j++)
                        buffer[j] = 0;
                    MatrAXPY(m, nb, b, ldb, u, udelta1, buffer, 0);
                    for (j = 0; j < nb; j++)
                        buffer[j] *= wi;
                }
                else
                {
                    for (j = 0; j < nb; j++)
                        buffer[j] = u[j * udelta1] * wi;
                }
                MatrAXPY(n, nb, buffer, 0, v, vdelta1, x, ldx);
            }
        }
    }

    static void SVBkSb(int m, int n, const float* w, size_t wstep,
            const float* u, size_t ustep, bool uT,
            const float* v, size_t vstep, bool vT,
            const float* b, size_t bstep, int nb,
            float* x, size_t xstep, uchar* buffer)
    {
        SVBkSbImpl(m, n, w, wstep ? (int)(wstep / sizeof(w[0])) : 1,
            u, (int)(ustep / sizeof(u[0])), uT,
            v, (int)(vstep / sizeof(v[0])), vT,
            b, (int)(bstep / sizeof(b[0])), nb,
            x, (int)(xstep / sizeof(x[0])),
            (double*)AlignPtr(buffer, sizeof(double)), (float)(DBL_EPSILON * 2));
    }

    void SVDcompute(const InputArray& _aarr, const OutputArray& _w,
        const OutputArray& _u, const OutputArray& _vt, int flags)
    {
        Tensor src = _aarr.GetTensor();
        int m = src.shape[0], n = src.shape[1];
        Depth depth = src.depth;
        bool compute_uv = _u.Needed() || _vt.Needed();
        bool full_uv = (flags & SVD::FULL_UV) != 0;

        CHECK_EQ(Depth::D4, src.depth);

        if (flags & SVD::NO_UV)
        {
            _u.Release();
            _vt.Release();
            compute_uv = full_uv = false;
        }

        bool at = false;
        if (m < n)
        {
            std::swap(m, n);
            at = true;
        }
        
        int urows = full_uv ? m : n;
        size_t esz = 1 * src.depth, astep = AlignSize(m * esz, 16), vstep = AlignSize(n * esz, 16);
        AutoBuffer<uchar> _buf(urows * astep + n * vstep + n * esz + 32);
        uchar* buf = AlignPtr(_buf.data(), 16);
        Tensor temp_a(Shape(n, m), Depth::D4, Packing::CHW, buf, { astep / esz, 1ULL });
        Tensor temp_w(Shape(n, 1), Depth::D4, Packing::CHW, buf + urows * astep);
        Tensor temp_u(Shape(urows, m), Depth::D4, Packing::CHW, buf, { astep / esz, 1ULL }), temp_v;

        if (compute_uv)
            temp_v = Tensor(Shape(n, n), Depth::D4, Packing::CHW, AlignPtr(buf + urows * astep + n * esz, 16), { vstep / esz, 1ULL });

        if (urows > n)
            memset(temp_u, 0, esz * temp_u.shape[0] * temp_u.steps[0]);
            //temp_u = Scalar::all(0);

        if (!at)
            Transpose(src, temp_a);
        else
            src.CopyTo(temp_a);

        JacobiSVD(temp_a, astep, temp_w,
            temp_v, vstep, m, n, compute_uv ? urows : 0);

        temp_w.CopyTo(_w);
        if (compute_uv)
        {
            if (!at)
            {
                Transpose(temp_u, _u);
                temp_v.CopyTo(_vt);
            }
            else
            {
                Transpose(temp_v, _u);
                temp_u.CopyTo(_vt);
            }
        }
    }

    void SVD::Compute(const InputArray& A, const OutputArray& w, const OutputArray& u, const OutputArray& vt, int flags)
    {
        SVDcompute(A, w, u, vt, flags);
    }

    void SVD::BackSubst(const InputArray& _w, const InputArray& _u,
        const InputArray& _vt, const InputArray& _rhs,
        const OutputArray& _dst)
    {
        Tensor w = _w.GetTensor(), u = _u.GetTensor(), vt = _vt.GetTensor(), rhs = _rhs.GetTensor();
        int esz = 1 * w.depth;
        auto depth = w.depth;
        auto packing = w.packing;
        auto allocator = w.allocator;

        uint m = u.shape[0], n = vt.shape[1], nb = rhs.empty() ? m : rhs.shape[1], nm = std::min(m, n);
        size_t wstep = w.shape[0] == 1 ? (size_t)esz : w.shape[1] == 1 ? (size_t)w.steps[0] * esz : (size_t)w.steps[0] * esz + esz;
        AutoBuffer<uchar> buffer(nb * sizeof(double) + 16);

        CHECK(w.depth == u.depth && u.depth == vt.depth && u.data && vt.data && w.data);
        CHECK(u.shape[1] >= nm && vt.shape[0] >= nm &&
            (w.shape == Shape(nm, 1) || w.shape == Shape(1, nm) || w.shape == Shape(vt.shape[0], u.shape[1])));
        CHECK(rhs.data == 0 || (rhs.depth == depth && rhs.shape[0] == m));

        _dst.Create({ n, nb }, depth, packing, allocator);
        Tensor dst = _dst.GetTensor();

        SVBkSb(m, n, w, wstep, u, u.steps[0] * esz, false,
            vt, vt.steps[0] * esz, true, rhs, rhs.empty() ? 0 : rhs.steps[0] * esz, nb,
            dst, dst.steps[0] * esz, buffer.data());
    }

    bool Invert(const InputArray& _src, const OutputArray& _dst, int method)
    {
        Tensor src = _src.GetTensor();
        size_t esz = 1 * src.depth;
        auto depth = src.depth;
        auto packing = src.packing;
        auto allocator = src.allocator;

        int m = src.shape[0], n = src.shape[1];
        
        if (method == DECOMP_SVD)
        {
            int nm = std::min(m, n);

            AutoBuffer<uchar> _buf((m * nm + nm + nm * n) * esz + sizeof(double));
            uchar* buf = AlignPtr((uchar*)_buf.data(), (int)esz);
            Tensor u({ m, nm }, depth, packing, buf);
            Tensor w({ nm, 1 }, depth, packing, (uchar*)u.data + esz * m * nm);
            Tensor vt({ nm, n }, depth, packing, (uchar*)w + nm * esz);

            SVD::Compute(src, w, u, vt);
            SVD::BackSubst(w, u, vt, Tensor(), _dst);

            return (w[0] >= FLT_EPSILON ?
                w[n - 1LL] / w[0] : 0);
        }

        CHECK_EQ(m, n);

        if (method == DECOMP_EIG)
        {
            AutoBuffer<uchar> _buf((n * n * 2LL + n) * esz + sizeof(double));
            uchar* buf = AlignPtr((uchar*)_buf.data(), (int)esz);
            Tensor u({ n, n }, depth, packing, buf);
            Tensor w({ n, 1 }, depth, packing, (uchar*)u.data + n * n * esz);
            Tensor vt({ n, n }, depth, packing, (uchar*)w.data + n * esz);
             
            Eigen(src, w, vt);
            Transpose(vt, u);
            SVD::BackSubst(w, u, vt, Tensor(), _dst);

            return (w[0] >= FLT_EPSILON ?
                w[n - 1LL] / w[0] : 0);
        }

        CHECK(method == DECOMP_LU || method == DECOMP_CHOLESKY);

        _dst.Create({ n, n }, depth, packing, allocator);
        Tensor dst = _dst.GetTensor();

        if (n <= 3)
        {
            bool result = false;
            const uchar* srcdata = src;
            uchar* dstdata = dst;
            size_t srcstep = src.steps[0] * esz;
            size_t dststep = dst.steps[0] * esz;

            auto Sf = [=](int y, int x)->const float& { return ((float*)(srcdata + y * srcstep))[x]; };
            auto Df = [=](int y, int x)->float& { return ((float*)(dstdata + y * dststep))[x]; };

            if (n == 2)
            {
                // det2
                double d = (double)Sf(0, 0) * Sf(1, 1) - (double)Sf(0, 1) * Sf(1, 0);
                if (d != 0.)
                {
                    result = true;
                    d = 1. / d;
                    double t0, t1;
                    t0 = Sf(0, 0) * d;
                    t1 = Sf(1, 1) * d;
                    Df(1, 1) = (float)t0;
                    Df(0, 0) = (float)t1;
                    t0 = -Sf(0, 1) * d;
                    t1 = -Sf(1, 0) * d;
                    Df(0, 1) = (float)t0;
                    Df(1, 0) = (float)t1;
                }
            }
            else if (n == 3)
            {
                // det3
                double d = 
                    Sf(0, 0) * ((double)Sf(1, 1) * Sf(2, 2) - (double)Sf(1, 2) * Sf(2, 1)) - 
                    Sf(0, 1) * ((double)Sf(1, 0) * Sf(2, 2) - (double)Sf(1, 2) * Sf(2, 0)) + 
                    Sf(0, 2) * ((double)Sf(1, 0) * Sf(2, 1) - (double)Sf(1, 1) * Sf(2, 0));

                if (d != 0.)
                {
                    double t[12];

                    result = true;
                    d = 1. / d;
                    t[0] = (((double)Sf(1, 1) * Sf(2, 2) - (double)Sf(1, 2) * Sf(2, 1)) * d);
                    t[1] = (((double)Sf(0, 2) * Sf(2, 1) - (double)Sf(0, 1) * Sf(2, 2)) * d);
                    t[2] = (((double)Sf(0, 1) * Sf(1, 2) - (double)Sf(0, 2) * Sf(1, 1)) * d);

                    t[3] = (((double)Sf(1, 2) * Sf(2, 0) - (double)Sf(1, 0) * Sf(2, 2)) * d);
                    t[4] = (((double)Sf(0, 0) * Sf(2, 2) - (double)Sf(0, 2) * Sf(2, 0)) * d);
                    t[5] = (((double)Sf(0, 2) * Sf(1, 0) - (double)Sf(0, 0) * Sf(1, 2)) * d);

                    t[6] = (((double)Sf(1, 0) * Sf(2, 1) - (double)Sf(1, 1) * Sf(2, 0)) * d);
                    t[7] = (((double)Sf(0, 1) * Sf(2, 0) - (double)Sf(0, 0) * Sf(2, 1)) * d);
                    t[8] = (((double)Sf(0, 0) * Sf(1, 1) - (double)Sf(0, 1) * Sf(1, 0)) * d);

                    Df(0, 0) = (float)t[0]; Df(0, 1) = (float)t[1]; Df(0, 2) = (float)t[2];
                    Df(1, 0) = (float)t[3]; Df(1, 1) = (float)t[4]; Df(1, 2) = (float)t[5];
                    Df(2, 0) = (float)t[6]; Df(2, 1) = (float)t[7]; Df(2, 2) = (float)t[8];
                }
            }
            else
            {
                CHECK_EQ(n, 1);
                double d = Sf(0, 0);
                if (d != 0.)
                {
                    result = true;
                    Df(0, 0) = (float)(1. / d);
                }
            }

            return result;
        }

        AutoBuffer<uchar> buf(esz * n * n);
        Tensor src1({ n, n }, depth, packing,  buf.data());
        src.CopyTo(src1);
        SetIdentity(dst);

        if (method == DECOMP_LU)
        {
            return LU(src1, src1.steps[0] * esz, n, dst, dst.steps[0] * esz, n);
        }
        if (method == DECOMP_CHOLESKY)
        {
            return Cholesky(src1, src1.steps[0] * esz, n, dst, dst.steps[0] * esz, n);
        }

        return false;
    }
}