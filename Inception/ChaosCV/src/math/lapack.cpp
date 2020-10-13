#include "core/simd.hpp"
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

    int givens(float* a, float* b, int n, float c, float s)
    {
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

    void SVDcompute(const Tensor& _aarr, Tensor& _w,
        Tensor& _u, Tensor& _vt, int flags)
    {
        Tensor src = _aarr;
        int m = src.shape[0], n = src.shape[1];
        Depth depth = src.depth;
        bool compute_uv = true; //_u.needed() || _vt.needed();
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
        Tensor temp_a(Shape(n, m), Depth::D4, Packing::CHW, buf, { astep, esz });
        Tensor temp_w(Shape(n, 1), Depth::D4, Packing::CHW, buf + urows * astep);
        Tensor temp_u(Shape(urows, m), Depth::D4, Packing::CHW, buf, { astep, esz }), temp_v;

        if (compute_uv)
            temp_v = Tensor(Shape(n, n), Depth::D4, Packing::CHW, AlignPtr(buf + urows * astep + n * esz, 16), { vstep, esz });

        if (urows > n)
            memset(temp_u, 0, (size_t)temp_u.shape[0] * temp_u.steps[0]);
            //temp_u = Scalar::all(0);

        if (!at)
            Transpose(src, temp_a);
        else
            src.CopyTo(temp_a);

        JacobiSVD(temp_a, temp_u.steps[0], temp_w,
            temp_v, temp_v.steps[0], m, n, compute_uv ? urows : 0);

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
}