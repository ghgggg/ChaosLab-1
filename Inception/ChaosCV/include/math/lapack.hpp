#pragma once

#include "core/core.hpp"

namespace chaos
{
    template<class Type, std::enable_if_t<std::is_floating_point_v<Type>, bool> = true>
    static inline bool LUImpl(Type* A, size_t astep, int m, Type* b, size_t bstep, int n, Type eps)
    {
        int i, j, k;
        bool ret = true;
        //astep /= sizeof(A[0]);
        //bstep /= sizeof(b[0]); 

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
                ret = !ret;
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

        return ret;
    }

    template<class Type, std::enable_if_t<std::is_floating_point_v<Type>, bool> = true>
    static inline bool CholeskyImpl(Type* A, size_t astep, int m, Type* b, size_t bstep, int n)
    {
        Type* L = A;
        int i, j, k;
        double s;
        //astep /= sizeof(A[0]);
        //bstep /= sizeof(b[0]);

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

    template<typename Type> 
    inline static int sign(Type x)
    {
        if (x >= (Type)0)
            return 1;
        else
            return -1;
    }

    template<class Type, std::enable_if_t<std::is_floating_point_v<Type>, bool> = true>
    static inline bool QRImpl(Type* A, size_t astep, int m, int n, int k, Type* b, size_t bstep, Type* hFactors, Type eps)
    {
        AutoBuffer<Type> buffer;
        size_t buf_size = m ? m + n : hFactors != NULL;
        buffer.Allocate(buf_size);
        Type* vl = buffer.data();
        if (hFactors == NULL)
            hFactors = vl + m;

        for (int l = 0; l < n; l++)
        {
            //generate vl
            int vlSize = m - l;
            Type vlNorm = (Type)0;
            for (int i = 0; i < vlSize; i++)
            {
                vl[i] = A[(l + i) * astep + l];
                vlNorm += vl[i] * vl[i];
            }
            Type tmpV = vl[0];
            vl[0] = vl[0] + sign(vl[0]) * std::sqrt(vlNorm);
            vlNorm = std::sqrt(vlNorm + vl[0] * vl[0] - tmpV * tmpV);
            for (int i = 0; i < vlSize; i++)
            {
                vl[i] /= vlNorm;
            }
            //multiply A_l*vl
            for (int j = l; j < n; j++)
            {
                Type v_lA = (Type)0;
                for (int i = l; i < m; i++)
                {
                    v_lA += vl[i - l] * A[i * astep + j];
                }

                for (int i = l; i < m; i++)
                {
                    A[i * astep + j] -= 2 * vl[i - l] * v_lA;
                }
            }

            //save vl and factors
            hFactors[l] = vl[0] * vl[0];
            for (int i = 1; i < vlSize; i++)
            {
                A[(l + i) * astep + l] = vl[i] / vl[0];
            }
        }

        if (b)
        {
            //generate new rhs
            for (int l = 0; l < n; l++)
            {
                //unpack vl
                vl[0] = (Type)1;
                for (int j = 1; j < m - l; j++)
                {
                    vl[j] = A[(j + l) * astep + l];
                }

                //h_l*x
                for (int j = 0; j < k; j++)
                {
                    Type v_lB = (Type)0;
                    for (int i = l; i < m; i++)
                        v_lB += vl[i - l] * b[i * bstep + j];

                    for (int i = l; i < m; i++)
                        b[i * bstep + j] -= 2 * vl[i - l] * v_lB * hFactors[l];
                }
            }
            //do back substitution
            for (int i = n - 1; i >= 0; i--)
            {
                for (int j = n - 1; j > i; j--)
                {
                    for (int p = 0; p < k; p++)
                        b[i * bstep + p] -= b[j * bstep + p] * A[i * astep + j];
                }
                if (std::abs(A[i * astep + i]) < eps)
                    return false;
                for (int p = 0; p < k; p++)
                    b[i * bstep + p] /= A[i * astep + i];
            }
        }

        return true;
    }


    template<typename Type, std::enable_if_t<std::is_floating_point_v<Type>, bool> = true>
    bool JacobiImpl(Type* A, size_t astep, Type* W, Type* V, size_t vstep, int n, uchar* buf)
    {
        const Type eps = std::numeric_limits<Type>::epsilon();
        int i, j, k, m;

        //astep /= sizeof(A[0]);
        if (V)
        {
            //vstep /= sizeof(V[0]);
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
            Type s = std::hypot(p, t);
            Type c = t / s;
            s = p / s; t = (p / t) * p;
            if (y < 0)
                s = -s, t = -t;
            A[astep * k + l] = 0;

            W[k] -= t;
            W[l] += t;

            Type a0, b0;

#undef rotate
#define rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c

            // rotate rows and columns k and l
            for (i = 0; i < k; i++)
                rotate(A[astep * i + k], A[astep * i + l]);
            for (i = k + 1; i < l; i++)
                rotate(A[astep * k + i], A[astep * i + l]);
            for (i = l + 1; i < n; i++)
                rotate(A[astep * k + i], A[astep * l + i]);

            // rotate eigenvectors
            if (V)
                for (i = 0; i < n; i++)
                    rotate(V[vstep * k + i], V[vstep * l + i]);
#undef rotate

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


    struct v_float32x4
    {
        typedef float lane_type;
        typedef __m128 vector_type;
        enum { nlanes = 4 };

        v_float32x4() : val(_mm_setzero_ps()) {}
        explicit v_float32x4(__m128 v) : val(v) {}
        v_float32x4(float v0, float v1, float v2, float v3)
        {
            val = _mm_setr_ps(v0, v1, v2, v3);
        }
        float get0() const
        {
            return _mm_cvtss_f32(val);
        }

        __m128 val;
    };

    inline v_float32x4 vx_setall_f32(float v) { return v_float32x4(_mm_set_ps1((float)v)); }
    inline v_float32x4 vx_load(const float* ptr) { return v_float32x4(_mm_loadu_ps(ptr)); }
    inline void v_store(float* ptr, const v_float32x4& a) { _mm_storeu_ps(ptr, a.val); };
    inline v_float32x4 operator*(const v_float32x4& a, const v_float32x4& b) { return v_float32x4(_mm_mul_ps(a.val, b.val)); }
    inline v_float32x4 operator+(const v_float32x4& a, const v_float32x4& b) { return v_float32x4(_mm_add_ps(a.val, b.val)); }
    inline v_float32x4 operator-(const v_float32x4& a, const v_float32x4& b) { return v_float32x4(_mm_sub_ps(a.val, b.val)); }

    inline void vx_cleanup() {} //??????

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

    template<typename Type, std::enable_if_t<std::is_floating_point_v<Type>, bool> = true> 
    void JacobiSVDImpl(Type* At, size_t astep, Type* _W, Type* Vt, size_t vstep, int m, int n, int n1, double minval, Type eps)
    {
        AutoBuffer<double> Wbuf(n);
        double* W = Wbuf.data();
        int i, j, k, iter, max_iter = std::max(m, 30);
        Type c, s;
        double sd;
        //astep /= sizeof(At[0]);
        //vstep /= sizeof(Vt[0]);

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
                    Type val = val0; //(rng.next() & 256) != 0 ? val0 : -val0;
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
    
}