#include "math/base.hpp"

namespace chaos
{

#if 0
    template<class Type, std::enable_if_t<std::is_floating_point_v<Type>, bool> = true>
    static inline bool LUImpl(Type* A, size_t astep, int m, Type* b, size_t bstep, int n, Type eps)
    {
        int i, j, k;
        bool ret = true;
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
#endif
}