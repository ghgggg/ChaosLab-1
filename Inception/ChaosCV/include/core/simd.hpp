#pragma once

#include "core.hpp"

namespace chaos
{
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

    inline void vx_cleanup() {}
}