#pragma once

#include "core/core.hpp"
#include "core/vec.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	// c = a + b
	CHAOS_API void Add(const InputArray& a, const InputArray& b, const OutputArray& c);
	// c = a - b
	CHAOS_API void Sub(const InputArray& a, const InputArray& b, const OutputArray& c);
	// c = a .* b
	CHAOS_API void Mul(const InputArray& a, const InputArray& b, const OutputArray& c);
	// c = a ./ b
	CHAOS_API void Div(const InputArray& a, const InputArray& b, const OutputArray& c);

	CHAOS_API void SetIdentity(const InputOutputArray& src, double val = 1.);
	CHAOS_API void Transpose(const InputArray& src, const OutputArray& dst);
	//CHAOS_API void Permute(const Tensor& src, Tensor& dst, const Vec<uint>& orders);

	CHAOS_API void Permute(const InputArray& src, const OutputArray& dst, const Vec<uint>& orders);
}