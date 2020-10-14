#pragma once

#include "core/core.hpp"
#include "core/vec.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	

	CHAOS_API void SetIdentity(const InputOutputArray& src, double val = 1.);
	CHAOS_API void Transpose(const InputArray& src, const OutputArray& dst);
	//CHAOS_API void Permute(const Tensor& src, Tensor& dst, const Vec<uint>& orders);

	CHAOS_API void Permute(const InputArray& src, const OutputArray& dst, const Vec<uint>& orders);
}