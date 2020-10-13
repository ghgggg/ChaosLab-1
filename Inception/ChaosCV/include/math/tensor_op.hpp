#pragma once

#include "core/core.hpp"
#include "core/vec.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	CHAOS_API void SetIdentity(Tensor& tensor, double val = 1.);
	CHAOS_API void Transpose(const Tensor& src, Tensor& dst); // with inplace 
	CHAOS_API void Permute(const Tensor& src, Tensor& dst, const Vec<uint>& orders);
}