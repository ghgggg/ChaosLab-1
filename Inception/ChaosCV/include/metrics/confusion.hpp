#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

#include <set>

namespace chaos
{
	namespace metrics
	{
		class CHAOS_API ConfusionMatrix
		{
		public:
			ConfusionMatrix();

			void operator()(bool is_pos, float score);
			void operator()(const Tensor& is_pos, const Tensor& score);

			void Apply();

			Tensor GetACC() const;
			Tensor GetTPR() const;
			Tensor GetFPR() const;
			Tensor GetPPV() const;
			Tensor GetThreshold() const;

		private:
			std::multiset<float, std::greater<float>> neg;
			std::multiset<float, std::greater<float>> pos;
			std::set<float, std::greater<float>> threshold;
			Tensor table;
		};
	}
}