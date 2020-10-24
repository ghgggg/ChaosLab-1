#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class Permute : public Layer
		{
		public:
			Permute();

			virtual void Set(const std::string& key, const ParamValue& value) override;
			virtual void Forward(const Tensor& bottom, Tensor& top, const Option& opt) const override;

			Vec<uint> orders;
		};
	}
}