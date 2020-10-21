#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		// y=w*x^t+b
		class InnerProduct : public Layer
		{
		public:
			InnerProduct();

			virtual void Set(const std::string& key, const ParamValue& val) override;

			virtual void Forward(const Tensor& bottom, Tensor& top, const Option& opt) const override;

			Tensor weight; // MxN
			Tensor bias;
			// 0=none, 1=relu, 2=leakyrelu, 3=clip, 4=sigmoid, 5=mish
			int activation_type = 0;
			Tensor activation_params;
		};
	}
}