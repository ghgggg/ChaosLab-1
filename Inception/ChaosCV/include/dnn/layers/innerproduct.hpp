#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{

		class InnerProduct : public Layer
		{
		public:
			InnerProduct();

			virtual void Set(const std::string& key, const ParamValue& val) override;

			// bottom KxM, top KxN
			virtual void Forward(const Tensor& bottom, Tensor& top, const Option& opt) const override;

			Tensor weight; // MxN
			Tensor bias;
		};
	}
}