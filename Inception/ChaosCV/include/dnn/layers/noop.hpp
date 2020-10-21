#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class Noop : public Layer
		{
		public:
			Noop();

			virtual void Forward(Tensor& blob, const Option& opt) const override;
			virtual void Forward(std::vector<Tensor>& blobs, const Option& opt) const override;
		};
	}
}