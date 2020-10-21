#include "dnn/layers/noop.hpp"

namespace chaos
{
	namespace dnn
	{
		Noop::Noop() : Layer("Noop")
		{
			support_inplace = true;
		}

		void Noop::Forward(Tensor& blob, const Option& opt) const { return; }
		void Noop::Forward(std::vector<Tensor>& blobs, const Option& opt) const { return; }
	}
}