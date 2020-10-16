#pragma once

#include "dnn/layer.hpp"

namespace chaos
{
	namespace dnn
	{
		class BinaryOp : public Layer
		{
		public:
			BinaryOp();

			virtual void Set(const std::string& key, const ParamValue& value) override;

			virtual void Forward(const std::vector<Tensor>& bottoms, std::vector<Tensor>& tops, const Option& opt) const override;

			enum OpType
			{
				ADD,
				MUL,
				SUB,
				DIV,
			};

			int op_type;
		};
	}
}