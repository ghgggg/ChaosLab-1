#include "dnn/layer_factory.hpp"

#include "dnn/layers/binary_op.hpp"
namespace chaos
{
	namespace dnn
	{
		REGISTER_LAYER("BinaryOp", BinaryOp);
	}
}