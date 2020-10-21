#include "dnn/layer_factory.hpp"

#define NAMESPACE_BEGIN namespace chaos { namespace dnn {
#define NAMESPACE_END } }

#include "dnn/layers/binary_op.hpp"
NAMESPACE_BEGIN
REGISTER_LAYER("BinaryOp", BinaryOp);
NAMESPACE_END

#include "dnn/layers/innerproduct.hpp"
NAMESPACE_BEGIN
REGISTER_LAYER("InnerProduct", InnerProduct);
NAMESPACE_END

#include "dnn/layers/noop.hpp"
NAMESPACE_BEGIN
REGISTER_LAYER("Noop", Noop);
NAMESPACE_END