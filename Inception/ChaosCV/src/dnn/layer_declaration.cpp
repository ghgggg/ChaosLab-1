#include "dnn/layer_factory.hpp"
#include "dnn/shader_factory.hpp"

#define NAMESPACE_BEGIN namespace chaos { namespace dnn {
#define NAMESPACE_END } }

#include "dnn/layers/binary_op.hpp"
NAMESPACE_BEGIN
REGISTER_LAYER("BinaryOp", BinaryOp);
NAMESPACE_END

#include "dnn/layers/innerproduct.hpp"
#include "dnn/layers/innerproduct_vulkan.hpp"
#include "dnn/layers/shaders/innerproduct.spv.hex.hpp"
NAMESPACE_BEGIN
class InnerProductFinal : virtual public InnerProduct, virtual public InnerProductVulkan
{
public:
	virtual void CreatePipeline(const Option& opt)
	{
		InnerProduct::CreatePipeline(opt);
		if (opt.use_vulkan_compute) { InnerProductVulkan::CreatePipeline(opt); }
	}

	virtual void DestroyPipeline(const Option& opt)
	{
		if (opt.use_vulkan_compute) { InnerProductVulkan::DestroyPipeline(opt); }
		InnerProduct::DestroyPipeline(opt);
	}
};
REGISTER_LAYER("InnerProduct", InnerProductFinal);
REGISTER_SHADER(innerproduct);
NAMESPACE_END

#include "dnn/layers/noop.hpp"
NAMESPACE_BEGIN
REGISTER_LAYER("Noop", Noop);
NAMESPACE_END

#include "dnn/layers/permute.hpp"
NAMESPACE_BEGIN
REGISTER_LAYER("Permute", Permute);
NAMESPACE_END