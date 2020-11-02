#include "dnn/layers/innerproduct_vulkan.hpp"

#include "dnn/shader_factory.hpp"

#include "core/vulkan/command.hpp"
#include "core/vulkan/pipeline.hpp"

namespace chaos
{
	namespace dnn
	{
		InnerProductVulkan::InnerProductVulkan() : InnerProduct()
		{
			support_vulkan = true;
		}

		void InnerProductVulkan::CreatePipeline(const Option& opt)
		{
			std::vector<VkSpecializationType> specializations(4 + 10);

			pipeline_innerproduct = new Pipeline(vkdev);
			pipeline_innerproduct->Create(LayerShaderRegistry::GetIndex("InnerProduct"), opt, specializations);
		}
		void InnerProductVulkan::DestroyPipeline(const Option& opt)
		{
			delete pipeline_innerproduct;
			pipeline_innerproduct = nullptr;
		}

		void InnerProductVulkan::UploadModel(VkTransfer& cmd, const Option& opt)
		{
			cmd.RecordUpload(weight, vk_weight, opt);
			if (not bias.empty())
				cmd.RecordUpload(bias, vk_bias, opt);
		}

		void InnerProductVulkan::Forward(const VkTensor& bottom, VkTensor& top, VkCompute& cmd, const Option& opt) const
		{

		}
	}
}