#pragma once

#include "innerproduct.hpp"

#include "core/vulkan/vk_tensor.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API InnerProductVulkan : virtual public InnerProduct
		{
		public:
			InnerProductVulkan();

			virtual void CreatePipeline(const Option& opt) override;
			virtual void DestroyPipeline(const Option& opt) override;

			virtual void UploadModel(VkTransfer& cmd, const Option& opt);

			using InnerProduct::Forward;
			virtual void Forward(const VkTensor& bottom, VkTensor& top, VkCompute& cmd, const Option& opt) const;

			VkTensor vk_weight;
			VkTensor vk_bias;

			Pipeline* pipeline_innerproduct;
		};
	}
}