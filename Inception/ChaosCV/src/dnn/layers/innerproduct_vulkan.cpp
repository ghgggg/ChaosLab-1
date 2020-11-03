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
			std::vector<VkSpecializationType> specializations(4 + 8);
			specializations[0].i = bias.empty();
			specializations[1].i = activation_type;
			specializations[2].f = 0;
			specializations[3].f = 0;
			specializations[4 + 0].i = 2;
			specializations[4 + 1].i = 3;
			specializations[4 + 2].i = 3;
			specializations[4 + 3].i = 1;
			specializations[4 + 4].i = 2;
			specializations[4 + 5].i = 2;
			specializations[4 + 6].i = 3;
			specializations[4 + 7].i = 1;

			pipeline_innerproduct = new Pipeline(vkdev);
			pipeline_innerproduct->SetOptimalLocalSizeXYZ(3, 1, 1);
			pipeline_innerproduct->Create(LayerShaderRegistry::GetIndex("innerproduct"), opt, specializations);
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
			Shape oshape = { 3,2 };
			top.Create(oshape, oshape.steps(), Depth::D4, Packing::CHW, opt.blob_vkallocator);

			//VkTensor bottom_shape;
			//VkTensor bottom_steps;
			//VkTensor top_shape;
			//VkTensor top_steps;

			//cmd.RecordUpload(Tensor(bottom.shape.size(), Depth::D4, Packing::CHW, (void*)bottom.shape.data()), bottom_shape, opt);
			//cmd.RecordUpload(Tensor(bottom.steps.size(), Depth::D4, Packing::CHW, (void*)bottom.steps.data()), bottom_steps, opt);
			//cmd.RecordUpload(Tensor(top.shape.size(), Depth::D4, Packing::CHW, (void*)top.shape.data()), top_shape, opt);
			//cmd.RecordUpload(Tensor(top.steps.size(), Depth::D4, Packing::CHW, (void*)top.steps.data()), top_shape, opt);
			//cmd.SubmitAndWait();

			std::vector<VkTensor> bindings(4);
			bindings[0] = bottom;
			bindings[1] = top;
			bindings[2] = vk_weight;
			bindings[3] = vk_weight;
			//bindings[4] = bottom_shape;
			//bindings[5] = bottom_steps;
			//bindings[6] = top_shape;
			//bindings[7] = top_steps;

			std::vector<VkConstantType> constants(8);
			constants[0].i = bottom.shape.size();
			constants[1].i = bottom.shape.GetX();
			constants[2].i = bottom.shape.GetY();
			constants[3].i = bottom.shape.GetZ();
			constants[4].i = top.shape.size();
			constants[5].i = top.shape.GetX();
			constants[6].i = top.shape.GetY();
			constants[7].i = top.shape.GetZ();

			cmd.RecordPipeline(pipeline_innerproduct, bindings, constants, top.shape);
		}
	}
}