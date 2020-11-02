#pragma once

#include "core/tensor.hpp"

#include "option.hpp"
#include "model.hpp"

namespace chaos
{
	class VulkanDevice;
	class VkAllocator;
	class VkTensor;
	class VkTransfer;
	class VkCompute;
	class Pipeline;
	namespace dnn
	{
		class CHAOS_API Layer
		{
		public:
			Layer(const std::string& type);
			virtual ~Layer() = default;

			virtual void Set(const std::string& key, const ParamValue& value) {}

			virtual void CreatePipeline(const Option& opt) {}
			virtual void DestroyPipeline(const Option& opt) {}

			virtual void Forward(const std::vector<Tensor>& bottoms, std::vector<Tensor>& tops, const Option& opt) const;
			virtual void Forward(const Tensor& blob, Tensor& top, const Option& opt) const;
			virtual void Forward(std::vector<Tensor>& blobs, const Option& opt) const;
			virtual void Forward(Tensor& blob, const Option& opt) const;



			const std::string type;
			std::string name;

			bool one_blob_only;
			bool support_inplace;
			bool support_vulkan = false;

			const VulkanDevice* vkdev;

			// blob index which this layer needs as input
			std::vector<int> bottoms_idx;
			// blob index which this layer produces as output
			std::vector<int> tops_idx;

			// shape hint
			std::vector<Shape> bottoms_shapes;
			std::vector<Shape> tops_shapes;
		};

		enum ActiveType
		{
			NONE,
			RELU,
			LEAKYRELU,
			CLIP,
			SOFTMAX,
			MISH,
		};

		enum BinOpType
		{
			ADD,
			MUL,
			SUB,
			DIV,
		};
	}
}