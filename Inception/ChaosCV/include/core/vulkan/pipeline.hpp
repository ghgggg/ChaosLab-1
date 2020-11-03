#pragma once

#include "core/core.hpp"

#include "gpu.hpp"
#include "vk_tensor.hpp"

#include <vulkan/vulkan.hpp>

#include <vector>

namespace chaos
{
    namespace dnn { class Option; }

    class CHAOS_API PipelineCache
    {
    public:
        PipelineCache(const VulkanDevice* _vkdev);

        ~PipelineCache();

        void Clear();

        void GetPipeline(const uint32_t* spv_data, size_t spv_data_size, const std::vector<VkSpecializationType>& specializations,
            uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
            VkShaderModule* shader_module,
            VkDescriptorSetLayout* descriptorset_layout,
            VkPipelineLayout* pipeline_layout,
            VkPipeline* pipeline,
            VkDescriptorUpdateTemplateKHR* descriptor_update_template,
            ShaderInfo& shader_info) const;

        void GetPipeline(int shader_type_index, const dnn::Option& opt, const std::vector<VkSpecializationType>& specializations,
            uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
            VkShaderModule* shader_module,
            VkDescriptorSetLayout* descriptorset_layout,
            VkPipelineLayout* pipeline_layout,
            VkPipeline* pipeline,
            VkDescriptorUpdateTemplateKHR* descriptor_update_template,
            ShaderInfo& shader_info) const;

    protected:
        void CreateShaderModule(int shader_type_index, const dnn::Option& opt, uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
            VkShaderModule* _shader_module, ShaderInfo& si) const;

        void NewPipeline(VkShaderModule shader_module, const ShaderInfo& shader_info, const std::vector<VkSpecializationType>& specializations,
            VkDescriptorSetLayout* descriptorset_layout,
            VkPipelineLayout* pipeline_layout,
            VkPipeline* pipeline,
            VkDescriptorUpdateTemplateKHR* descriptor_update_template) const;

    public:
        const VulkanDevice* vkdev;

        // digest -> artifact
        struct pipeline_cache_digest
        {
        public:
            pipeline_cache_digest(const uint32_t* spv_data, size_t spv_data_size, const std::vector<VkSpecializationType>& specializations,
                uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z);
            pipeline_cache_digest(int shader_type_index, const dnn::Option& opt, const std::vector<VkSpecializationType>& specializations,
                uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z);

            bool operator==(const pipeline_cache_digest& rhs) const
            {
                return d0 == rhs.d0 && d1 == rhs.d1;
            }

            bool operator!=(const pipeline_cache_digest& rhs) const
            {
                return d0 != rhs.d0 || d1 != rhs.d1;
            }

            union
            {
                struct
                {
                    union
                    {
                        uint32_t spv_data_murmur3;
                        int shader_type_index;
                    };
                    unsigned char opt_local_size_bits[4];
                };

                uint64_t d0;
            };

            union
            {
                struct
                {
                    uint32_t specializations_murmur3;
                    uint32_t specializations_fnv1a;
                };

                uint64_t d1;
            };
        };

        struct pipeline_cache_artifact
        {
        public:
            VkShaderModule shader_module;
            VkDescriptorSetLayout descriptorset_layout;
            VkPipelineLayout pipeline_layout;
            VkPipeline pipeline;
            VkDescriptorUpdateTemplateKHR descriptor_update_template;
            ShaderInfo shader_info; // TODO use pointer ?
        };

        mutable std::vector<pipeline_cache_digest> cache_digests;
        mutable std::vector<pipeline_cache_artifact> cache_artifacts;
        mutable std::mutex cache_lock;
    };


	class CHAOS_API Pipeline
	{
	public:
		Pipeline(const VulkanDevice* vkdev);
		virtual ~Pipeline();

        void SetOptimalLocalSizeXYZ(int x = 4, int y = 4, int z = 4);
        void SetOptimalLocalSizeXYZ(const Shape& local_size_xyz);
        void SetLocalSizeXYZ(int x, int y, int z);

        void Create(const uint32_t* spv_data, size_t spv_data_size, const std::vector<VkSpecializationType>& specializations);

        void Create(int shader_type_index, const dnn::Option& opt, const std::vector<VkSpecializationType>& specializations);


        const VulkanDevice* vkdev;

        VkShaderModule shader_module;
        VkDescriptorSetLayout descriptorset_layout;
        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;
        VkDescriptorUpdateTemplateKHR descriptor_update_template;

        ShaderInfo shader_info;

        uint32_t local_size_x;
        uint32_t local_size_y;
        uint32_t local_size_z;
	};
}