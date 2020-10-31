#pragma once

#include "core/core.hpp"
#include "vk_tensor.hpp"

#include <vulkan/vulkan.h>

namespace chaos
{
    union VkSpecializationType
    {
        int i;
        float f;
        uint32_t u;
    };

    union VkConstantType
    {
        int i;
        float f;
    };

	// instance
	CHAOS_API void CreateGPUInstance();
	CHAOS_API void DestroyGPUInstance();

    // get info
    CHAOS_API int GetGPUCount();
    CHAOS_API int GetDefaultGPUIndex();

	class CHAOS_API GPUInfo
	{
	public:
        // vulkan physical device
        VkPhysicalDevice physical_device;

        // memory properties
        VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;

        // info
        uint32_t api_version;
        uint32_t driver_version;
        uint32_t vendor_id;
        uint32_t device_id;
        uint8_t pipeline_cache_uuid[VK_UUID_SIZE];

        // 0 = discrete gpu
        // 1 = integrated gpu
        // 2 = virtual gpu
        // 3 = cpu
        int type;

        // hardware limit
        uint32_t max_shared_memory_size;
        uint32_t max_workgroup_count[3];
        uint32_t max_workgroup_invocations;
        uint32_t max_workgroup_size[3];
        size_t memory_map_alignment;
        size_t buffer_offset_alignment;
        size_t non_coherent_atom_size;
        size_t buffer_image_granularity;
        uint32_t max_image_dimension_1d;
        uint32_t max_image_dimension_2d;
        uint32_t max_image_dimension_3d;
        float timestamp_period;

        // runtime
        uint32_t compute_queue_family_index;
        uint32_t graphics_queue_family_index;
        uint32_t transfer_queue_family_index;

        uint32_t compute_queue_count;
        uint32_t graphics_queue_count;
        uint32_t transfer_queue_count;

        // property
        bool unified_compute_transfer_queue;

        // bug is not feature
        bool bug_storage_buffer_no_l1;
        bool bug_layout_binding_id_alias;
        bool bug_corrupted_online_pipeline_cache;

        // but sometimes bug is a feature
        bool bug_implicit_fp16_arithmetic;

        // fp16 and int8 feature
        bool support_fp16_packed;
        bool support_fp16_storage;
        bool support_fp16_arithmetic;
        bool support_int8_storage;
        bool support_int8_arithmetic;

        // ycbcr conversion feature
        bool support_ycbcr_conversion;

        // extension capability
        int support_VK_KHR_8bit_storage;
        int support_VK_KHR_16bit_storage;
        int support_VK_KHR_bind_memory2;
        int support_VK_KHR_dedicated_allocation;
        int support_VK_KHR_descriptor_update_template;
        int support_VK_KHR_external_memory;
        int support_VK_KHR_get_memory_requirements2;
        int support_VK_KHR_maintenance1;
        int support_VK_KHR_push_descriptor;
        int support_VK_KHR_sampler_ycbcr_conversion;
        int support_VK_KHR_shader_float16_int8;
        int support_VK_KHR_shader_float_controls;
        int support_VK_KHR_storage_buffer_storage_class;
        int support_VK_KHR_swapchain;
        int support_VK_EXT_memory_budget;
        int support_VK_EXT_queue_family_foreign;
	};

    CHAOS_API const GPUInfo& GetGPUInfo(int device_index = GetDefaultGPUIndex());

    class VkAllocator;
    class VkTensor;
    class Pipeline;
    class PipelineCache;
    class CHAOS_API VulkanDevice
    {
    public:
        VulkanDevice(int device_index = GetDefaultGPUIndex());

        ~VulkanDevice();

        const GPUInfo& info;

        VkDevice GetDevice() const { return device; }

        VkShaderModule CreateShaderModule(int shader_type_index, uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z) const;
        VkShaderModule CompileShaderModule(const uint32_t* spv_data, size_t spv_data_size) const;

        // with fixed workgroup size
        VkShaderModule CompileShaderModule(const uint32_t* spv_data, size_t spv_data_size, uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z) const;

        // helper for creating pipeline
        void CreateDescriptorsetLayout(int binding_count, const int* binding_types, VkDescriptorSetLayout* descriptor_set_layout) const;
        void CreatePipelineLayout(int push_constant_count, VkDescriptorSetLayout descriptor_set_layout, VkPipelineLayout* pipeline_layout) const;
        void CreatePipeline(VkShaderModule shader_module, VkPipelineLayout pipeline_layout, const std::vector<VkSpecializationType>& specializations, VkPipeline* pipeline) const;
        void CreateDescriptorUpdateTemplate(int binding_count, const int* binding_types, VkDescriptorSetLayout descriptor_set_layout, VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplateKHR* descriptor_update_template) const;

        uint32_t FindMemoryIndex(uint32_t memory_type_bits, VkFlags required, VkFlags preferred, VkFlags preferred_not) const;
        bool IsMappable(uint32_t memory_type_index) const;
        bool IsCoherent(uint32_t memory_type_index) const;

        VkQueue AcquireQueue(uint32_t queue_family_index) const;
        void ReclaimQueue(uint32_t queue_family_index, VkQueue queue) const;

        const VkSampler* immutable_texelfetch_sampler() const { return &texelfetch_sampler; }
        VkTensor GetDummyBuffer() const;
        const PipelineCache* GetPipelineCache() const;

        // VK_KHR_bind_memory2
        PFN_vkBindBufferMemory2KHR vkBindBufferMemory2KHR;
        //PFN_vkBindImageMemory2KHR vkBindImageMemory2KHR;

        // VK_KHR_descriptor_update_template
        PFN_vkCreateDescriptorUpdateTemplateKHR vkCreateDescriptorUpdateTemplateKHR;
        PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR;
        PFN_vkUpdateDescriptorSetWithTemplateKHR vkUpdateDescriptorSetWithTemplateKHR;

        // VK_KHR_get_memory_requirements2
        PFN_vkGetBufferMemoryRequirements2KHR vkGetBufferMemoryRequirements2KHR;
        //PFN_vkGetImageMemoryRequirements2KHR vkGetImageMemoryRequirements2KHR;
        //PFN_vkGetImageSparseMemoryRequirements2KHR vkGetImageSparseMemoryRequirements2KHR;

        // VK_KHR_maintenance1
        PFN_vkTrimCommandPoolKHR vkTrimCommandPoolKHR;

        // VK_KHR_push_descriptor
        PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR;
        PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR;

        // VK_KHR_sampler_ycbcr_conversion
        PFN_vkCreateSamplerYcbcrConversionKHR vkCreateSamplerYcbcrConversionKHR;
        PFN_vkDestroySamplerYcbcrConversionKHR vkDestroySamplerYcbcrConversionKHR;

        // VK_KHR_swapchain
        PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR;
        PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR;
        PFN_vkQueuePresentKHR vkQueuePresentKHR;
        //PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR;
        //PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR;

    protected:
        // device extension
        void InitDeviceExtension();

    private:
        VkDevice device;

        // hardware queue
        mutable std::vector<VkQueue> compute_queues;
        mutable std::vector<VkQueue> graphics_queues;
        mutable std::vector<VkQueue> transfer_queues;

        mutable std::mutex queue_lock;

        // default blob allocator for each queue
        mutable std::vector<VkAllocator*> blob_allocators;
        mutable std::mutex blob_allocator_lock;

        // default staging allocator for each queue
        mutable std::vector<VkAllocator*> staging_allocators;
        mutable std::mutex staging_allocator_lock;

        // nearest sampler for texelfetch
        VkSampler texelfetch_sampler;

        //VkAllocator* dummy_allocator;
        VkTensor dummy_buffer;

        // device-wide pipeline cache
        PipelineCache* pipeline_cache;
    };

    CHAOS_API VulkanDevice* GetGPUDevice(int device_index = GetDefaultGPUIndex());

    // info from spirv
    class ShaderInfo
    {
    public:
        int specialization_count;
        int binding_count;
        int push_constant_count;

        // 0 = null
        // 1 = storage buffer
        // 2 = storage image
        // 3 = combined image sampler
        int binding_types[16]; // 16 is large enough I think ...
    };

    const ShaderInfo& GetShaderInfo(int shader_type_index);
    void ResolveShaderInfo(const uint32_t* spv_data, size_t spv_data_size, ShaderInfo& shader_info);
}