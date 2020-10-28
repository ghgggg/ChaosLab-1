#pragma once

#include "core/core.hpp"

#include <vulkan/vulkan.h>

namespace chaos
{
	// instance
	CHAOS_API void CreateGPUInstance();
	CHAOS_API void DestroyGPUInstance();

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
}