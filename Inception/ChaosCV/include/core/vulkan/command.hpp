#pragma once

#include "core/core.hpp"

#include "pipeline.hpp"

#include <vulkan/vulkan.hpp>

#include <vector>

namespace chaos
{
	namespace dnn { class Option; }
    class Pipeline;
	class CHAOS_API VkCompute
	{
	public:
		VkCompute(const VulkanDevice* vkdev);
		virtual ~VkCompute();


        void RecordUpload(const Tensor& src, VkTensor& dst, const dnn::Option& opt);
        void RecordDownload(const VkTensor& src, Tensor& dst, const dnn::Option& opt);

        void RecordPipeline(const Pipeline* pipeline, const std::vector<VkTensor>& buffer_bindings, const std::vector<VkConstantType>& constants, const Shape& dispatcher);

        void SubmitAndWait();

        void Reset();

    protected:
        void Init();
        void BeginCommandBuffer();
        void EndCommandBuffer();

        const VulkanDevice* vkdev;

        VkCommandPool compute_command_pool;

        VkCommandBuffer compute_command_buffer;

        VkFence compute_command_fence;

        std::vector<VkTensor> upload_staging_buffers;
        std::vector<VkTensor> download_post_buffers;
        std::vector<Tensor> download_post_mats_fp16;
        std::vector<Tensor> download_post_mats;

        // the good-old path for device without VK_KHR_push_descriptor
        std::vector<VkDescriptorPool> descriptor_pools;
        std::vector<VkDescriptorSet> descriptor_sets;

        struct Record
        {
            enum
            {
                TYPE_BIND_DESCRIPTOR_SETS,
                TYPE_BIND_PIPELINE,
                TYPE_BUFFER_BARRIERS,
                TYPE_COPY_BUFFER,
                TYPE_DISPATCH,
                TYPE_POST_DOWNLOAD,
                TYPE_PUSH_CONSTANTS,
            };

            int type;
            VkCommandBuffer command_buffer;

            union
            {
                struct
                {
                    VkPipelineBindPoint bind_point;
                    VkPipelineLayout pipeline_layout;
                    uint32_t descriptor_set_count;
                    uint32_t descriptor_set_offset;
                } bind_descriptor_sets;
                struct
                {
                    VkPipelineBindPoint bind_point;
                    VkPipeline pipeline;
                } bind_pipeline;
                struct
                {
                    VkPipelineStageFlags src_stage;
                    VkPipelineStageFlags dst_stage;
                    uint32_t barrier_count;
                    const VkBufferMemoryBarrier* barriers;
                } buffer_barriers;
                struct
                {
                    VkBuffer src;
                    VkBuffer dst;
                    uint32_t region_count;
                    const VkBufferCopy* regions;
                } copy_buffer;
                struct
                {
                    uint32_t group_count_x;
                    uint32_t group_count_y;
                    uint32_t group_count_z;
                } dispatch;
                struct
                {
                    uint32_t download_post_buffer_mat_offset;
                    uint32_t download_post_mat_fp16_offset;
                } post_download;
                struct
                {
                    VkPipelineLayout pipeline_layout;
                    VkShaderStageFlags stage_flags;
                    uint32_t size;
                    const void* values;
                } push_constants;
            };
        };

        std::vector<Record> delayed_records;
	};

	class CHAOS_API VkTransfer
	{
	public:
        VkTransfer(const VulkanDevice* vkdev);
        ~VkTransfer();

		void RecordUpload(const Tensor& src, VkTensor& dst, const dnn::Option& opt);

        void SubmitAndWait();

    protected:
        void Init();
        void BeginCommandBuffer();
        void EndCommandBuffer();

	protected:
		const VulkanDevice* vkdev;

		VkCommandPool compute_command_pool;
		VkCommandPool transfer_command_pool;

		VkCommandBuffer upload_command_buffer;
		VkCommandBuffer compute_command_buffer;

		VkSemaphore upload_compute_semaphore;

		VkFence upload_command_fence;
		VkFence compute_command_fence;

		std::vector<VkTensor> upload_staging_buffers;
	};
}