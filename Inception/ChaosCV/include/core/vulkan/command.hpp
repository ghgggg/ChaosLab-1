#pragma once

#include "core/core.hpp"

#include "pipeline.hpp"

#include <vulkan/vulkan.hpp>

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

        void RecordPipeline(const Pipeline* pipeline, const std::vector<VkTensor>& buffer_bindings, const std::vector<VkConstantType>& constants, const VkTensor& dispatcher);
        void RecordPipeline(const Pipeline* pipeline, const std::vector<VkTensor>& buffer_bindings, const std::vector<VkConstantType>& constants, const Tensor& dispatcher);

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
                TYPE_COPY_BUFFER,
                TYPE_BUFFER_BARRIERS,
                TYPE_POST_DOWNLOAD,
            };

            int type;
            VkCommandBuffer command_buffer;

            union
            {
                struct
                {
                    VkBuffer src;
                    VkBuffer dst;
                    uint32_t region_count;
                    const VkBufferCopy* regions;
                } copy_buffer;
                struct
                {
                    VkPipelineStageFlags src_stage;
                    VkPipelineStageFlags dst_stage;
                    uint32_t barrier_count;
                    const VkBufferMemoryBarrier* barriers;
                } buffer_barriers;
                struct
                {
                    uint32_t download_post_buffer_mat_offset;
                    uint32_t download_post_mat_fp16_offset;
                } post_download;
            };
        };

        std::vector<Record> delayed_records;
	};

	class CHAOS_API VkTransfer
	{
	public:
        VkTransfer(const VulkanDevice* vkdev);
        ~VkTransfer();

		void RecordUpload(const Tensor& src, VkTensor& dst, const dnn::Option& opt, bool flatten = true);

        int SubmitAndWait();
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