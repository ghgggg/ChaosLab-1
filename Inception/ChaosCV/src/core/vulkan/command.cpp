#include "core/vulkan/vk_allocator.hpp"
#include "core/vulkan/command.hpp"
#include "dnn/option.hpp"

namespace chaos
{
	VkCompute::VkCompute(const VulkanDevice* vkdev) : vkdev(vkdev)
	{
        compute_command_pool = 0;
        compute_command_buffer = 0;
        compute_command_fence = 0;

        Init();
	}
	VkCompute::~VkCompute()
	{
        if (!vkdev->info.support_VK_KHR_push_descriptor)
        {
            for (size_t i = 0; i < descriptor_sets.size(); i++)
            {
                vkFreeDescriptorSets(vkdev->GetDevice(), descriptor_pools[i], 1, &descriptor_sets[i]);
                vkDestroyDescriptorPool(vkdev->GetDevice(), descriptor_pools[i], 0);
            }
        }

        vkDestroyFence(vkdev->GetDevice(), compute_command_fence, 0);

        vkFreeCommandBuffers(vkdev->GetDevice(), compute_command_pool, 1, &compute_command_buffer);
        vkDestroyCommandPool(vkdev->GetDevice(), compute_command_pool, 0);
	}


    void VkCompute::RecordUpload(const Tensor& src, VkTensor& dst, const dnn::Option& opt)
    {
        const Tensor& _src = src;

        // upload
        VkTensor& dst_staging = dst;
        if (opt.blob_vkallocator->mappable)
        {
            dst_staging.CreateLike(_src, opt.blob_vkallocator);
        }
        else
        {
            dst_staging.CreateLike(_src, opt.staging_vkallocator);
        }
        if (dst_staging.empty()) return;

        // stash staging
        upload_staging_buffers.push_back(dst_staging);

        // memcpy src to device
        memcpy(dst_staging.mapped_data(), src.data, src.shape[0] * src.steps[0] * src.depth * src.packing);
        dst_staging.allocator->Flush(dst_staging.data);

        // mark device host-write @ null
        dst_staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        dst_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

        //// resolve dst_elempack
        //int dims = src_fp16.dims;
        //int elemcount = 0;
        //if (dims == 1) elemcount = src_fp16.elempack * src_fp16.w;
        //if (dims == 2) elemcount = src_fp16.elempack * src_fp16.h;
        //if (dims == 3) elemcount = src_fp16.elempack * src_fp16.c;

        //int dst_elempack = 1;
        //if (opt.use_shader_pack8)
        //    dst_elempack = elemcount % 8 == 0 ? 8 : elemcount % 4 == 0 ? 4 : 1;
        //else
        //    dst_elempack = elemcount % 4 == 0 ? 4 : 1;

        // gpu cast to fp16 on the fly (integrated gpu)
        // vkdev->ConvertPacking(dst_staging, dst, dst_elempack, *this, opt);
        // 从底层代码看，commadn需要pipeline
        // 所以还需要RecordPipeline
    }

    void VkCompute::RecordDownload(const VkTensor& src, Tensor& dst, const dnn::Option& opt)
    {
        //// resolve dst_elempack
        //int dims = src.dims;
        //int elemcount = 0;
        //if (dims == 1) elemcount = src.elempack * src.w;
        //if (dims == 2) elemcount = src.elempack * src.h;
        //if (dims == 3) elemcount = src.elempack * src.c;

        //int dst_elempack = 1;
        //if (opt.use_packing_layout)
        //    dst_elempack = elemcount % 4 == 0 ? 4 : 1;
        //else
        //    dst_elempack = 1;

        // gpu cast to fp32 on the fly (integrated gpu)
        dnn::Option opt_staging = opt;
        if (vkdev->info.type != 0)
        {
            opt_staging.use_fp16_packed = false;
            opt_staging.use_fp16_storage = false;
        }

        const VkTensor& dst_staging = src;
        //if (opt_staging.blob_vkallocator->mappable)
        //{
        //    vkdev->convert_packing(src, dst_staging, dst_elempack, *this, opt_staging);
        //}
        //else
        //{
        //    opt_staging.blob_vkallocator = opt.staging_vkallocator;
        //    vkdev->convert_packing(src, dst_staging, dst_elempack, *this, opt_staging);
        //}

        // barrier device any @ compute to host-read @ compute
        if (dst_staging.data->access_flags & VK_ACCESS_HOST_WRITE_BIT || dst_staging.data->stage_flags != VK_PIPELINE_STAGE_HOST_BIT)
        {
            VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
            barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[0].pNext = 0;
            barriers[0].srcAccessMask = dst_staging.data->access_flags;
            barriers[0].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].buffer = dst_staging.buffer();
            barriers[0].offset = dst_staging.buffer_offset();
            barriers[0].size = dst_staging.buffer_capacity();

            VkPipelineStageFlags src_stage = dst_staging.data->stage_flags;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_HOST_BIT;

            if (vkdev->info.support_VK_KHR_push_descriptor)
            {
                vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
                delete[] barriers;
            }
            else
            {
                Record r;
                r.type = Record::TYPE_BUFFER_BARRIERS;
                r.command_buffer = compute_command_buffer;
                r.buffer_barriers.src_stage = src_stage;
                r.buffer_barriers.dst_stage = dst_stage;
                r.buffer_barriers.barrier_count = 1;
                r.buffer_barriers.barriers = barriers;
                delayed_records.push_back(r);
            }

            // mark device host-read @ any
            dst_staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
            dst_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
        }

        // create dst
        Tensor _dst;
        _dst.CreateLike(dst_staging, opt.blob_allocator);
        //if (dst_fp16.empty())
        //    return;

        // download
        download_post_buffers.push_back(dst_staging);
        download_post_mats_fp16.push_back(_dst);

        // post memcpy device to dst
        {
            Record r;
            r.type = Record::TYPE_POST_DOWNLOAD;
            r.command_buffer = 0;
            r.post_download.download_post_buffer_mat_offset = download_post_buffers.size() - 1;
            r.post_download.download_post_mat_fp16_offset = download_post_mats_fp16.size() - 1;
            delayed_records.push_back(r);
        }

        dst = _dst;
    }

    void VkCompute::SubmitAndWait()
    {
        if (!vkdev->info.support_VK_KHR_push_descriptor)
        {
            BeginCommandBuffer();

            const size_t record_count = delayed_records.size();

            // handle delayed records
            for (size_t i = 0; i < record_count; i++)
            {
                const Record& r = delayed_records[i];

                switch (r.type)
                {
                case Record::TYPE_COPY_BUFFER:
                {
                    vkCmdCopyBuffer(r.command_buffer, r.copy_buffer.src, r.copy_buffer.dst, r.copy_buffer.region_count, r.copy_buffer.regions);
                    delete[] r.copy_buffer.regions;
                    break;
                }
                case Record::TYPE_BUFFER_BARRIERS: // buffer_barrers:
                {
                    vkCmdPipelineBarrier(r.command_buffer, r.buffer_barriers.src_stage, r.buffer_barriers.dst_stage, 0, 0, 0, r.buffer_barriers.barrier_count, r.buffer_barriers.barriers, 0, 0);
                    delete[] r.buffer_barriers.barriers;
                    break;
                }
                case Record::TYPE_POST_DOWNLOAD: //post_download:
                default:
                    break;
                }
            }
        }

        // end command buffer
        {
            EndCommandBuffer();
        }

        // acquire queue and reclaim on return
        VkQueue compute_queue = vkdev->AcquireQueue(vkdev->info.compute_queue_family_index);
        CHECK_NE(compute_queue, 0) << "out of compute queue";

        // submit compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkQueueSubmit failed %d", ret);
            //if (ret != VK_SUCCESS)
            //{
            //    NCNN_LOGE("vkQueueSubmit failed %d", ret);
            //    vkdev->ReclaimQueue(vkdev->info.compute_queue_family_index, compute_queue);
            //    return -1;
            //}
        }

        vkdev->ReclaimQueue(vkdev->info.compute_queue_family_index, compute_queue);

        // wait
        {
            VkResult ret = vkWaitForFences(vkdev->GetDevice(), 1, &compute_command_fence, VK_TRUE, UINT64_MAX);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkWaitForFences failed %d", ret);
        }

        // handle delayed post records
        for (size_t i = 0; i < delayed_records.size(); i++)
        {
            const Record& r = delayed_records[i];

            switch (r.type)
            {
            case Record::TYPE_POST_DOWNLOAD: //post_download:
            {
                const VkTensor& src = download_post_buffers[r.post_download.download_post_buffer_mat_offset];
                Tensor& dst = download_post_mats_fp16[r.post_download.download_post_mat_fp16_offset];

                // NCNN_LOGE("post_download  %p +%d ~%d  -> %p", src.buffer(), src.buffer_offset(), src.buffer_capacity(), dst.data);

                src.allocator->Invalidate(src.data);
                memcpy(dst.data, src.mapped_data(), dst.shape[0] * dst.steps[0] * dst.depth * dst.packing);
                break;
            }
            default:
                break;
            }
        }

        delayed_records.clear();
    }

    void VkCompute::Reset()
    {
        upload_staging_buffers.clear();
        download_post_buffers.clear();
        download_post_mats_fp16.clear();
        download_post_mats.clear();

        if (!vkdev->info.support_VK_KHR_push_descriptor)
        {
            for (size_t i = 0; i < descriptor_sets.size(); i++)
            {
                vkFreeDescriptorSets(vkdev->GetDevice(), descriptor_pools[i], 1, &descriptor_sets[i]);
                vkDestroyDescriptorPool(vkdev->GetDevice(), descriptor_pools[i], 0);
            }
            descriptor_pools.clear();
            descriptor_sets.clear();
        }

        delayed_records.clear();

        // reset command buffer and fence
        {
            VkResult ret = vkResetCommandBuffer(compute_command_buffer, 0);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkResetCommandBuffer failed %d", ret);
        }
        {
            VkResult ret = vkResetFences(vkdev->GetDevice(), 1, &compute_command_fence);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkResetFences failed %d", ret);
        }

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            BeginCommandBuffer();
        }
    }


    void VkCompute::Init()
    {
        // compute_command_pool
        {
            VkCommandPoolCreateInfo commandPoolCreateInfo;
            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolCreateInfo.pNext = 0;
            commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index;

            VkResult ret = vkCreateCommandPool(vkdev->GetDevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkCreateCommandPool failed %d", ret);
        }

        // compute_command_buffer
        {
            VkCommandBufferAllocateInfo commandBufferAllocateInfo;
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.pNext = 0;
            commandBufferAllocateInfo.commandPool = compute_command_pool;
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.commandBufferCount = 1;

            VkResult ret = vkAllocateCommandBuffers(vkdev->GetDevice(), &commandBufferAllocateInfo, &compute_command_buffer);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkAllocateCommandBuffers failed %d", ret);
        }

        // compute_command_fence
        {
            VkFenceCreateInfo fenceCreateInfo;
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.pNext = 0;
            fenceCreateInfo.flags = 0;

            VkResult ret = vkCreateFence(vkdev->GetDevice(), &fenceCreateInfo, 0, &compute_command_fence);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkCreateFence failed %d", ret);
        }

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            BeginCommandBuffer();
        }
    }


    void VkCompute::BeginCommandBuffer()
    {
        VkCommandBufferBeginInfo commandBufferBeginInfo;
        commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBufferBeginInfo.pNext = 0;
        commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        commandBufferBeginInfo.pInheritanceInfo = 0;

        VkResult ret = vkBeginCommandBuffer(compute_command_buffer, &commandBufferBeginInfo);
        CHECK_EQ(ret, VK_SUCCESS) << Format("vkBeginCommandBuffer failed %d", ret);
    }

    void VkCompute::EndCommandBuffer()
    {
        VkResult ret = vkEndCommandBuffer(compute_command_buffer);
        CHECK_EQ(ret, VK_SUCCESS) << Format("vkEndCommandBuffer failed %d", ret);
    }
}