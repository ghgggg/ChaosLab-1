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
        if (opt_staging.blob_vkallocator->mappable)
        {
            //std::cout << "mappable" << std::endl;
        }
        else
        {
            opt_staging.blob_vkallocator = opt.staging_vkallocator;

        }
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
        Tensor& _dst = dst;
        _dst.CreateLike(dst_staging, opt.blob_allocator);
        //if (dst_fp16.empty())
        //    return;

        // download
        download_post_buffers.push_back(dst_staging);
        download_post_mats.push_back(_dst);
        //download_post_mats_fp16.push_back(_dst);

        // post memcpy device to dst
        {
            Record r;
            r.type = Record::TYPE_POST_DOWNLOAD;
            r.command_buffer = 0;
            r.post_download.download_post_buffer_mat_offset = (uint32_t)download_post_buffers.size() - 1;
            //r.post_download.download_post_mat_fp16_offset = (uint32_t)download_post_mats_fp16.size() - 1;
            delayed_records.push_back(r);
        }
    }

    void VkCompute::RecordPipeline(const Pipeline* pipeline, const std::vector<VkTensor>& buffer_bindings, const std::vector<VkConstantType>& constants, const Shape& dispatcher)
    {
        const int buffer_binding_count = (int)buffer_bindings.size();
        //const int image_binding_count = (int)image_bindings.size();
        const int constant_count = (int)constants.size();

        const int binding_count = buffer_binding_count; // + image_binding_count;

        if (binding_count != pipeline->shader_info.binding_count)
        {
            LOG(ERROR) << Format("binding_count not match, expect %d but got %d", pipeline->shader_info.binding_count, buffer_binding_count);
            //NCNN_LOGE("binding_count not match, expect %d but got %d + %d", pipeline->shader_info.binding_count, buffer_binding_count, image_binding_count);
        }

        if (constant_count != pipeline->shader_info.push_constant_count)
        {
            LOG(ERROR) << Format("push_constant_count not match, expect %d but got %d", pipeline->shader_info.push_constant_count, constant_count);
        }

        int buffer_index = 0;
        int image_index = 0;
        for (int i = 0; i < binding_count; i++)
        {
            int binding_type = pipeline->shader_info.binding_types[i];

            if (binding_type == 1)
            {
                const VkTensor& binding = buffer_bindings[buffer_index]; // .empty() ? vkdev->GetDummyBuffer() : buffer_bindings[buffer_index];
                buffer_index++;

                // NCNN_LOGE("binding #%d buffer = %d %d %d %d @ %lu %d = %p +%ld ~%ld", i, binding.dims, binding.w, binding.h, binding.c, binding.elemsize, binding.elempack, binding.buffer(), binding.buffer_offset(), binding.buffer_capacity());

                if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
                {
                    // barrier device any @ compute/null to shader-readwrite @ compute
                    VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
                    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    barriers[0].pNext = 0;
                    barriers[0].srcAccessMask = binding.data->access_flags;
                    barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barriers[0].buffer = binding.buffer();
                    barriers[0].offset = binding.buffer_offset();
                    barriers[0].size = binding.buffer_capacity();

                    VkPipelineStageFlags src_stage = binding.data->stage_flags;
                    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

                    if (vkdev->info.support_VK_KHR_push_descriptor)
                    {
                        vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
                        delete[] barriers;
                    }
                    else
                    {
                        Record r;
                        r.type = Record::TYPE_BUFFER_BARRIERS; // buffer_barrers;
                        r.command_buffer = compute_command_buffer;
                        r.buffer_barriers.src_stage = src_stage;
                        r.buffer_barriers.dst_stage = dst_stage;
                        r.buffer_barriers.barrier_count = 1;
                        r.buffer_barriers.barriers = barriers;
                        delayed_records.push_back(r);
                    }

                    // mark device shader-readwrite @ compute
                    binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                }
            }
            else if (binding_type == 2)
            {
                LOG(FATAL) << "not now";
            }
            else // if (binding_type == 3)
            {
                LOG(FATAL) << "not now";
            }
        }

        // record bind pipeline
        {
            if (vkdev->info.support_VK_KHR_push_descriptor)
            {
                vkCmdBindPipeline(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
            }
            else
            {
                Record r;
                r.type = Record::TYPE_BIND_PIPELINE; //_bind_pipeline;
                r.command_buffer = compute_command_buffer;
                r.bind_pipeline.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
                r.bind_pipeline.pipeline = pipeline->pipeline;
                delayed_records.push_back(r);
            }
        }

        // record update bindings
        if (binding_count > 0)
        {
            std::vector<unsigned char> descriptorInfos;
            {
                descriptorInfos.resize(sizeof(VkDescriptorBufferInfo) * buffer_binding_count/* + sizeof(VkDescriptorImageInfo) * image_binding_count*/);

                unsigned char* p_descriptorInfos = descriptorInfos.data();
                int descriptorBufferInfo_index = 0;
                int descriptorImageInfo_index = 0;
                for (int i = 0; i < binding_count; i++)
                {
                    int binding_type = pipeline->shader_info.binding_types[i];

                    if (binding_type == 1)
                    {
                        const VkTensor& binding = buffer_bindings[descriptorBufferInfo_index].empty() ? vkdev->GetDummyBuffer() : buffer_bindings[descriptorBufferInfo_index];
                        descriptorBufferInfo_index++;

                        VkDescriptorBufferInfo descriptorBufferInfo;
                        descriptorBufferInfo.buffer = binding.buffer();
                        descriptorBufferInfo.offset = binding.buffer_offset();
                        descriptorBufferInfo.range = binding.shape[0] * binding.steps[0] * binding.depth * binding.packing; // binding.total() * binding.elemsize;

                        memcpy(p_descriptorInfos, &descriptorBufferInfo, sizeof(VkDescriptorBufferInfo));
                        p_descriptorInfos += sizeof(VkDescriptorBufferInfo);
                    }
                    else //if (binding_type == 2 || binding_type == 3)
                    {
                        LOG(FATAL) << "not now";
                        //const VkImageMat& binding = image_bindings[descriptorImageInfo_index].empty() ? vkdev->get_dummy_image() : image_bindings[descriptorImageInfo_index];
                        //descriptorImageInfo_index++;

                        //// we always use immutable nearest sampler set in descroptor layout during pipeline creation
                        //VkDescriptorImageInfo descriptorImageInfo;
                        //descriptorImageInfo.sampler = 0;
                        //descriptorImageInfo.imageView = binding.imageview();
                        //descriptorImageInfo.imageLayout = binding.data->image_layout;

                        //memcpy(p_descriptorInfos, &descriptorImageInfo, sizeof(VkDescriptorImageInfo));
                        //p_descriptorInfos += sizeof(VkDescriptorImageInfo);
                    }
                }
            }

            if (vkdev->info.support_VK_KHR_push_descriptor)
            {
                vkdev->vkCmdPushDescriptorSetWithTemplateKHR(compute_command_buffer, pipeline->descriptor_update_template, pipeline->pipeline_layout, 0, descriptorInfos.data());
            }
            else
            {
                // create new descriptor_pool and descriptorset
                VkDescriptorPool descriptor_pool;
                {
                    int image_binding_count = 0;
                    int sampler_binding_count = 0;
                    for (int i = 0; i < binding_count; i++)
                    {
                        int binding_type = pipeline->shader_info.binding_types[i];

                        if (binding_type == 2)
                            image_binding_count++;
                        else // if (binding_type == 3)
                            sampler_binding_count++;
                    }

                    VkDescriptorPoolSize poolSizes[3];
                    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    poolSizes[0].descriptorCount = buffer_binding_count;
                    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    poolSizes[1].descriptorCount = image_binding_count;
                    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    poolSizes[2].descriptorCount = sampler_binding_count;

                    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
                    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                    descriptorPoolCreateInfo.pNext = 0;
                    descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                    descriptorPoolCreateInfo.maxSets = 1;
                    descriptorPoolCreateInfo.poolSizeCount = 3;
                    descriptorPoolCreateInfo.pPoolSizes = poolSizes;

                    VkResult ret = vkCreateDescriptorPool(vkdev->GetDevice(), &descriptorPoolCreateInfo, 0, &descriptor_pool);
                    CHECK_EQ(ret, VK_SUCCESS) << Format("vkCreateDescriptorPool failed %d", ret);
                }
                descriptor_pools.push_back(descriptor_pool);

                VkDescriptorSet descriptorset;
                {
                    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
                    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    descriptorSetAllocateInfo.pNext = 0;
                    descriptorSetAllocateInfo.descriptorPool = descriptor_pool;
                    descriptorSetAllocateInfo.descriptorSetCount = 1;
                    descriptorSetAllocateInfo.pSetLayouts = &pipeline->descriptorset_layout;

                    VkResult ret = vkAllocateDescriptorSets(vkdev->GetDevice(), &descriptorSetAllocateInfo, &descriptorset);
                    CHECK_EQ(ret, VK_SUCCESS) << Format("vkAllocateDescriptorSets failed %d", ret);
                }
                descriptor_sets.push_back(descriptorset);

                if (vkdev->info.support_VK_KHR_descriptor_update_template)
                {
                    vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->GetDevice(), descriptorset, pipeline->descriptor_update_template, descriptorInfos.data());
                }
                else
                {
                    std::vector<VkWriteDescriptorSet> writeDescriptorSets(binding_count);
                    {
                        const unsigned char* p_descriptorInfos = descriptorInfos.data();
                        for (int i = 0; i < binding_count; i++)
                        {
                            int binding_type = pipeline->shader_info.binding_types[i];

                            writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                            writeDescriptorSets[i].pNext = 0;
                            writeDescriptorSets[i].dstSet = descriptorset;
                            writeDescriptorSets[i].dstBinding = i;
                            writeDescriptorSets[i].dstArrayElement = 0;
                            writeDescriptorSets[i].descriptorCount = 1;
                            writeDescriptorSets[i].pTexelBufferView = 0;

                            if (binding_type == 1)
                            {
                                writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                                writeDescriptorSets[i].pImageInfo = 0;
                                writeDescriptorSets[i].pBufferInfo = (const VkDescriptorBufferInfo*)p_descriptorInfos;

                                p_descriptorInfos += sizeof(VkDescriptorBufferInfo);
                            }
                            else if (binding_type == 2)
                            {
                                writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                                writeDescriptorSets[i].pImageInfo = (const VkDescriptorImageInfo*)p_descriptorInfos;
                                writeDescriptorSets[i].pBufferInfo = 0;

                                p_descriptorInfos += sizeof(VkDescriptorImageInfo);
                            }
                            else // if (binding_type == 3)
                            {
                                writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                                writeDescriptorSets[i].pImageInfo = (const VkDescriptorImageInfo*)p_descriptorInfos;
                                writeDescriptorSets[i].pBufferInfo = 0;

                                p_descriptorInfos += sizeof(VkDescriptorImageInfo);
                            }
                        }
                    }

                    vkUpdateDescriptorSets(vkdev->GetDevice(), binding_count, writeDescriptorSets.data(), 0, 0);
                }

                Record r;
                r.type = Record::TYPE_BIND_DESCRIPTOR_SETS; //_bind_descriptorsets;
                r.command_buffer = compute_command_buffer;
                r.bind_descriptor_sets.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
                r.bind_descriptor_sets.pipeline_layout = pipeline->pipeline_layout;
                r.bind_descriptor_sets.descriptor_set_count = 1;
                r.bind_descriptor_sets.descriptor_set_offset = (uint32_t)descriptor_sets.size() - 1;
                delayed_records.push_back(r);
            }
        }

        // record push constants
        if (constant_count > 0)
        {
            if (vkdev->info.support_VK_KHR_push_descriptor)
            {
                vkCmdPushConstants(compute_command_buffer, pipeline->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constant_count * sizeof(VkConstantType), constants.data());
            }
            else
            {
                uint32_t size = constant_count * sizeof(VkConstantType);
                unsigned char* constant_values = new unsigned char[size];
                memcpy(constant_values, constants.data(), size);

                Record r;
                r.type = Record::TYPE_PUSH_CONSTANTS; //_push_constants;
                r.command_buffer = compute_command_buffer;
                r.push_constants.pipeline_layout = pipeline->pipeline_layout;
                r.push_constants.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
                r.push_constants.size = size;
                r.push_constants.values = constant_values;
                delayed_records.push_back(r);
            }
        }

        // record dispatch
        {
            uint32_t group_count_x = (dispatcher.GetX() + pipeline->local_size_x - 1) / pipeline->local_size_x;
            uint32_t group_count_y = (dispatcher.GetY() + pipeline->local_size_y - 1) / pipeline->local_size_y;
            uint32_t group_count_z = (dispatcher.GetZ() + pipeline->local_size_z - 1) / pipeline->local_size_z;

            if (vkdev->info.support_VK_KHR_push_descriptor)
            {
                vkCmdDispatch(compute_command_buffer, group_count_x, group_count_y, group_count_z);
            }
            else
            {
                Record r;
                r.type = Record::TYPE_DISPATCH; // _dispatch;
                r.command_buffer = compute_command_buffer;
                r.dispatch.group_count_x = group_count_x;
                r.dispatch.group_count_y = group_count_y;
                r.dispatch.group_count_z = group_count_z;
                delayed_records.push_back(r);
            }
        }
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
                case Record::TYPE_BIND_DESCRIPTOR_SETS: //bind_descriptorsets:
                {
                    vkCmdBindDescriptorSets(r.command_buffer, r.bind_descriptor_sets.bind_point, r.bind_descriptor_sets.pipeline_layout, 0, r.bind_descriptor_sets.descriptor_set_count, &descriptor_sets[r.bind_descriptor_sets.descriptor_set_offset], 0, 0);
                    break;
                }
                case Record::TYPE_BIND_PIPELINE: //bind_pipeline:
                {
                    vkCmdBindPipeline(r.command_buffer, r.bind_pipeline.bind_point, r.bind_pipeline.pipeline);
                    break;
                }
                case Record::TYPE_BUFFER_BARRIERS: // buffer_barrers:
                {
                    vkCmdPipelineBarrier(r.command_buffer, r.buffer_barriers.src_stage, r.buffer_barriers.dst_stage, 0, 0, 0, r.buffer_barriers.barrier_count, r.buffer_barriers.barriers, 0, 0);
                    delete[] r.buffer_barriers.barriers;
                    break;
                }
                case Record::TYPE_COPY_BUFFER:
                {
                    vkCmdCopyBuffer(r.command_buffer, r.copy_buffer.src, r.copy_buffer.dst, r.copy_buffer.region_count, r.copy_buffer.regions);
                    delete[] r.copy_buffer.regions;
                    break;
                }
                case Record::TYPE_DISPATCH: //dispatch:
                {
                    vkCmdDispatch(r.command_buffer, r.dispatch.group_count_x, r.dispatch.group_count_y, r.dispatch.group_count_z);
                    break;
                }
                case Record::TYPE_PUSH_CONSTANTS: //push_constants:
                {
                    vkCmdPushConstants(r.command_buffer, r.push_constants.pipeline_layout, r.push_constants.stage_flags, 0, r.push_constants.size, r.push_constants.values);
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
                Tensor& dst = download_post_mats[r.post_download.download_post_buffer_mat_offset];
                //Tensor& dst = download_post_mats[r.post_download.download_post_buffer_mat_offset];
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
        //download_post_mats_fp16.clear();
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

    VkTransfer::VkTransfer(const VulkanDevice* vkdev) : vkdev(vkdev)
    {
        compute_command_pool = 0;
        transfer_command_pool = 0;

        upload_command_buffer = 0;
        compute_command_buffer = 0;

        upload_compute_semaphore = 0;

        upload_command_fence = 0;
        compute_command_fence = 0;

        Init();
    }
    VkTransfer::~VkTransfer()
    {
        vkDestroyFence(vkdev->GetDevice(), compute_command_fence, 0);

        vkFreeCommandBuffers(vkdev->GetDevice(), compute_command_pool, 1, &compute_command_buffer);
        vkDestroyCommandPool(vkdev->GetDevice(), compute_command_pool, 0);

        if (!vkdev->info.unified_compute_transfer_queue)
        {
            vkDestroyFence(vkdev->GetDevice(), upload_command_fence, 0);

            vkDestroySemaphore(vkdev->GetDevice(), upload_compute_semaphore, 0);

            vkFreeCommandBuffers(vkdev->GetDevice(), transfer_command_pool, 1, &upload_command_buffer);
            vkDestroyCommandPool(vkdev->GetDevice(), transfer_command_pool, 0);
        }
    }

    void VkTransfer::RecordUpload(const Tensor& src, VkTensor& dst, const dnn::Option& opt)
    {
        Tensor src_flattened = src;

        // create dst
        dst.CreateLike(src_flattened, opt.blob_vkallocator);

        if (dst.empty())
        {
            return;
        }

        if (dst.allocator->mappable)
        {
            // memcpy src_flattened to device
            memcpy(dst.mapped_data(), src_flattened.data, src_flattened.shape[0] * src_flattened.steps[0] * src_flattened.depth * src_flattened.packing);
            dst.allocator->Flush(dst.data);

            // barrier device host-write @ null to shader-read @ compute
            {
                VkBufferMemoryBarrier barrier;
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.pNext = 0;
                barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = dst.buffer();
                barrier.offset = dst.buffer_offset();
                barrier.size = dst.buffer_capacity();

                VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
                VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

                vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
            }

            // mark device shader-readwrite @ compute
            dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
            dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            return;
        }

        // create staging
        VkTensor dst_staging;
        dst_staging.CreateLike(src_flattened, opt.staging_vkallocator);

        // memcpy src_flattened to staging
        memcpy(dst_staging.mapped_data(), src_flattened.data, src_flattened.shape[0]* src_flattened.steps[0]* src_flattened.depth * src_flattened.packing); //src_flattened.total() * src_flattened.elemsize
        dst_staging.allocator->Flush(dst_staging.data);

        VkCommandBuffer command_buffer;
        if (vkdev->info.unified_compute_transfer_queue)
        {
            command_buffer = compute_command_buffer;
        }
        else
        {
            command_buffer = upload_command_buffer;
        }

        // barrier staging host-write @ null to transfer-read @ queue
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = dst_staging.buffer();
            barrier.offset = dst_staging.buffer_offset();
            barrier.size = dst_staging.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

            vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }

        // record staging to device
        {
            VkBufferCopy region;
            region.srcOffset = dst_staging.buffer_offset();
            region.dstOffset = dst.buffer_offset();
            region.size = std::min(dst_staging.buffer_capacity(), dst.buffer_capacity());

            vkCmdCopyBuffer(command_buffer, dst_staging.buffer(), dst.buffer(), 1, &region);
        }

        if (vkdev->info.unified_compute_transfer_queue)
        {
            // barrier device transfer-write @ compute to shader-read @ compute
            {
                VkBufferMemoryBarrier barrier;
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.pNext = 0;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = dst.buffer();
                barrier.offset = dst.buffer_offset();
                barrier.size = dst.buffer_capacity();

                VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

                vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
            }
        }
        else
        {
            // queue ownership transfer transfer-write @ transfer to shader-read @ compute

            // release
            {
                VkBufferMemoryBarrier barrier;
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.pNext = 0;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = 0;
                barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index;
                barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index;
                barrier.buffer = dst.buffer();
                barrier.offset = dst.buffer_offset();
                barrier.size = dst.buffer_capacity();

                VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

                vkCmdPipelineBarrier(upload_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
            }

            // acquire
            {
                VkBufferMemoryBarrier barrier;
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.pNext = 0;
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index;
                barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index;
                barrier.buffer = dst.buffer();
                barrier.offset = dst.buffer_offset();
                barrier.size = dst.buffer_capacity();

                VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

                vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
            }
        }

        // mark device shader-readwrite @ compute
        dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        // stash staging
        upload_staging_buffers.push_back(dst_staging);
    }

    void VkTransfer::SubmitAndWait()
    {
        // end command buffer
        {
            EndCommandBuffer();
        }

        VkQueue compute_queue = vkdev->AcquireQueue(vkdev->info.compute_queue_family_index);
        CHECK_NE(compute_queue, 0) << "out of compute queue";

        if (vkdev->info.unified_compute_transfer_queue)
        {
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
                //    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
                //    return -1;
                //}
            }
        }
        else
        {
            VkQueue transfer_queue = vkdev->AcquireQueue(vkdev->info.transfer_queue_family_index);
            CHECK_NE(compute_queue, 0) << "out of compute queue";
            //if (transfer_queue == 0)
            //{
            //    NCNN_LOGE("out of transfer queue");
            //    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
            //    return -1;
            //}

            // submit upload compute
            {
                VkSubmitInfo submitInfo;
                submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submitInfo.pNext = 0;
                submitInfo.waitSemaphoreCount = 0;
                submitInfo.pWaitSemaphores = 0;
                submitInfo.pWaitDstStageMask = 0;
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &upload_command_buffer;
                submitInfo.signalSemaphoreCount = 1;
                submitInfo.pSignalSemaphores = &upload_compute_semaphore;

                VkResult ret = vkQueueSubmit(transfer_queue, 1, &submitInfo, upload_command_fence);
                CHECK_EQ(ret, VK_SUCCESS) << Format("vkQueueSubmit failed %d", ret);
                //if (ret != VK_SUCCESS)
                //{
                //    NCNN_LOGE("vkQueueSubmit failed %d", ret);
                //    vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index, transfer_queue);
                //    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
                //    return -1;
                //}
            }
            {
                VkPipelineStageFlags wait_dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; // FIXME

                VkSubmitInfo submitInfo;
                submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submitInfo.pNext = 0;
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores = &upload_compute_semaphore;
                submitInfo.pWaitDstStageMask = &wait_dst_stage;
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &compute_command_buffer;
                submitInfo.signalSemaphoreCount = 0;
                submitInfo.pSignalSemaphores = 0;

                VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
                CHECK_EQ(ret, VK_SUCCESS) << Format("vkQueueSubmit failed %d", ret);
                //if (ret != VK_SUCCESS)
                //{
                //    NCNN_LOGE("vkQueueSubmit failed %d", ret);
                //    vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index, transfer_queue);
                //    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
                //    return -1;
                //}
            }

            vkdev->ReclaimQueue(vkdev->info.transfer_queue_family_index, transfer_queue);
        }

        vkdev->ReclaimQueue(vkdev->info.compute_queue_family_index, compute_queue);

        // wait
        if (vkdev->info.unified_compute_transfer_queue)
        {
            VkResult ret = vkWaitForFences(vkdev->GetDevice(), 1, &compute_command_fence, VK_TRUE, UINT64_MAX);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkWaitForFences failed %d", ret);
        }
        else
        {
            VkFence fences[2] = { upload_command_fence, compute_command_fence };

            VkResult ret = vkWaitForFences(vkdev->GetDevice(), 2, fences, VK_TRUE, UINT64_MAX);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkWaitForFences failed %d", ret);
        }
    }

    void VkTransfer::Init()
    {
        // compute_command_pool
        {
            VkCommandPoolCreateInfo commandPoolCreateInfo;
            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolCreateInfo.pNext = 0;
            commandPoolCreateInfo.flags = 0;
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

        if (!vkdev->info.unified_compute_transfer_queue)
        {
            // transfer_command_pool
            {
                VkCommandPoolCreateInfo commandPoolCreateInfo;
                commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                commandPoolCreateInfo.pNext = 0;
                commandPoolCreateInfo.flags = 0;
                commandPoolCreateInfo.queueFamilyIndex = vkdev->info.transfer_queue_family_index;

                VkResult ret = vkCreateCommandPool(vkdev->GetDevice(), &commandPoolCreateInfo, 0, &transfer_command_pool);
                CHECK_EQ(ret, VK_SUCCESS) << Format("vkCreateCommandPool failed %d", ret);
            }

            // upload_command_buffer
            {
                VkCommandBufferAllocateInfo commandBufferAllocateInfo;
                commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                commandBufferAllocateInfo.pNext = 0;
                commandBufferAllocateInfo.commandPool = transfer_command_pool;
                commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                commandBufferAllocateInfo.commandBufferCount = 1;

                VkResult ret = vkAllocateCommandBuffers(vkdev->GetDevice(), &commandBufferAllocateInfo, &upload_command_buffer);
                CHECK_EQ(ret, VK_SUCCESS) << Format("vkAllocateCommandBuffers failed %d", ret);
            }

            // upload_compute_semaphore
            {
                VkSemaphoreCreateInfo semaphoreCreateInfo;
                semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
                semaphoreCreateInfo.pNext = 0;
                semaphoreCreateInfo.flags = 0;

                VkResult ret = vkCreateSemaphore(vkdev->GetDevice(), &semaphoreCreateInfo, 0, &upload_compute_semaphore);
                CHECK_EQ(ret, VK_SUCCESS) << Format("vkCreateSemaphore failed %d", ret);
            }

            // upload_command_fence
            {
                VkFenceCreateInfo fenceCreateInfo;
                fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                fenceCreateInfo.pNext = 0;
                fenceCreateInfo.flags = 0;

                VkResult ret = vkCreateFence(vkdev->GetDevice(), &fenceCreateInfo, 0, &upload_command_fence);
                CHECK_EQ(ret, VK_SUCCESS) << Format("vkCreateFence failed %d", ret);
            }
        }

        BeginCommandBuffer();
    }

    void VkTransfer::BeginCommandBuffer()
    {
        {
            VkCommandBufferBeginInfo commandBufferBeginInfo;
            commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            commandBufferBeginInfo.pNext = 0;
            commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            commandBufferBeginInfo.pInheritanceInfo = 0;

            VkResult ret = vkBeginCommandBuffer(compute_command_buffer, &commandBufferBeginInfo);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkBeginCommandBuffer failed %d", ret);
        }

        if (!vkdev->info.unified_compute_transfer_queue)
        {
            VkCommandBufferBeginInfo commandBufferBeginInfo;
            commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            commandBufferBeginInfo.pNext = 0;
            commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            commandBufferBeginInfo.pInheritanceInfo = 0;

            VkResult ret = vkBeginCommandBuffer(upload_command_buffer, &commandBufferBeginInfo);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkBeginCommandBuffer failed %d", ret);
        }
    }

    void VkTransfer::EndCommandBuffer()
    {
        {
            VkResult ret = vkEndCommandBuffer(compute_command_buffer);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkEndCommandBuffer failed %d", ret);
        }

        if (!vkdev->info.unified_compute_transfer_queue)
        {
            VkResult ret = vkEndCommandBuffer(upload_command_buffer);
            CHECK_EQ(ret, VK_SUCCESS) << Format("vkEndCommandBuffer failed %d", ret);
        }
    }
}