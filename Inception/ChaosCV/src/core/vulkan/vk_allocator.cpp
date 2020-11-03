#include "core/vulkan/gpu.hpp"
#include "core/vulkan/vk_allocator.hpp"

namespace chaos
{
    VkAllocator::VkAllocator(const VulkanDevice* _vkdev)
        : vkdev(_vkdev)
    {
        buffer_memory_type_index = (uint32_t)-1;
        //image_memory_type_index = (uint32_t)-1;
        mappable = false;
        coherent = false;
    }

    static inline size_t RoundUp(size_t n, size_t multiple)
    {
        return (n + multiple - 1) / multiple * multiple;
    }

    static inline size_t RoundDown(size_t n, size_t multiple)
    {
        return n / multiple * multiple;
    }

    void VkAllocator::Flush(VkBufferMemory* data)
    {
        if (coherent) return;

        VkMappedMemoryRange mappedMemoryRange;
        mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        mappedMemoryRange.pNext = 0;
        mappedMemoryRange.memory = data->memory;
        mappedMemoryRange.offset = RoundDown(data->offset, vkdev->info.non_coherent_atom_size);
        mappedMemoryRange.size = RoundUp(data->offset + data->capacity, vkdev->info.non_coherent_atom_size) - mappedMemoryRange.offset;

        VkResult ret = vkFlushMappedMemoryRanges(vkdev->GetDevice(), 1, &mappedMemoryRange);
        CHECK_EQ(ret, VK_SUCCESS) << Format("vkFlushMappedMemoryRanges failed %d", ret);
    }

    void VkAllocator::Invalidate(VkBufferMemory* ptr)
    {
        if (coherent) return;

        VkMappedMemoryRange mappedMemoryRange;
        mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        mappedMemoryRange.pNext = 0;
        mappedMemoryRange.memory = ptr->memory;
        mappedMemoryRange.offset = RoundDown(ptr->offset, vkdev->info.non_coherent_atom_size);
        mappedMemoryRange.size = RoundUp(ptr->offset + ptr->capacity, vkdev->info.non_coherent_atom_size) - mappedMemoryRange.offset;

        VkResult ret = vkInvalidateMappedMemoryRanges(vkdev->GetDevice(), 1, &mappedMemoryRange);
        CHECK_EQ(ret, VK_SUCCESS) << Format("vkInvalidateMappedMemoryRanges failed %d", ret);
    }

    VkBuffer VkAllocator::CreateBuffer(size_t size, VkBufferUsageFlags usage)
    {
        VkBufferCreateInfo bufferCreateInfo;
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.pNext = 0;
        bufferCreateInfo.flags = 0;
        bufferCreateInfo.size = size;
        bufferCreateInfo.usage = usage;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferCreateInfo.queueFamilyIndexCount = 0;
        bufferCreateInfo.pQueueFamilyIndices = 0;

        VkBuffer buffer = 0;
        VkResult ret = vkCreateBuffer(vkdev->GetDevice(), &bufferCreateInfo, 0, &buffer);
        CHECK_EQ(ret, VK_SUCCESS) << Format("vkCreateBuffer failed %d", ret);

        return buffer;
    }

    VkDeviceMemory VkAllocator::AllocateMemory(size_t size, uint32_t memory_type_index)
    {
        VkMemoryAllocateInfo memoryAllocateInfo;
        memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryAllocateInfo.pNext = 0;
        memoryAllocateInfo.allocationSize = size;
        memoryAllocateInfo.memoryTypeIndex = memory_type_index;

        VkDeviceMemory memory = 0;
        VkResult ret = vkAllocateMemory(vkdev->GetDevice(), &memoryAllocateInfo, 0, &memory);
        CHECK_EQ(ret, VK_SUCCESS) << Format("vkAllocateMemory failed %d", ret);

        return memory;
    }

    VkDeviceMemory VkAllocator::AllocateDedicatedMemory(size_t size, uint32_t memory_type_index, VkImage image, VkBuffer buffer)
    {
        VkMemoryAllocateInfo memoryAllocateInfo;
        memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryAllocateInfo.pNext = 0;
        memoryAllocateInfo.allocationSize = size;
        memoryAllocateInfo.memoryTypeIndex = memory_type_index;

        VkMemoryDedicatedAllocateInfoKHR memoryDedicatedAllocateInfo;
        memoryDedicatedAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
        memoryDedicatedAllocateInfo.pNext = 0;
        memoryDedicatedAllocateInfo.image = image;
        memoryDedicatedAllocateInfo.buffer = buffer;
        memoryAllocateInfo.pNext = &memoryDedicatedAllocateInfo;

        VkDeviceMemory memory = 0;
        VkResult ret = vkAllocateMemory(vkdev->GetDevice(), &memoryAllocateInfo, 0, &memory);
        CHECK_EQ(ret, VK_SUCCESS) << Format("vkAllocateMemory failed %d", ret);

        return memory;
    }




    static inline size_t least_common_multiple(size_t a, size_t b)
    {
        if (a == b)
            return a;

        if (a > b)
            return least_common_multiple(b, a);

        size_t lcm = b;
        while (lcm % a != 0)
        {
            lcm += b;
        }

        return lcm;
    }

    VkBlobAllocator::VkBlobAllocator(const VulkanDevice* _vkdev)
        : VkAllocator(_vkdev)
    {
        buffer_offset_alignment = vkdev->info.buffer_offset_alignment;
        bind_memory_offset_alignment = vkdev->info.buffer_image_granularity;

        if (vkdev->info.type == 1)
        {
            // on integrated gpu, there may be device local only memory too, eg. AMD APU
            // assuming larger alignment always keeps us safe :)

            // least common multiple for memory_map_alignment and buffer_offset_alignment and non_coherent_atom_size
            buffer_offset_alignment = least_common_multiple(buffer_offset_alignment, vkdev->info.memory_map_alignment);
            buffer_offset_alignment = least_common_multiple(buffer_offset_alignment, vkdev->info.non_coherent_atom_size);
        }

        block_size = AlignSize(16 * 1024 * 1024, (int)buffer_offset_alignment); // 16M
    }


    VkBlobAllocator::~VkBlobAllocator()
    {
        Clear();
    }

    void VkBlobAllocator::Clear()
    {
        //     NCNN_LOGE("VkBlobAllocator %lu", buffer_blocks.size());

        for (size_t i = 0; i < buffer_blocks.size(); i++)
        {
            VkBufferMemory* ptr = buffer_blocks[i];

            //         std::list< std::pair<size_t, size_t> >::iterator it = buffer_budgets[i].begin();
            //         while (it != buffer_budgets[i].end())
            //         {
            //             NCNN_LOGE("VkBlobAllocator budget %p %lu %lu", ptr->buffer, it->first, it->second);
            //             it++;
            //         }

            if (mappable)
                vkUnmapMemory(vkdev->GetDevice(), ptr->memory);

            vkDestroyBuffer(vkdev->GetDevice(), ptr->buffer, 0);
            vkFreeMemory(vkdev->GetDevice(), ptr->memory, 0);

            delete ptr;
        }
        buffer_blocks.clear();

        buffer_budgets.clear();

        for (size_t i = 0; i < image_memory_blocks.size(); i++)
        {
            VkDeviceMemory memory = image_memory_blocks[i];

            //         std::list< std::pair<size_t, size_t> >::iterator it = image_memory_budgets[i].begin();
            //         while (it != image_memory_budgets[i].end())
            //         {
            //             NCNN_LOGE("VkBlobAllocator budget %p %lu %lu", memory, it->first, it->second);
            //             it++;
            //         }

            vkFreeMemory(vkdev->GetDevice(), memory, 0);
        }
        image_memory_blocks.clear();

        image_memory_budgets.clear();
    }

    VkBufferMemory* VkBlobAllocator::FastMalloc(size_t size)
    {
        size_t aligned_size = AlignSize(size, (int)buffer_offset_alignment);

        const int buffer_block_count = (int)buffer_blocks.size();

        // find first spare space in buffer_blocks
        for (int i = 0; i < buffer_block_count; i++)
        {
            std::list<std::pair<size_t, size_t> >::iterator it = buffer_budgets[i].begin();
            while (it != buffer_budgets[i].end())
            {
                size_t budget_size = it->second;
                if (budget_size < aligned_size)
                {
                    it++;
                    continue;
                }

                // return sub buffer
                VkBufferMemory* ptr = new VkBufferMemory;

                ptr->buffer = buffer_blocks[i]->buffer;
                ptr->offset = it->first;
                ptr->memory = buffer_blocks[i]->memory;
                ptr->capacity = aligned_size;
                ptr->mapped_data = buffer_blocks[i]->mapped_data;
                ptr->access_flags = 0;
                ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

                // adjust buffer_budgets
                if (budget_size == aligned_size)
                {
                    buffer_budgets[i].erase(it);
                }
                else
                {
                    it->first += aligned_size;
                    it->second -= aligned_size;
                }

                // NCNN_LOGE("VkBlobAllocator M %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

                return ptr;
            }
        }

        size_t new_block_size = std::max(block_size, aligned_size);

        // create new block
        VkBufferMemory* block = new VkBufferMemory;

        block->buffer = CreateBuffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        block->offset = 0;

        // TODO respect VK_KHR_dedicated_allocation ?

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(vkdev->GetDevice(), block->buffer, &memoryRequirements);

        // setup memory type and alignment
        if (buffer_memory_type_index == (uint32_t)-1)
        {
            if (vkdev->info.type == 1)
            {
                // integrated gpu, prefer unified memory
                buffer_memory_type_index = vkdev->FindMemoryIndex(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
            }
            else
            {
                // discrete gpu, device local
                buffer_memory_type_index = vkdev->FindMemoryIndex(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            }

            mappable = vkdev->IsMappable(buffer_memory_type_index);
            coherent = vkdev->IsCoherent(buffer_memory_type_index);
        }

        block->memory = AllocateMemory(memoryRequirements.size, buffer_memory_type_index);

        // ignore memoryRequirements.alignment as we always bind at zero offset
        vkBindBufferMemory(vkdev->GetDevice(), block->buffer, block->memory, 0);

        block->mapped_data = 0;
        if (mappable)
        {
            vkMapMemory(vkdev->GetDevice(), block->memory, 0, new_block_size, 0, &block->mapped_data);
        }

        buffer_blocks.push_back(block);

        // return sub buffer
        VkBufferMemory* ptr = new VkBufferMemory;

        ptr->buffer = block->buffer;
        ptr->offset = 0;
        ptr->memory = block->memory;
        ptr->capacity = aligned_size;
        ptr->mapped_data = block->mapped_data;
        ptr->access_flags = 0;
        ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

        // adjust buffer_budgets
        std::list<std::pair<size_t, size_t> > budget;
        if (new_block_size > aligned_size)
        {
            budget.push_back(std::make_pair(aligned_size, new_block_size - aligned_size));
        }
        buffer_budgets.push_back(budget);

        //     NCNN_LOGE("VkBlobAllocator M %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

        return ptr;
    }

    void VkBlobAllocator::FastFree(VkBufferMemory* data)
    {
        //     NCNN_LOGE("VkBlobAllocator F %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

        const int buffer_block_count = (int)buffer_blocks.size();

        int block_index = -1;
        for (int i = 0; i < buffer_block_count; i++)
        {
            if (buffer_blocks[i]->buffer == data->buffer && buffer_blocks[i]->memory == data->memory)
            {
                block_index = i;
                break;
            }
        }

        if (block_index == -1)
        {
            //NCNN_LOGE("FATAL ERROR! unlocked VkBlobAllocator get wild %p", ptr->buffer);
            delete data;
            return;
        }

        // merge
        std::list<std::pair<size_t, size_t> >::iterator it_merge_left = buffer_budgets[block_index].end();
        std::list<std::pair<size_t, size_t> >::iterator it_merge_right = buffer_budgets[block_index].end();
        std::list<std::pair<size_t, size_t> >::iterator it = buffer_budgets[block_index].begin();
        for (; it != buffer_budgets[block_index].end(); it++)
        {
            if (it->first + it->second == data->offset)
            {
                it_merge_left = it;
            }
            else if (data->offset + data->capacity == it->first)
            {
                it_merge_right = it;
            }
        }

        if (it_merge_left != buffer_budgets[block_index].end() && it_merge_right != buffer_budgets[block_index].end())
        {
            it_merge_left->second = it_merge_right->first + it_merge_right->second - it_merge_left->first;
            buffer_budgets[block_index].erase(it_merge_right);
        }
        else if (it_merge_left != buffer_budgets[block_index].end())
        {
            it_merge_left->second = data->offset + data->capacity - it_merge_left->first;
        }
        else if (it_merge_right != buffer_budgets[block_index].end())
        {
            it_merge_right->second = it_merge_right->first + it_merge_right->second - data->offset;
            it_merge_right->first = data->offset;
        }
        else
        {
            if (data->offset == 0)
            {
                // chain leading block
                buffer_budgets[block_index].push_front(std::make_pair(data->offset, data->capacity));
            }
            else
            {
                buffer_budgets[block_index].push_back(std::make_pair(data->offset, data->capacity));
            }
        }

        delete data;
    }





    VkStagingAllocator::VkStagingAllocator(const VulkanDevice* _vkdev) : VkAllocator(_vkdev)
    {
        mappable = true;
        coherent = true;

        size_compare_ratio = 192; // 0.75f * 256
    }

    VkStagingAllocator::~VkStagingAllocator()
    {
        Clear();
    }

    void VkStagingAllocator::SetSizeCompareRatio(float scr)
    {
        //if (scr < 0.f || scr > 1.f)
        //{
        //    NCNN_LOGE("invalid size compare ratio %f", scr);
        //    return;
        //}
        CHECK(scr < 0.f || scr > 1.f) << "invalid size compare ratio " << scr;

        size_compare_ratio = (unsigned int)(scr * 256);
    }

    void VkStagingAllocator::Clear()
    {
        //     NCNN_LOGE("VkStagingAllocator %lu", buffer_budgets.size());

        for (std::list<VkBufferMemory*>::iterator it = buffer_budgets.begin(); it != buffer_budgets.end(); it++)
        {
            VkBufferMemory* ptr = *it;

            //         NCNN_LOGE("VkStagingAllocator F %p", ptr->buffer);

            vkUnmapMemory(vkdev->GetDevice(), ptr->memory);
            vkDestroyBuffer(vkdev->GetDevice(), ptr->buffer, 0);
            vkFreeMemory(vkdev->GetDevice(), ptr->memory, 0);

            delete ptr;
        }
        buffer_budgets.clear();
    }

    VkBufferMemory* VkStagingAllocator::FastMalloc(size_t size)
    {
        // find free budget
        std::list<VkBufferMemory*>::iterator it = buffer_budgets.begin();
        for (; it != buffer_budgets.end(); it++)
        {
            VkBufferMemory* ptr = *it;

            size_t capacity = ptr->capacity;

            // size_compare_ratio ~ 100%
            if (capacity >= size && ((capacity * size_compare_ratio) >> 8) <= size)
            {
                buffer_budgets.erase(it);
                return ptr;
            }
        }

        VkBufferMemory* ptr = new VkBufferMemory;

        ptr->buffer = CreateBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        ptr->offset = 0;

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(vkdev->GetDevice(), ptr->buffer, &memoryRequirements);

        // setup memory type
        if (buffer_memory_type_index == (uint32_t)-1)
        {
            buffer_memory_type_index = vkdev->FindMemoryIndex(memoryRequirements.memoryTypeBits, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }

        ptr->memory = AllocateMemory(memoryRequirements.size, buffer_memory_type_index);

        // ignore memoryRequirements.alignment as we always bind at zero offset
        vkBindBufferMemory(vkdev->GetDevice(), ptr->buffer, ptr->memory, 0);

        ptr->capacity = size;

        vkMapMemory(vkdev->GetDevice(), ptr->memory, 0, size, 0, &ptr->mapped_data);

        ptr->access_flags = 0;
        ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

        return ptr;
    }

    void VkStagingAllocator::FastFree(VkBufferMemory* ptr)
    {
        // return to buffer_budgets
        buffer_budgets.push_back(ptr);
    }
}