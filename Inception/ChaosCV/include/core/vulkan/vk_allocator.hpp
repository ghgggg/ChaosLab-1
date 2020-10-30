#pragma once

#include "core/core.hpp"

#include <vulkan/vulkan.hpp>

namespace chaos
{
    class VulkanDevice;
    class VkBufferMemory
    {
    public:
        VkBuffer buffer;

        // the base offset assigned by allocator
        size_t offset;
        size_t capacity;

        VkDeviceMemory memory;
        void* mapped_data;

        // buffer state, modified by command functions internally
        mutable VkAccessFlags access_flags;
        mutable VkPipelineStageFlags stage_flags;

        // initialize and modified by mat
        int ref_cnt;
    };


    class CHAOS_API VkAllocator
    {
    public:
        VkAllocator(const VulkanDevice* _vkdev);
        virtual ~VkAllocator() { Clear(); }
        virtual void Clear() {}

        virtual VkBufferMemory* FastMalloc(size_t size) = 0;
        virtual void FastFree(VkBufferMemory* data) = 0;
        virtual void Flush(VkBufferMemory* data);
        virtual void Invalidate(VkBufferMemory* data);

    public:
        const VulkanDevice* vkdev;
        uint32_t buffer_memory_type_index;
        //uint32_t image_memory_type_index;
        bool mappable;
        bool coherent;

    protected:
        VkBuffer CreateBuffer(size_t size, VkBufferUsageFlags usage);
        VkDeviceMemory AllocateMemory(size_t size, uint32_t memory_type_index);
        VkDeviceMemory AllocateDedicatedMemory(size_t size, uint32_t memory_type_index, VkImage image, VkBuffer buffer);

        //VkImage CreateImage(VkImageType type, int width, int height, int depth, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage);
        //VkImageView CreateImageView(VkImageViewType type, VkImage image, VkFormat format);
    };

    class CHAOS_API VkBlobAllocator : public VkAllocator
    {
    public:
        VkBlobAllocator(const VulkanDevice* vkdev);
        virtual ~VkBlobAllocator();

    public:
        // release all budgets immediately
        virtual void Clear();

        virtual VkBufferMemory* FastMalloc(size_t size) override;
        virtual void FastFree(VkBufferMemory* ptr) override;

    protected:
        size_t block_size;
        size_t buffer_offset_alignment;
        size_t bind_memory_offset_alignment;
        std::vector<std::list<std::pair<size_t, size_t>>> buffer_budgets;
        std::vector<VkBufferMemory*> buffer_blocks;
        std::vector<std::list<std::pair<size_t, size_t>>> image_memory_budgets;
        std::vector<VkDeviceMemory> image_memory_blocks;
    };


    class CHAOS_API VkStagingAllocator : public VkAllocator
    {
    public:
        VkStagingAllocator(const VulkanDevice* vkdev);
        virtual ~VkStagingAllocator();

    public:
        // ratio range 0 ~ 1
        // default cr = 0.75
        void SetSizeCompareRatio(float scr);

        // release all budgets immediately
        virtual void Clear();

        virtual VkBufferMemory* FastMalloc(size_t size) override;
        virtual void FastFree(VkBufferMemory* ptr) override;

    protected:
        unsigned int size_compare_ratio; // 0~256
        std::list<VkBufferMemory*> buffer_budgets;
    };
    
}