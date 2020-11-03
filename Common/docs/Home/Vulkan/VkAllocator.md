# VkAllocator

# VkBlobAllocator
这部分应该是用于矩阵运算的，在独显上，无法进行map  
CPU无法获取该Allocator分配的内存，因此无法使用Command实现download操作

# VkStagingAllocator
CPU可以通过mapped_data获取GPU数据