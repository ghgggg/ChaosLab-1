# VkInstance
在使用任何和Vulkan相关的对象之前，必须实列化`VkInstance`  

`class VulkanInstanceHolder`对VkInstance进行了封装，便于自动释放相关的资源  
在`CreateGPUInstance`中，通过vkCreateInstance创建并获取所有的GPUInfo保存到`g_gpu_infos`数组中，最大GPU数量为8.

# GPUInfo
通过`VkPhysicalDevice`以及`VkPhysicalDeviceMemoryProperties`获取硬件信息  
  - `VkPhysicalDevice`  
  对系统中 GPU 硬件的抽象，每个 GPU 对应一个物理设备。另外，每个实例下可以有多个物理设备  
  - `VkPhysicalDeviceMemoryProperties`  
  用于存储获取的基于指定GPU的设备内存属性，包括内存类型数量、内存类型列表、内存堆数量、内存堆列表等  

在`TryCreateGPUInstance`之后，直接从静态变量`g_gpu_infos`数组中获取对应device index的`GPUInfo`

# VulkanDevice
 - `VkDevice`  
 基于物理设备创建的逻辑设备，本质上是存储信息的软件结构，其中主要保留了与对应物理设备相关的资源。每个物理设备可以对应多个逻辑设备
 - `VkQueue`  
 功能为接收提交的任务，将任务按序由所属GPU硬件依次执行
 - `VkAllocator`  
 Vulkan内存池，具体请参考[VkAllocator](/Home/Vulkan/VkAllocator)文档
 - `VkSampler`  
 暂时不清楚此对象的作用
