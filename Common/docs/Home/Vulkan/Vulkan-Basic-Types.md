|**Vulkan Types**|**Description**|
|:---|:---|
|VkInstance|用于存储Vulkan程序相关状态的软件结构，可以在逻辑上区分不同的Vulkan应用程序或者同一应用程序内部不同的Vulkan上下文|
|VkPhysicalDevice|对系统中 GPU 硬件的抽象，每个 GPU 对应一个物理设备。另外，每个实例下可以有多个物理设备|
|VkDevice|基于物理设备创建的逻辑设备，本质上是存储信息的软件结构，其中主要保留了与对应物理设备相关的资源。每个物理设备可以对应多个逻辑设备|
|VkDeviceMemory|设备内存的逻辑抽象，缓冲(VuBuffer)、图形(VkImage)都需要绑定设备内存才能正常工作|
|VkBuffer|设备内存的一种使用模式，这种模式下对应的内存用于存储各种数据。比如：绘制用顶点信息数据、绘制用一致变量数据等|