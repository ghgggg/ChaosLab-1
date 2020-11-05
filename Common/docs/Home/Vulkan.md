# 前言
以下内容为通过NCNN的具体实现对Vulkan的学习结果，可能存在错误以及过时，会在逐步添加Vulkan的时候记录下每个相关对象的理解和使用方法等，请谨慎食用！

# 整体计算流程
1. Create GPU Instance  
   实例化```VkInstance```并获取GPU Info以及```VkDevice```
2. 创建对应的VkAllocator  
   管理```VkBuffer```和```VkDeviceMemory```对象
3. 创建Command用于执行实际的命令  
   包括数据的Upload、Download以及Clone，Pipeline的绑定（感觉用词不准确）
4. Create Pipeline  
   在项目中只使用了Compute Shader，可以理解为Pipeline是对数据处理的过程的封装，Shader上实现具体的功能，理解为实际在GPU上运行的代码。
5. Submit and Wait  
   上传数据，并使用Command Record Pipeline后，下载数据到CPU。提交后再次提交前，需要对Command进行Reset
6. 销毁资源
   在销毁VkInstance之前，需要销毁所有和Vulkan相关的资源

# 模块
从计算流程中可以看到，Vulkan的代码至少包含5个比较重要的部分：VkDevice，VkAllocator，Command，Pipeline和Shader。更详细的信息可以参考[Basic Types](/Home/Vulkan/Vulkan-Basic-Types)以及相关的文档
## VkDevice
Vulkan获取的逻辑设备，后续所有的模块都需要传入```VkDevice```
## [VkAllocator](/Home/Vulkan/VkAllocator)
显存管理（```VkDeviceMemory```）,```VkBuffer```需要绑定设备内存，所有GPU矩阵的内存分配都需要指定VkAllocator
## [Command](/Home/Vulkan/Command)
命令缓冲，记录或执行计算任务的各个命令
## [Pipeline](/Home/Vulkan/Pipeline)
管线，包括了具体的布局、顶点数据输入情况等设置
## [Shader](/Home/Vulkan/Vulkan-Shader)
着色器，个人理解是GPU执行单元，通过Record Pipeline执行具体的操作，其代码类似C，需要通过sdk提供的编译器编译成二进制码使用