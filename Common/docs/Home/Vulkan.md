# 前言
以下内容为通过NCNN的具体实现对Vulkan的学习结果，可能存在错误以及过时，会在逐步添加Vulkan的时候记录下每个相关对象的理解和使用方法等，请谨慎食用！

# 整体计算流程
1. Create GPU Instance  
   实例化```VkInstance```并获取GPU Info以及```VkDevice```
2. 创建对应的Allocator  
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
从计算流程中可以看到，Vulkan的代码至少包含5个比较重要的部分：VkDevice，Allocator，Command，Pipeline和Shader。
