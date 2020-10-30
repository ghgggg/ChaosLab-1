# Vulkan
以下内容为通过NCNN的具体实现对Vulkan的学习结果，可能存在错误以及过时，会在逐步添加Vulkan的时候记录下每个相关对象的理解和使用方法等，请谨慎食用！

# 整理流程
*.comp是Layer的GPU代码  
通过vulkan对应的工具和脚本生成static layer_shader_registry[]   

在GPUInstance的过程中，获取ShaderInfo（以前的版本似乎是直接在CreateGPUInstance中CreateShaderModule）   

如果给定一个Tensor，需要将数据上传到GPU显存中，那么在VkCompute中，需要一个Pipeline将数据上传，Pipeline会依据需要对Shader做CreateShaderModule   