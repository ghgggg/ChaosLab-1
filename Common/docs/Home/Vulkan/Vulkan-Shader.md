着色器模块，在Vulkan中必须以二进制字节码的格式使用，可以通过SDK中提供的glslangValidator将类似*.comp的GLSL代码编译成SPIR-V。  
GLSL代码需要一个main函数作为入口。

# 传递数据
主题上分为三种方式（大概）   
layout constant_id 通过VkSpecializationType传递，需要在create pipeline的时候确定下来，整个生存周期都无法改变   
layout binding传递矩阵，通过readonly和writeonly来限定只读或者只写   
layout push_constant用于传递一些常量，在生存周期中可以更新   

# 解析
类型：`uint32_t`  
数据最开头5个值分别表示：
 - magic 
 - version 
 - generator 
 - bound 
 - schema  
接下来由多个op组成，对于每个op，类型（低16位）和长度（高16位）存储于第一个`uint32_t`中，依据op的类型会有不同的解析形式。目前已知的op类型包含以下几种（仅出现于NCNN代码中中）：   

|**OP**|**Value**|**Description**|
|:---|:---:|---|
|Name|5|可以获取Shader的名字|
|MemberName|6|-|
|ExecutionMode|16|在mode=17时，在NCNN中应该是对应设置数据的3个维度|
|TypeImage|25|-|
|TypeSampledImage|27|-|
|TypePointer|32|-|
|SpecContant|50|-|
|SpecContantComposite|51|-|
|Variable|59|-|
|Decorate|71|-|

# 约定
-Dsfp=float -Dsfpvec2=vec2 -Dsfpvec4=vec4 -Dsfpvec8=mat2x4 -Dsfpmat4=mat4  
-Dafp=float -Dafpvec2=vec2 -Dafpvec4=vec4 -Dafpvec8=mat2x4 -Dafpmat4=mat4  
直觉上afp和sfp是用来区分binding data和临时变量的   
-D psc(x)=(x==0?p.x:x) 三目运算符
-D buffer_ld1(buf,i)=buf[i] 取buffer，packing 1
-D buffer_st1(buf,i,v)={buf[i]=v;} 设置buffer， packing 1

# Pipeline
目前看到的layout类型包括3种
 - constant_id
 - binding
 - push_constant

# 编译
glslangValidator -Dsfp=float -Dafp=float "-D buffer_ld1(buf,i)=buf[i]" "-D buffer_st1(buf,i,v)={buf[i]=v;}" "-D psc(x)=(x==0?p.x:x)" -V  --vn innerproduct -x -o innerproduct.spv.hex.hpp innerproduct.comp

# 其他
gl_GlobalInvocationID是当前执行单元在全局工作组中的位置的一种有效的三维索引


