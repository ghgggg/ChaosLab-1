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

# 其他
gl_GlobalInvocationID是当前执行单元在全局工作组中的位置的一种有效的三维索引


