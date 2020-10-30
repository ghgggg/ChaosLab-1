# Pipeline Cache
这东西是干啥的？

# Pipeline

# Usage
```cpp
auto pipeline = new Pipeline(vkdev);
pipeline->SetOptimalLoaclSizeXYZ(local_size_xyz);
pipeline->Create(LayerShaderType::packing, opt, specializations)

cmd.RecordPipeline(pipeline, buffer_bindings, image_bingdings, constants, top_blobs)
```