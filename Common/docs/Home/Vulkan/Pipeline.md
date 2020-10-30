# Usage
```cpp
auto pipeline = new Pipeline(vkdev);
pipeline->SetOptimalLoaclSizeXYZ(local_size_xyz);
pipeline->Create(LayerShaderType::packing, opt, specializations)
```