#include "core/vulkan/pipeline.hpp"

#include "dnn/option.hpp"

namespace chaos
{
    // https://en.wikipedia.org/wiki/MurmurHash
    static uint32_t murmur3_32(const uint32_t* data, int size)
    {
        uint32_t h = 0;

        for (int i = 0; i < size; i++)
        {
            uint32_t k = *data++;

            k *= 0xcc9e2d51;
            k = (k << 15) | (k >> (32 - 15));
            k *= 0x1b873593;

            h ^= k;
            h = (h << 13) | (h >> (32 - 13));
            h = (h * 5) + 0xe6546b64;
        }

        h ^= size * 4;

        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;

        return h;
    }

    // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1a_hash
    static uint32_t fnv1a_32(const uint8_t* data, int size)
    {
        uint32_t h = 0x811c9dc5;

        for (int i = 0; i < size; i++)
        {
            h ^= (uint32_t)*data++;
            h *= 0x01000193;
        }

        return h;
    }

    PipelineCache::pipeline_cache_digest::pipeline_cache_digest(const uint32_t* spv_data, size_t spv_data_size, const std::vector<VkSpecializationType>& specializations,
        uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z)
    {
        spv_data_murmur3 = murmur3_32(spv_data, (int)spv_data_size / 4);

        // encode opt
        opt_local_size_bits[0] = 0;

        // encode local_size
        opt_local_size_bits[1] = local_size_x;
        opt_local_size_bits[2] = local_size_y;
        opt_local_size_bits[3] = local_size_z;

        // encode specializations
        const int specialization_count = (int)specializations.size();
        specializations_murmur3 = murmur3_32((const uint32_t*)specializations.data(), specialization_count);
        specializations_fnv1a = fnv1a_32((const uint8_t*)specializations.data(), specialization_count * sizeof(VkSpecializationType));
    }

    PipelineCache::pipeline_cache_digest::pipeline_cache_digest(int _shader_type_index, const dnn::Option& opt, const std::vector<VkSpecializationType>& specializations,
        uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z)
    {
        shader_type_index = _shader_type_index;

        // encode opt
        opt_local_size_bits[0] = opt.use_fp16_packed << 6
            | opt.use_fp16_storage << 5
            | opt.use_fp16_arithmetic << 4
            | opt.use_int8_storage << 3
            | opt.use_int8_arithmetic << 2;

        // encode local_size
        opt_local_size_bits[1] = local_size_x;
        opt_local_size_bits[2] = local_size_y;
        opt_local_size_bits[3] = local_size_z;

        // encode specializations
        const int specialization_count = (int)specializations.size();
        specializations_murmur3 = murmur3_32((const uint32_t*)specializations.data(), specialization_count);
        specializations_fnv1a = fnv1a_32((const uint8_t*)specializations.data(), specialization_count * sizeof(VkSpecializationType));
    }


    PipelineCache::PipelineCache(const VulkanDevice* vkdev) : vkdev(vkdev) {}

    PipelineCache::~PipelineCache() { Clear(); }

    void PipelineCache::Clear()
    {
        std::lock_guard lock(cache_lock);

        for (size_t i = 0; i < cache_artifacts.size(); i++)
        {
            const pipeline_cache_artifact& cc = cache_artifacts[i];

            if (vkdev->info.support_VK_KHR_descriptor_update_template)
            {
                if (cc.descriptor_update_template)
                {
                    vkdev->vkDestroyDescriptorUpdateTemplateKHR(vkdev->GetDevice(), cc.descriptor_update_template, 0);
                }
            }

            if (cc.pipeline)
            {
                vkDestroyPipeline(vkdev->GetDevice(), cc.pipeline, 0);
            }

            if (cc.pipeline_layout)
            {
                vkDestroyPipelineLayout(vkdev->GetDevice(), cc.pipeline_layout, 0);
            }

            if (cc.descriptorset_layout)
            {
                vkDestroyDescriptorSetLayout(vkdev->GetDevice(), cc.descriptorset_layout, 0);
            }

            if (cc.shader_module)
            {
                vkDestroyShaderModule(vkdev->GetDevice(), cc.shader_module, 0);
            }
        }

        cache_digests.clear();
        cache_artifacts.clear();
    }

    void PipelineCache::GetPipeline(const uint32_t* spv_data, size_t spv_data_size, const std::vector<VkSpecializationType>& specializations,
        uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
        VkShaderModule* _shader_module,
        VkDescriptorSetLayout* descriptorset_layout,
        VkPipelineLayout* pipeline_layout,
        VkPipeline* pipeline,
        VkDescriptorUpdateTemplateKHR* descriptor_update_template,
        ShaderInfo& shader_info) const
    {
        std::lock_guard lock(cache_lock);

        pipeline_cache_digest key(spv_data, spv_data_size, specializations, local_size_x, local_size_y, local_size_z);

        if (!vkdev->info.bug_corrupted_online_pipeline_cache)
        {
            // find cache
            for (size_t i = 0; i < cache_digests.size(); i++)
            {
                if (cache_digests[i] != key)
                    continue;

                // hit cache
                const pipeline_cache_artifact& cc = cache_artifacts[i];

                *_shader_module = cc.shader_module;
                *descriptorset_layout = cc.descriptorset_layout;
                *pipeline_layout = cc.pipeline_layout;
                *pipeline = cc.pipeline;
                *descriptor_update_template = cc.descriptor_update_template;
                shader_info = cc.shader_info;

                return;
            }
        }

        ResolveShaderInfo(spv_data, spv_data_size, shader_info);

        VkShaderModule shader_module = vkdev->CompileShaderModule(spv_data, spv_data_size, local_size_x, local_size_y, local_size_z);
        CHECK(shader_module) << "create_shader_module failed";

        NewPipeline(shader_module, shader_info, specializations, descriptorset_layout, pipeline_layout, pipeline, descriptor_update_template);
        //if (ret != 0)
        //{
        //    //NCNN_LOGE("new_pipeline failed");
        //    vkDestroyShaderModule(vkdev->GetDevice(), shader_module, 0);
        //    LOG(FATAL) << "NewPipeline fai";
        //    return;
        //}

        *_shader_module = shader_module;

        // save to cache
        {
            pipeline_cache_artifact cc;

            cc.shader_module = *_shader_module;
            cc.descriptorset_layout = *descriptorset_layout;
            cc.pipeline_layout = *pipeline_layout;
            cc.pipeline = *pipeline;
            cc.descriptor_update_template = *descriptor_update_template;
            //cc.shader_info = shader_info;

            cache_digests.push_back(key);
            cache_artifacts.push_back(cc);
        }
    }

    void PipelineCache::GetPipeline(int shader_type_index, const dnn::Option& opt, const std::vector<VkSpecializationType>& specializations,
        uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
        VkShaderModule* _shader_module,
        VkDescriptorSetLayout* descriptorset_layout,
        VkPipelineLayout* pipeline_layout,
        VkPipeline* pipeline,
        VkDescriptorUpdateTemplateKHR* descriptor_update_template,
        ShaderInfo& shader_info) const
    {
        std::lock_guard lock(cache_lock);

        pipeline_cache_digest key(shader_type_index, opt, specializations, local_size_x, local_size_y, local_size_z);

        if (!vkdev->info.bug_corrupted_online_pipeline_cache)
        {
            // find cache
            for (size_t i = 0; i < cache_digests.size(); i++)
            {
                if (cache_digests[i] != key)
                    continue;

                // hit cache
                const pipeline_cache_artifact& cc = cache_artifacts[i];

                *_shader_module = cc.shader_module;
                *descriptorset_layout = cc.descriptorset_layout;
                *pipeline_layout = cc.pipeline_layout;
                *pipeline = cc.pipeline;
                *descriptor_update_template = cc.descriptor_update_template;
                shader_info = cc.shader_info;

            }
        }

        int ret = 0;

        // create new pipeline
        VkShaderModule shader_module = 0;
        CreateShaderModule(shader_type_index, opt, local_size_x, local_size_y, local_size_z, &shader_module, shader_info);

        NewPipeline(shader_module, shader_info, specializations, descriptorset_layout, pipeline_layout, pipeline, descriptor_update_template);

        *_shader_module = shader_module;

        // save to cache
        {
            pipeline_cache_artifact cc;

            cc.shader_module = *_shader_module;
            cc.descriptorset_layout = *descriptorset_layout;
            cc.pipeline_layout = *pipeline_layout;
            cc.pipeline = *pipeline;
            cc.descriptor_update_template = *descriptor_update_template;
            cc.shader_info = shader_info;

            cache_digests.push_back(key);
            cache_artifacts.push_back(cc);
        }
    }




    void PipelineCache::CreateShaderModule(int shader_type_index, const dnn::Option& opt, uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
        VkShaderModule* _shader_module, ShaderInfo& si) const
    {
        // ncnn_add_shader cmake macro
        // 0 = fp32
        // 1 = fp16p
        // 2 = fp16pa
        // 3 = fp16s
        // 4 = fp16sa
        // 5 = image
        // 6 = image_fp16p
        // 7 = image_fp16pa
        // 8 = image_fp16s
        // 9 = image_fp16sa
        if (vkdev->info.support_fp16_storage && opt.use_fp16_storage && vkdev->info.support_fp16_arithmetic && opt.use_fp16_arithmetic)
        {
            shader_type_index += 4;
        }
        else if (vkdev->info.support_fp16_packed && opt.use_fp16_packed && vkdev->info.support_fp16_arithmetic && opt.use_fp16_arithmetic)
        {
            shader_type_index += 2;
        }
        else if (vkdev->info.support_fp16_storage && opt.use_fp16_storage)
        {
            shader_type_index += 3;
        }
        else if (vkdev->info.support_fp16_packed && opt.use_fp16_packed)
        {
            shader_type_index += 1;
        }

        si = GetShaderInfo(shader_type_index);

        VkShaderModule shader_module = vkdev->CreateShaderModule(shader_type_index, local_size_x, local_size_y, local_size_z);
        CHECK(shader_module) << "create shader modul failed";

        *_shader_module = shader_module;
    }

    void PipelineCache::NewPipeline(VkShaderModule shader_module, const ShaderInfo& shader_info, const std::vector<VkSpecializationType>& specializations,
        VkDescriptorSetLayout* _descriptorset_layout,
        VkPipelineLayout* _pipeline_layout,
        VkPipeline* _pipeline,
        VkDescriptorUpdateTemplateKHR* _descriptor_update_template) const
    {
        VkDescriptorSetLayout descriptorset_layout = 0;
        VkPipelineLayout pipeline_layout = 0;
        VkPipeline pipeline = 0;
        VkDescriptorUpdateTemplateKHR descriptor_update_template = 0;

        // create new pipeline
        if ((int)specializations.size() != shader_info.specialization_count)
        {
            //NCNN_LOGE("pipeline specialization count mismatch, expect %d but got %d", shader_info.specialization_count, (int)specializations.size());
            // ERROR_PipelineCache:
            if (vkdev->info.support_VK_KHR_descriptor_update_template)
            {
                if (descriptor_update_template)
                {
                    vkdev->vkDestroyDescriptorUpdateTemplateKHR(vkdev->GetDevice(), descriptor_update_template, 0);
                }
            }

            if (pipeline)
            {
                vkDestroyPipeline(vkdev->GetDevice(), pipeline, 0);
            }

            if (pipeline_layout)
            {
                vkDestroyPipelineLayout(vkdev->GetDevice(), pipeline_layout, 0);
            }

            if (descriptorset_layout)
            {
                vkDestroyDescriptorSetLayout(vkdev->GetDevice(), descriptorset_layout, 0);
            }
            LOG(FATAL) << Format("pipeline specialization count mismatch, expect %d but got %d", shader_info.specialization_count, (int)specializations.size());
        }
        // 需要释放额外的内容，则
        vkdev->CreateDescriptorsetLayout(shader_info.binding_count, shader_info.binding_types, &descriptorset_layout);

        vkdev->CreatePipelineLayout(shader_info.push_constant_count, descriptorset_layout, &pipeline_layout);

        vkdev->CreatePipeline(shader_module, pipeline_layout, specializations, &pipeline);

        if (vkdev->info.support_VK_KHR_descriptor_update_template)
        {
            vkdev->CreateDescriptorUpdateTemplate(shader_info.binding_count, shader_info.binding_types, descriptorset_layout, pipeline_layout, &descriptor_update_template);
        }

        *_descriptorset_layout = descriptorset_layout;
        *_pipeline_layout = pipeline_layout;
        *_pipeline = pipeline;
        *_descriptor_update_template = descriptor_update_template;
    }




    Pipeline::Pipeline(const VulkanDevice* vkdev) : vkdev(vkdev)
    {
        shader_module = 0;
        descriptorset_layout = 0;
        pipeline_layout = 0;
        pipeline = 0;
        descriptor_update_template = 0;

        local_size_x = 1;
        local_size_y = 1;
        local_size_z = 1;
    }
    Pipeline::~Pipeline() {}

    void Pipeline::SetOptimalLocalSizeXYZ(int x, int y, int z)
    {
        SetOptimalLocalSizeXYZ(Shape(z, y, x));
    }
    void Pipeline::SetOptimalLocalSizeXYZ(const Shape& local_size_xyz)
    {
        int w = local_size_xyz.GetX();
        int h = local_size_xyz.GetY();
        int c = local_size_xyz.GetZ();

        if (w == 0 && h == 0 && c == 0)
        {
            // fallback to the common and safe 4x4x4
            w = 4;
            h = 4;
            c = 4;
        }

        w = std::min(w, (int)vkdev->info.max_workgroup_size[0]);
        h = std::min(h, (int)vkdev->info.max_workgroup_size[1]);
        c = std::min(c, (int)vkdev->info.max_workgroup_size[2]);

        if (w * h * c <= (int)vkdev->info.max_workgroup_invocations)
        {
            return SetLocalSizeXYZ(w, h, c);
        }

        int max_local_size_xy = (int)vkdev->info.max_workgroup_invocations / c;

        int wh_max = std::max(1, (int)sqrt(max_local_size_xy));
        while (w * h >= wh_max)
        {
            w = std::max(1, w / 2);
            h = std::max(1, h / 2);
        }

        SetLocalSizeXYZ(w, h, c);
    }
    void Pipeline::SetLocalSizeXYZ(int x, int y, int z)
    {
        local_size_x = x; // w;
        local_size_y = y; // h;
        local_size_z = z; // c;
    }

    void Pipeline::Create(const uint32_t* spv_data, size_t spv_data_size, const std::vector<VkSpecializationType>& specializations)
    {
        const PipelineCache* pipeline_cache = vkdev->GetPipelineCache();

        // get from pipeline cache
        return pipeline_cache->GetPipeline(spv_data, spv_data_size, specializations, local_size_x, local_size_y, local_size_z,
            &shader_module, &descriptorset_layout, &pipeline_layout, &pipeline, &descriptor_update_template,
            shader_info);
    }

    void Pipeline::Create(int shader_type_index, const dnn::Option& opt, const std::vector<VkSpecializationType>& specializations)
    {
        const PipelineCache* pipeline_cache = opt.pipeline_cache ? opt.pipeline_cache : vkdev->GetPipelineCache();

        // get from pipeline cache
        return pipeline_cache->GetPipeline(shader_type_index, opt, specializations, local_size_x, local_size_y, local_size_z,
            &shader_module, &descriptorset_layout, &pipeline_layout, &pipeline, &descriptor_update_template,
            shader_info);
    }

}