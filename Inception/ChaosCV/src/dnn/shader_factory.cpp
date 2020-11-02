#include "dnn/shader_factory.hpp"

namespace chaos
{
	namespace dnn
	{
		LayerShaderRegistry::ShaderRegistry& LayerShaderRegistry::Registry()
		{
			static auto registry = Ptr<ShaderRegistry>(new ShaderRegistry());
			return *registry;
		}

		void LayerShaderRegistry::AddShader(const char* name, const uint32_t* spv_data, size_t size)
		{
			ShaderRegistry& registry = Registry();
			registry.push_back({name, spv_data, size});
		}

		int LayerShaderRegistry::GetIndex(const char* name)
		{
			const ShaderRegistry& registry = Registry();
			int idx = 0;
			for (const auto& shader : registry)
			{
				if (0 == std::strcmp(shader.name, name)) return idx;
				idx++;
			}
			LOG(FATAL) << "can not find shader named " << name;
			return -1;
		}

		ShaderRegisterer::ShaderRegisterer(const char* name, const uint32_t* spv_data, size_t size)
		{
			LayerShaderRegistry::AddShader(name, spv_data, size);
		}
	}
}