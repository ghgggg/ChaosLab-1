#pragma once

#include "core/core.hpp"
#include "core/vulkan/gpu.hpp"

#include <map>

namespace chaos
{
	namespace dnn
	{
		struct LayerShaderRegistryEntry
		{
			const char* name;
			const uint32_t* spv_data;
			size_t spv_data_size;
		};

		class CHAOS_API LayerShaderRegistry
		{
		public:
			using ShaderRegistry = std::vector<LayerShaderRegistryEntry>;

			static ShaderRegistry& Registry();
			static void AddShader(const char* name, const uint32_t* spv_data, size_t size);

			static int GetIndex(const char* name);
		private:
			LayerShaderRegistry() = default;
		};

		class CHAOS_API ShaderRegisterer
		{
		public:
			ShaderRegisterer(const char* name, const uint32_t* data, size_t size);
		};
	}
}

#define REGISTER_SHADER(name) \
static ShaderRegisterer shader_##name(#name, ##name##_spv_data, sizeof(##name##_spv_data));