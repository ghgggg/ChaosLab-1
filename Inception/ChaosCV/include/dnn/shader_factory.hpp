#pragma once

#include "core/core.hpp"

namespace chaos
{
	namespace dnn
	{
		struct LayerShaderRegistryEntry
		{
			//const char* name;
			const uint32_t* spv_data;
			size_t spv_data_size;
		};

		class CHAOS_API LayerShaderRegistry
		{
		public:
			using ShaderRegistry = std::vector<LayerShaderRegistryEntry>;

			static ShaderRegistry& Registry();
			static void AddShader(const uint32_t* spv_data, size_t size);

		private:
			LayerShaderRegistry() = default;
		};

		//class CHAOS_API ShaderRegisterer
		//{
		//public:
		//	ShaderRegisterer(const char* name, const uint32_t* data, size_t size);
		//};

	}
}