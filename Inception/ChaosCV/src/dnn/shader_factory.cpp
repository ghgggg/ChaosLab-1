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

		void LayerShaderRegistry::AddShader(const uint32_t* spv_data, size_t size)
		{
			ShaderRegistry& registry = Registry();
		}
	}
}