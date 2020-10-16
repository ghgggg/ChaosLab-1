#include "dnn/layer_factory.hpp"

namespace chaos
{
	namespace dnn
	{
		LayerRegistry::CreatorRegistry& LayerRegistry::Registry()
		{
			static auto registry = Ptr<CreatorRegistry>(new CreatorRegistry());
			return *registry;
		}

		void LayerRegistry::AddCreator(const std::string& type, Creator creator)
		{
			CreatorRegistry& registry = Registry();
			CHECK_EQ(registry.count(type), 0) << "Layer type " << type << " already registered.";
			registry[type] = creator;
		}

		Ptr<Layer> LayerRegistry::CreateLayer(const std::string& type)
		{
			CreatorRegistry& registry = Registry();
			CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
				<< " (known types: " << GetLayerTypeListString() << ")";
			return registry[type]();
		}

		std::vector<std::string> LayerRegistry::GetLayerTypeListString()
		{
			CreatorRegistry& registry = Registry();
			std::vector<std::string> layer_types;
			for (const auto& [type, creator] : registry)
			{
				layer_types.push_back(type);
			}
			return layer_types;
		}

		LayerRegisterer::LayerRegisterer(const std::string& type, const LayerRegistry::Creator& creator)
		{
			LayerRegistry::AddCreator(type, creator);
		}
	}
}