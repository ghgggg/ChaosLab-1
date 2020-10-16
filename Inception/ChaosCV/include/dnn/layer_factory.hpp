#pragma once

#include "layer.hpp"

#include <map>
#include <functional>

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API LayerRegistry
		{
		public:
			using Creator = std::function<Ptr<Layer>()>;
			using CreatorRegistry = std::map<std::string, Creator>;

			static CreatorRegistry& Registry();

			static void AddCreator(const std::string& type, Creator creator);

			static Ptr<Layer> CreateLayer(const std::string& type);

			static std::vector<std::string> GetLayerTypeListString();

			// Layer registry should never be instantiated - everything is done with its
			// static variables.
			LayerRegistry() = delete;
		};

		class CHAOS_API LayerRegisterer
		{
		public:
			LayerRegisterer(const std::string& type, const LayerRegistry::Creator& creator);
		};
	}
}

#define REGISTER_LAYER(name, type) \
chaos::Ptr<chaos::dnn::Layer> Create##type() { return chaos::Ptr<chaos::dnn::Layer>(new type()); } \
static chaos::dnn::LayerRegisterer creator_##type(name, Create##type);
