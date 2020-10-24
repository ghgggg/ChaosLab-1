#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	namespace dnn
	{
		class Executor;
		class CHAOS_API Net
		{
		public:
			virtual ~Net() = default;

			virtual Ptr<Executor> BindExecutor() const = 0;

			virtual void AddLayer(const std::string& type) {};

			static Ptr<Net> CreateNet();
			static Ptr<Net> LoadNet();
			//static Ptr<Net> LoadChaosNet();
			//static Ptr<Net> LoadMxNet();
			//static Ptr<Net> LoadCaffeNet();
			//static Ptr<Net> LoadVINO();
		};

		class CHAOS_API Executor
		{
		public:
			virtual ~Executor() = default;

			virtual void SetLayerData(const std::string& name, const Tensor& data) const = 0;
			virtual void GetLayerData(const std::string& name, Tensor& data) const = 0;
			virtual void Forward() const = 0;
		};
	}
}