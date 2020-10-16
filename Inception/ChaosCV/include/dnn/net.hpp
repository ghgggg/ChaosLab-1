#pragma once

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Net
		{
		public:
			virtual ~Net() = default;

			static Ptr<Net> Load();
		};

		class CHAOS_API Executor
		{
		public:
			virtual ~Executor() = default;

			void SetLayerData(const std::string& name, const Tensor& data) const;
			void GetLayerData(const std::string& name, Tensor& data) const;
			void Forward() const;


		};
	}
}