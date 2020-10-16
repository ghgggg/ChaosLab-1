#pragma once

#include "core/allocator.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Option
		{
		public:
			Option() = default;

			// ligth mode
			// intermediate blob will be recycled when enabled
			// disable by default
			bool light_mode = false;

			// blob memory allocator
			Allocator* blob_allocator = nullptr;
			// workspace allocator
			Allocator* workspace_allocator = nullptr;
		};
	}
}