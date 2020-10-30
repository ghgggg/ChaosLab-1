#pragma once

#include "core/allocator.hpp"

namespace chaos
{
	class VkAllocator;
	class PipelineCache;
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

			// blob memory allocator
			VkAllocator* blob_vkallocator = nullptr;

			// workspace memory allocator
			VkAllocator* workspace_vkallocator = nullptr;

			// staging memory allocator
			VkAllocator* staging_vkallocator = nullptr;

			// pipeline cache
			PipelineCache* pipeline_cache;

			// enable quantized int8 inference
			// use low-precision int8 path for quantized model
			// changes should be applied before loading network structure and weight
			// enabled by default
			bool use_int8_inference;

			// enable vulkan compute
			bool use_vulkan_compute;

			// enable options for gpu inference
			bool use_fp16_packed;
			bool use_fp16_storage;
			bool use_fp16_arithmetic;
			bool use_int8_storage;
			bool use_int8_arithmetic;
		};
	}
}