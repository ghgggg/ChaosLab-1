#include "core/vulkan/gpu.hpp"

namespace chaos
{
	static std::mutex g_instance_lock;

	class VulkanInstanceHolder
	{
	public:
		VulkanInstanceHolder() { instance = 0; }
		~VulkanInstanceHolder() { DestroyGPUInstance(); }

		operator VkInstance() { return instance; }
		VkInstance instance;
	};

	static VulkanInstanceHolder g_instance;

	static int g_gpu_count = 0;
	static int g_default_gpu_index = -1;

	static constexpr int MAX_GPU_COUNT = 8;
	static GPUInfo g_gpu_infos[MAX_GPU_COUNT];



	int support_VK_KHR_external_memory_capabilities = 0;
	int support_VK_KHR_get_physical_device_properties2 = 0;
	int support_VK_KHR_get_surface_capabilities2 = 0;
	int support_VK_KHR_surface = 0;
	int support_VK_EXT_debug_utils = 0;


	// VK_KHR_external_memory_capabilities
	PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR vkGetPhysicalDeviceExternalBufferPropertiesKHR = 0;

	// VK_KHR_get_physical_device_properties2
	PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR = 0;
	PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceFormatProperties2KHR vkGetPhysicalDeviceFormatProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceImageFormatProperties2KHR vkGetPhysicalDeviceImageFormatProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR = 0;

	// VK_KHR_get_surface_capabilities2
	PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR = 0;
	PFN_vkGetPhysicalDeviceSurfaceFormats2KHR vkGetPhysicalDeviceSurfaceFormats2KHR = 0;

	// VK_KHR_surface
	PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR = 0;
	PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR = 0;
	PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR = 0;
	PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR = 0;
	PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR = 0;


	static void InitInstanceExtension()
	{
		if (support_VK_KHR_external_memory_capabilities)
		{
			vkGetPhysicalDeviceExternalBufferPropertiesKHR = (PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");
		}

		if (support_VK_KHR_get_physical_device_properties2)
		{
			vkGetPhysicalDeviceFeatures2KHR = (PFN_vkGetPhysicalDeviceFeatures2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceFeatures2KHR");
			vkGetPhysicalDeviceProperties2KHR = (PFN_vkGetPhysicalDeviceProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceProperties2KHR");
			vkGetPhysicalDeviceFormatProperties2KHR = (PFN_vkGetPhysicalDeviceFormatProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceFormatProperties2KHR");
			vkGetPhysicalDeviceImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceImageFormatProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceImageFormatProperties2KHR");
			vkGetPhysicalDeviceQueueFamilyProperties2KHR = (PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceQueueFamilyProperties2KHR");
			vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceMemoryProperties2KHR");
			vkGetPhysicalDeviceSparseImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR");
		}

		if (support_VK_KHR_get_surface_capabilities2)
		{
			vkGetPhysicalDeviceSurfaceCapabilities2KHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR");
			vkGetPhysicalDeviceSurfaceFormats2KHR = (PFN_vkGetPhysicalDeviceSurfaceFormats2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceFormats2KHR");
		}

		if (support_VK_KHR_surface)
		{
			vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)vkGetInstanceProcAddr(g_instance, "vkDestroySurfaceKHR");
			vkGetPhysicalDeviceSurfaceSupportKHR = (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceSupportKHR");
			vkGetPhysicalDeviceSurfaceCapabilitiesKHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
			vkGetPhysicalDeviceSurfaceFormatsKHR = (PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceFormatsKHR");
			vkGetPhysicalDeviceSurfacePresentModesKHR = (PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfacePresentModesKHR");
		}
	}



	static uint32_t FindDeviceComputeQueue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
	{
		// first try, compute only queue
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
				&& !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
			{
				return i;
			}
		}

		// second try, any queue with compute and graphics
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
				&& (queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
			{
				return i;
			}
		}

		// third try, any queue with compute
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if (queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				return i;
			}
		}

		return -1;
	}

	static uint32_t FindDeviceGraphicsQueue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
	{
		// first try, graphics only queue
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if ((queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				&& !(queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT))
			{
				return i;
			}
		}

		// second try, any queue with graphics and compute
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if ((queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				&& (queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT))
			{
				return i;
			}
		}

		// third try, any queue with graphics
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if (queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				return i;
			}
		}

		return -1;
	}

	static uint32_t FindDeviceTransferQueue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
	{
		// first try, transfer only queue
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if ((queueFamilyProperty.queueFlags & VK_QUEUE_TRANSFER_BIT)
				&& !(queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
				&& !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
			{
				return i;
			}
		}

		// second try, any queue with transfer
		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
		{
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

			if (queueFamilyProperty.queueFlags & VK_QUEUE_TRANSFER_BIT)
			{
				return i;
			}
		}

		// third try, use compute queue
		uint32_t compute_queue_index = FindDeviceComputeQueue(queueFamilyProperties);
		if (compute_queue_index != (uint32_t)-1)
		{
			return compute_queue_index;
		}

		// fourth try, use graphics queue
		uint32_t graphics_queue_index = FindDeviceGraphicsQueue(queueFamilyProperties);
		if (graphics_queue_index != (uint32_t)-1)
		{
			return graphics_queue_index;
		}

		//     NCNN_LOGE("no transfer queue");
		return -1;
	}

	static int FindDefaultVulkanDeviceIndex()
	{
		// first try, discrete gpu
		for (int i = 0; i < g_gpu_count; i++)
		{
			if (g_gpu_infos[i].type == 0)
				return i;
		}

		// second try, integrated gpu
		for (int i = 0; i < g_gpu_count; i++)
		{
			if (g_gpu_infos[i].type == 1)
				return i;
		}

		// third try, any probed device
		if (g_gpu_count > 0)
			return 0;

		LOG(INFO) << "no vulkan device";
		return -1;
	}

	void CreateGPUInstance()
	{
		std::lock_guard lock(g_instance_lock);

		if ((VkInstance)g_instance != 0) return;

		VkResult ret;
		
		std::vector<const char*> enabledExtensions;

		uint32_t instanceExtensionPropertyCount;
		ret = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionPropertyCount, NULL);
		CHECK_EQ(ret, VK_SUCCESS) << Format("vkEnumerateInstanceExtensionProperties failed %d", ret);

		std::vector<VkExtensionProperties> instanceExtensionProperties(instanceExtensionPropertyCount);
		ret = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionPropertyCount, instanceExtensionProperties.data());
		CHECK_EQ(ret, VK_SUCCESS) << Format("vkEnumerateInstanceExtensionProperties failed %d", ret);

		support_VK_KHR_get_physical_device_properties2 = 0;
		support_VK_KHR_get_surface_capabilities2 = 0;
		support_VK_KHR_surface = 0;
		support_VK_EXT_debug_utils = 0;

		for (uint32_t j = 0; j < instanceExtensionPropertyCount; j++)
		{
			const VkExtensionProperties& exp = instanceExtensionProperties[j];

			if (strcmp(exp.extensionName, "VK_KHR_external_memory_capabilities") == 0)
				support_VK_KHR_external_memory_capabilities = exp.specVersion;
			else if (strcmp(exp.extensionName, "VK_KHR_get_physical_device_properties2") == 0)
				support_VK_KHR_get_physical_device_properties2 = exp.specVersion;
			else if (strcmp(exp.extensionName, "VK_KHR_get_surface_capabilities2") == 0)
				support_VK_KHR_get_surface_capabilities2 = exp.specVersion;
			else if (strcmp(exp.extensionName, "VK_KHR_surface") == 0)
				support_VK_KHR_surface = exp.specVersion;
			else if (strcmp(exp.extensionName, "VK_EXT_debug_utils") == 0)
				support_VK_EXT_debug_utils = exp.specVersion;
		}

		if (support_VK_KHR_external_memory_capabilities)
			enabledExtensions.push_back("VK_KHR_external_memory_capabilities");
		if (support_VK_KHR_get_physical_device_properties2)
			enabledExtensions.push_back("VK_KHR_get_physical_device_properties2");
		if (support_VK_KHR_get_surface_capabilities2)
			enabledExtensions.push_back("VK_KHR_get_surface_capabilities2");
		if (support_VK_KHR_surface)
			enabledExtensions.push_back("VK_KHR_surface");

		VkApplicationInfo applicationInfo;
		applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		applicationInfo.pNext = 0;
		applicationInfo.pApplicationName = "ChaosCV";
		applicationInfo.applicationVersion = 0;
		applicationInfo.pEngineName = "ChaosCV";
		applicationInfo.engineVersion = 20200727;
		applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

		VkInstanceCreateInfo instanceCreateInfo;
		instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.pNext = 0;
		instanceCreateInfo.flags = 0;
		instanceCreateInfo.pApplicationInfo = &applicationInfo;
		instanceCreateInfo.enabledLayerCount = 0; // enabledLayers.size();
		instanceCreateInfo.ppEnabledLayerNames = nullptr; // enabledLayers.data();
		instanceCreateInfo.enabledExtensionCount = (uint32_t)enabledExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();

		VkInstance instance = 0;
		ret = vkCreateInstance(&instanceCreateInfo, 0, &instance);
		CHECK_EQ(ret, VK_SUCCESS) << Format("vkCreateInstance failed %d", ret);

		g_instance.instance = instance;

		InitInstanceExtension();

		uint32_t physicalDeviceCount = 0;
		ret = vkEnumeratePhysicalDevices(g_instance, &physicalDeviceCount, 0);
		CHECK_EQ(ret, VK_SUCCESS) << Format("vkEnumeratePhysicalDevices failed %d", ret);

		if (physicalDeviceCount > MAX_GPU_COUNT)
			physicalDeviceCount = MAX_GPU_COUNT;

		std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);

		ret = vkEnumeratePhysicalDevices(g_instance, &physicalDeviceCount, physicalDevices.data());
		CHECK_EQ(ret, VK_SUCCESS) << Format("vkEnumeratePhysicalDevices failed %d", ret);

		// find proper device and queue
		int gpu_info_index = 0;
		for (uint32_t i = 0; i < physicalDeviceCount; i++)
		{
			const VkPhysicalDevice& physicalDevice = physicalDevices[i];
			GPUInfo& gpu_info = g_gpu_infos[gpu_info_index];

			// device type
			VkPhysicalDeviceProperties physicalDeviceProperties;
			vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

			gpu_info.bug_storage_buffer_no_l1 = false;
			gpu_info.bug_layout_binding_id_alias = false;
			gpu_info.bug_corrupted_online_pipeline_cache = false;
			gpu_info.bug_implicit_fp16_arithmetic = false;

			if (physicalDeviceProperties.vendorID == 0x5143 
				&& physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 0, 49))
			{
				// qcom adreno with old buggy driver cannot handle binding id alias
				gpu_info.bug_layout_binding_id_alias = true;
			}

			if (physicalDeviceProperties.vendorID == 0x5143 
				&& physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 0, 66))
			{
				// qcom adreno with old buggy driver cannot share created pipeline properly
				gpu_info.bug_corrupted_online_pipeline_cache = true;
			}

			if (physicalDeviceProperties.vendorID == 0x5143 
				&& !(physicalDeviceProperties.deviceID == 0x6040001 
					|| physicalDeviceProperties.deviceID == 0x6050002))
			{
				// NOTE but qcom855/qcom855plus/qcom865 are known exceptions
				// qcom adreno storage buffer without L1 cache
				gpu_info.bug_storage_buffer_no_l1 = true;
			}

			if (physicalDeviceProperties.vendorID == 0x13b5
				&& (physicalDeviceProperties.deviceID == 0x7500001
					|| physicalDeviceProperties.deviceID == 0x8602000
					|| physicalDeviceProperties.deviceID == 0x8800020))
			{
				// these arm mali midgard era driver cannot handle binding id alias
				gpu_info.bug_layout_binding_id_alias = true;
			}

			if (physicalDeviceProperties.vendorID == 0x13b5
				&& (physicalDeviceProperties.deviceID == 0x7500001
					|| physicalDeviceProperties.deviceID == 0x8602000
					|| physicalDeviceProperties.deviceID == 0x8800020
					|| physicalDeviceProperties.deviceID == 0x70901010
					|| physicalDeviceProperties.deviceID == 0x74021000
					|| physicalDeviceProperties.deviceID == 0x60a00002
					|| physicalDeviceProperties.deviceID == 0x62210001))
			{
				// NOTE rk3288/rk3399/t880/g51/g52/g71/g72
				// however, g76/g77 has explicit fp16 arithmetic
				// arm mali driver accept spirv with fp16 arithmetic
				gpu_info.bug_implicit_fp16_arithmetic = true;
			}

			if (physicalDeviceProperties.vendorID == 0x5143
				&& (physicalDeviceProperties.deviceID == 0x6030001
					|| physicalDeviceProperties.deviceID == 0x6040001
					|| physicalDeviceProperties.deviceID == 0x6050002))
			{
				// TODO enable devices other than qcom845/qcom855/qcom855plus/qcom865
				// qcom adreno driver accept spirv with fp16 arithmetic
				gpu_info.bug_implicit_fp16_arithmetic = true;
			}

			gpu_info.physical_device = physicalDevice;

			// info
			gpu_info.api_version = physicalDeviceProperties.apiVersion;
			gpu_info.driver_version = physicalDeviceProperties.driverVersion;
			gpu_info.vendor_id = physicalDeviceProperties.vendorID;
			gpu_info.device_id = physicalDeviceProperties.deviceID;
			memcpy(gpu_info.pipeline_cache_uuid, physicalDeviceProperties.pipelineCacheUUID, VK_UUID_SIZE);

			if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
				gpu_info.type = 0;
			else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
				gpu_info.type = 1;
			else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
				gpu_info.type = 2;
			else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
				gpu_info.type = 3;
			else
				gpu_info.type = -1;

			// device capability
			gpu_info.max_shared_memory_size = physicalDeviceProperties.limits.maxComputeSharedMemorySize;

			gpu_info.max_workgroup_count[0] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];
			gpu_info.max_workgroup_count[1] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[1];
			gpu_info.max_workgroup_count[2] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[2];

			gpu_info.max_workgroup_invocations = physicalDeviceProperties.limits.maxComputeWorkGroupInvocations;

			gpu_info.max_workgroup_size[0] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[0];
			gpu_info.max_workgroup_size[1] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[1];
			gpu_info.max_workgroup_size[2] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[2];

			gpu_info.memory_map_alignment = physicalDeviceProperties.limits.minMemoryMapAlignment;
			gpu_info.buffer_offset_alignment = physicalDeviceProperties.limits.minStorageBufferOffsetAlignment;
			gpu_info.non_coherent_atom_size = physicalDeviceProperties.limits.nonCoherentAtomSize;
			gpu_info.buffer_image_granularity = physicalDeviceProperties.limits.bufferImageGranularity;
			gpu_info.max_image_dimension_1d = physicalDeviceProperties.limits.maxImageDimension1D;
			gpu_info.max_image_dimension_2d = physicalDeviceProperties.limits.maxImageDimension2D;
			gpu_info.max_image_dimension_3d = physicalDeviceProperties.limits.maxImageDimension3D;

			gpu_info.timestamp_period = physicalDeviceProperties.limits.timestampPeriod;

			// find compute queue
			uint32_t queueFamilyPropertiesCount;
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);

			std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertiesCount);
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties.data());

			gpu_info.compute_queue_family_index = FindDeviceComputeQueue(queueFamilyProperties);
			gpu_info.graphics_queue_family_index = FindDeviceGraphicsQueue(queueFamilyProperties);
			gpu_info.transfer_queue_family_index = FindDeviceTransferQueue(queueFamilyProperties);

			gpu_info.compute_queue_count = queueFamilyProperties[gpu_info.compute_queue_family_index].queueCount;
			gpu_info.graphics_queue_count = queueFamilyProperties[gpu_info.graphics_queue_family_index].queueCount;
			gpu_info.transfer_queue_count = queueFamilyProperties[gpu_info.transfer_queue_family_index].queueCount;

			gpu_info.unified_compute_transfer_queue = gpu_info.compute_queue_family_index == gpu_info.transfer_queue_family_index;

			// cache memory properties
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &gpu_info.physicalDeviceMemoryProperties);

			// get device extension
			uint32_t deviceExtensionPropertyCount = 0;
			ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, NULL);
			CHECK_EQ(ret, VK_SUCCESS) << Format("vkEnumerateDeviceExtensionProperties failed %d", ret);

			std::vector<VkExtensionProperties> deviceExtensionProperties(deviceExtensionPropertyCount);
			ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, deviceExtensionProperties.data());
			CHECK_EQ(ret, VK_SUCCESS) << Format("vkEnumerateDeviceExtensionProperties failed %d", ret);

			// extension capability
			gpu_info.support_VK_KHR_8bit_storage = 0;
			gpu_info.support_VK_KHR_16bit_storage = 0;
			gpu_info.support_VK_KHR_bind_memory2 = 0;
			gpu_info.support_VK_KHR_dedicated_allocation = 0;
			gpu_info.support_VK_KHR_descriptor_update_template = 0;
			gpu_info.support_VK_KHR_external_memory = 0;
			gpu_info.support_VK_KHR_get_memory_requirements2 = 0;
			gpu_info.support_VK_KHR_maintenance1 = 0;
			gpu_info.support_VK_KHR_push_descriptor = 0;
			gpu_info.support_VK_KHR_sampler_ycbcr_conversion = 0;
			gpu_info.support_VK_KHR_shader_float16_int8 = 0;
			gpu_info.support_VK_KHR_shader_float_controls = 0;
			gpu_info.support_VK_KHR_storage_buffer_storage_class = 0;
			gpu_info.support_VK_KHR_swapchain = 0;
			gpu_info.support_VK_EXT_queue_family_foreign = 0;

			for (uint32_t j = 0; j < deviceExtensionPropertyCount; j++)
			{
				const VkExtensionProperties& exp = deviceExtensionProperties[j];

				if (strcmp(exp.extensionName, "VK_KHR_8bit_storage") == 0)
					gpu_info.support_VK_KHR_8bit_storage = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_16bit_storage") == 0)
					gpu_info.support_VK_KHR_16bit_storage = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_bind_memory2") == 0)
					gpu_info.support_VK_KHR_bind_memory2 = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_dedicated_allocation") == 0)
					gpu_info.support_VK_KHR_dedicated_allocation = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_descriptor_update_template") == 0)
					gpu_info.support_VK_KHR_descriptor_update_template = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_external_memory") == 0)
					gpu_info.support_VK_KHR_external_memory = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_get_memory_requirements2") == 0)
					gpu_info.support_VK_KHR_get_memory_requirements2 = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_maintenance1") == 0)
					gpu_info.support_VK_KHR_maintenance1 = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_push_descriptor") == 0)
					gpu_info.support_VK_KHR_push_descriptor = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_sampler_ycbcr_conversion") == 0)
					gpu_info.support_VK_KHR_sampler_ycbcr_conversion = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_shader_float16_int8") == 0)
					gpu_info.support_VK_KHR_shader_float16_int8 = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_shader_float_controls") == 0)
					gpu_info.support_VK_KHR_shader_float_controls = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_storage_buffer_storage_class") == 0)
					gpu_info.support_VK_KHR_storage_buffer_storage_class = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_swapchain") == 0)
					gpu_info.support_VK_KHR_swapchain = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_EXT_memory_budget") == 0)
					gpu_info.support_VK_EXT_memory_budget = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_EXT_queue_family_foreign") == 0)
					gpu_info.support_VK_EXT_queue_family_foreign = exp.specVersion;
			}

			// check features
			gpu_info.support_fp16_packed = true;
			gpu_info.support_fp16_storage = false;
			gpu_info.support_fp16_arithmetic = false;
			gpu_info.support_int8_storage = false;
			gpu_info.support_int8_arithmetic = false;
			gpu_info.support_ycbcr_conversion = false;
			if (support_VK_KHR_get_physical_device_properties2)
			{
				void* queryExtensionFeatures = 0;

				// query int8 storage
				VkPhysicalDevice8BitStorageFeaturesKHR query8BitStorageFeatures;
				query8BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
				query8BitStorageFeatures.pNext = 0;
				if (gpu_info.support_VK_KHR_8bit_storage)
				{
					query8BitStorageFeatures.pNext = queryExtensionFeatures;
					queryExtensionFeatures = &query8BitStorageFeatures;
				}

				// query fp16/int16 storage
				VkPhysicalDevice16BitStorageFeaturesKHR query16BitStorageFeatures;
				query16BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
				query16BitStorageFeatures.pNext = 0;
				if (gpu_info.support_VK_KHR_16bit_storage)
				{
					query16BitStorageFeatures.pNext = queryExtensionFeatures;
					queryExtensionFeatures = &query16BitStorageFeatures;
				}

				// query fp16/int8 arithmetic
				VkPhysicalDeviceFloat16Int8FeaturesKHR queryFloat16Int8Features;
				queryFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
				queryFloat16Int8Features.pNext = 0;
				if (gpu_info.support_VK_KHR_shader_float16_int8)
				{
					queryFloat16Int8Features.pNext = queryExtensionFeatures;
					queryExtensionFeatures = &queryFloat16Int8Features;
				}

				// query ycbcr_conversion
				VkPhysicalDeviceSamplerYcbcrConversionFeaturesKHR querySamplerYcbcrConversionFeatures;
				querySamplerYcbcrConversionFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES_KHR;
				querySamplerYcbcrConversionFeatures.pNext = 0;
				if (gpu_info.support_VK_KHR_sampler_ycbcr_conversion)
				{
					querySamplerYcbcrConversionFeatures.pNext = queryExtensionFeatures;
					queryExtensionFeatures = &querySamplerYcbcrConversionFeatures;
				}

				VkPhysicalDeviceFeatures2KHR queryFeatures;
				queryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR,
					queryFeatures.pNext = queryExtensionFeatures;

				vkGetPhysicalDeviceFeatures2KHR(physicalDevice, &queryFeatures);

				if (gpu_info.support_VK_KHR_8bit_storage)
				{
					gpu_info.support_int8_storage = query8BitStorageFeatures.storageBuffer8BitAccess && query8BitStorageFeatures.uniformAndStorageBuffer8BitAccess;
				}
				if (gpu_info.support_VK_KHR_16bit_storage && queryFeatures.features.shaderStorageImageExtendedFormats)
				{
					// shaderStorageImageExtendedFormats enables r16f format in storage image
					gpu_info.support_fp16_storage = query16BitStorageFeatures.storageBuffer16BitAccess && query16BitStorageFeatures.uniformAndStorageBuffer16BitAccess;
				}
				if (gpu_info.support_VK_KHR_shader_float16_int8)
				{
					gpu_info.support_fp16_arithmetic = queryFloat16Int8Features.shaderFloat16;
					gpu_info.support_int8_arithmetic = queryFloat16Int8Features.shaderInt8;
				}
				if (gpu_info.support_VK_KHR_sampler_ycbcr_conversion)
				{
					gpu_info.support_ycbcr_conversion = querySamplerYcbcrConversionFeatures.samplerYcbcrConversion;
				}
			}
			else
			{
				// // TODO
				// VkPhysicalDeviceFeatures features;
				// vkGetPhysicalDeviceFeatures(physicalDevice, &features);
			}

			if (physicalDeviceProperties.vendorID == 0x13b5)
			{
				// the 16bit_storage implementation of arm mali driver is buggy :[
				gpu_info.support_fp16_storage = false;
			}

			if (physicalDeviceProperties.vendorID == 0x10002 && physicalDeviceProperties.deviceID == 0x70006214 && physicalDeviceProperties.apiVersion == VK_MAKE_VERSION(1, 1, 82))
			{
				// the 16bit_storage implementation of vivante gc1700 driver is buggy :[
				gpu_info.support_fp16_storage = false;
			}

			if (gpu_info.bug_implicit_fp16_arithmetic)
			{
				// force capability on as long as the driver accept spirv with fp16 arithmetic :D
				gpu_info.support_fp16_arithmetic = true;
			}

			LOG(INFO) << Format("[%u %s]  queueC=%u[%u]  queueG=%u[%u]  queueT=%u[%u]", i, physicalDeviceProperties.deviceName,
				gpu_info.compute_queue_family_index, gpu_info.compute_queue_count,
				gpu_info.graphics_queue_family_index, gpu_info.graphics_queue_count,
				gpu_info.transfer_queue_family_index, gpu_info.transfer_queue_count);

			LOG(INFO) << Format("[%u %s]  bugsbn1=%d  buglbia=%d  bugcopc=%d  bugihfa=%d", i, physicalDeviceProperties.deviceName,
				gpu_info.bug_storage_buffer_no_l1, gpu_info.bug_layout_binding_id_alias, gpu_info.bug_corrupted_online_pipeline_cache, gpu_info.bug_implicit_fp16_arithmetic);

			LOG(INFO) << Format("[%u %s]  fp16p=%d  fp16s=%d  fp16a=%d  int8s=%d  int8a=%d", i, physicalDeviceProperties.deviceName,
				gpu_info.support_fp16_packed, gpu_info.support_fp16_storage, gpu_info.support_fp16_arithmetic,
				gpu_info.support_int8_storage, gpu_info.support_int8_arithmetic);

			gpu_info_index++;
		}

		g_gpu_count = gpu_info_index;

		// the default gpu device
		g_default_gpu_index = FindDefaultVulkanDeviceIndex();
	}

	void DestroyGPUInstance()
	{
		std::lock_guard lock(g_instance_lock);

		if ((VkInstance)g_instance == 0) return;

		for (int i = 0; i < MAX_GPU_COUNT; i++)
		{
			//delete g_default_vkdev[i];
			//g_default_vkdev[i] = 0;
		}

		vkDestroyInstance(g_instance, 0);
		g_instance.instance = 0;
	}
}