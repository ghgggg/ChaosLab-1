#include "core/core.hpp"

#include <stdarg.h>
#include <Windows.h>

namespace chaos
{
	int64 GetTickCount()
	{
		LARGE_INTEGER counter;
		QueryPerformanceCounter(&counter);
		return static_cast<int64>(counter.QuadPart);
	}

	double GetTickFrequency()
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		return static_cast<double>(freq.QuadPart);
	}
}