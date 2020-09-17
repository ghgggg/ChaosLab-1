#pragma once

#ifdef CHAOS_EXPORT
#define CHAOS_API __declspec(dllexport)
#else
#define CHAOS_API __declspec(dllimport)
#endif

namespace chaos
{
	using uchar = unsigned char;
	using int8 = __int8;
	using uint8 = unsigned __int8;  // unsigned int8
	using int16 = __int16;
	using uint16 = unsigned __int16;
	using uint = unsigned __int32; // unsigned int32
	using int64 = __int64;
	using uint64 = unsigned __int64;
}