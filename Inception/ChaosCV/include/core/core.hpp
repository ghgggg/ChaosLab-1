#pragma once

#include "def.hpp"
#include "log.hpp"
#include "file.hpp"
#include "allocator.hpp"

#include <string>

namespace chaos
{
	template<class Type>
	using Ptr = std::shared_ptr<Type>;

	/// <summary>Split the string data by delimiter</summary>
	CHAOS_API std::vector<std::string> Split(const std::string& data, const std::string& delimiter);

	/// <summary>
	/// <para>Returns a text string formatted using the printf-like expression</para>
	/// <para>The function acts like sprintf but forms and returns an STL string. It can be used to form an error</para>
	/// <para>message in the Exception constructor.</para>
	/// </summary>
	/// <param name="fmt">@param fmt printf-compatible formatting specifiers.</param>
	CHAOS_API std::string Format(const char* fmt, ...);

	/// <summary>
	/// <para>The function returns the number of ticks after the certain event(for example, when the machine was</para>
	/// <para>turned on).It can be used to initialize RNG or to measure a function execution time by reading the</para>
	/// <para>tick count beforeand after the function call.</para>
	/// </summary>
	CHAOS_API int64 GetTickCount();
	/// <summary>
	/// <para>The function returns the number of ticks per second.That is, the following code computes the</para>
	/// <para>execution time in seconds</para>
	/// </summary>
	CHAOS_API double GetTickFrequency();
}