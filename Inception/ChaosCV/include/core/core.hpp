#pragma once

#include "def.hpp"

namespace chaos
{
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