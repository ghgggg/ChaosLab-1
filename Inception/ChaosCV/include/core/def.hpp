#pragma once

#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <mutex>

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

	enum class LogSeverity
	{
		INFO,
		WARNING,
		ERROR,
		FATAL,
	};
}

#define CHAOS_PREDICT_BRANCH_NOT_TAKEN(x) x

#define LOG(severity) chaos::LogMessage(__FILE__, __LINE__, chaos::LogSeverity::##severity).stream()

#define LOG_IF(severity, condition) \
  !(condition) ? (void) 0 : chaos::LogMessageVoidify() & LOG(severity)

// CHECK dies with a fatal error if condition is not true.
#define CHECK(condition) LOG_IF(FATAL, CHAOS_PREDICT_BRANCH_NOT_TAKEN(!(condition))) << "Check failed: " #condition ". "

#define CHECK_EQ(val1, val2) CHECK(val1 == val2)
#define CHECK_NE(val1, val2) CHECK(val1 != val2)
#define CHECK_LE(val1, val2) CHECK(val1 <= val2)
#define CHECK_LT(val1, val2) CHECK(val1 <  val2)
#define CHECK_GE(val1, val2) CHECK(val1 >= val2)
#define CHECK_GT(val1, val2) CHECK(val1 >  val2)

#define CHECK_NOTNULL(val) chaos::CheckNotNull(__FILE__, __LINE__, "'" #val "' must be not NULL", (val))