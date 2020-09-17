#include "core/core.hpp"

//#include <stdarg.h>
#include <regex>
#include <Windows.h>

namespace chaos
{
	std::vector<std::string> Split(const std::string& data, const std::string& delimiter)
	{
		std::regex regex{ delimiter };
		return std::vector<std::string> {
			std::sregex_token_iterator(data.begin(), data.end(), regex, -1),
				std::sregex_token_iterator()
		};
	}

	int chaos_vsnprintf(char* buf, int len, const char* fmt, va_list args)
	{
		if (len <= 0) return len == 0 ? 1024 : -1;
		int res = _vsnprintf_s(buf, len, _TRUNCATE, fmt, args);
		// ensure null terminating on VS
		if (res >= 0 && res < len)
		{
			buf[res] = 0;
			return res;
		}
		else
		{
			buf[len - 1] = 0; // truncate happened
			return res >= len ? res : (len * 2);
		}
	}

	std::string Format(const char* fmt, ...)
	{
		AutoBuffer<char, 1024> buf;

		for (; ; )
		{
			va_list va;
			va_start(va, fmt);
			int bsize = static_cast<int>(buf.size());
			int len = chaos_vsnprintf(buf.data(), bsize, fmt, va);
			va_end(va);

			//CHECK(len >= 0) << "check format string for errors";
			if (len >= bsize)
			{
				buf.Resize(len + 1LL);
				continue;
			}
			buf[bsize - 1LL] = 0;
			return std::string(buf.data(), len);
		}
	}

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