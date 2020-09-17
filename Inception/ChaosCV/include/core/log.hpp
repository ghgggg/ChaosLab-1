#pragma once

#include "def.hpp"

#include <sstream>

namespace chaos
{
	class CHAOS_API LogMessage
	{
	public:
		LogMessage(const char* file, int line, const LogSeverity& severity);
		LogMessage(const char* file, int line, const char* message, const LogSeverity& severity);
		~LogMessage();

		std::ostream& stream();
	private:
		void Flush();

		std::stringstream message_data; // use stringstream to replace LogMessageData
		LogSeverity severity;
	};

	class CHAOS_API LogMessageVoidify
	{
	public:
		LogMessageVoidify() {}
		// This has to be an operator with a precedence lower than << but
		// higher than ?:
		void operator&(std::ostream&) {}
	};

	template <class Type>
	Type CheckNotNull(const char* file, int line, const char* message, Type&& t)
	{
		if (t == nullptr) LogMessage(file, line, message, LogSeverity::FATAL);
		return std::forward<Type>(t);
	}
}