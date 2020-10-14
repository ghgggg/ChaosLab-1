#include "core/core.hpp"

namespace chaos
{
	//static int log_level;
	static std::mutex mtx;

	std::ostream& operator<<(std::ostream& stream, const LogSeverity& severity)
	{
		switch (severity)
		{
		case LogSeverity::INFO:
			return stream << "INFO";
		case LogSeverity::WARNING:
			return stream << "WARNING";
		case LogSeverity::ERROR:
			return stream << "ERROR";
		case LogSeverity::FATAL:
			return stream << "FATAL";
		default:
			return stream; // never reachable
		}
	}

	bool operator>=(const LogSeverity& severity, int level)
	{
		return static_cast<int>(severity) >= level;
	}

	LogMessage::LogMessage(const char* file, int line, const LogSeverity& severity)
		: severity(severity)
	{
		//if (severity >= log_level)
		{
			File _file(file);

			// Get time stamp
			time_t time_stamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			tm time;
			localtime_s(&time, &time_stamp);

			// Head likes "[INFO 2018-05-21 17:31:04 xx.cpp:21]"
			message_data << "[" << severity
				<< Format(" %04d-%02d-%02d %02d:%02d:%02d ", time.tm_year + 1990, time.tm_mon + 1, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec)
				<< _file.name << "." << _file.type << ":" << line << "] ";
		}
	}

	LogMessage::LogMessage(const char* file, int line, const char* message, const LogSeverity& severity) : LogMessage(file, line, severity)
	{
		/*if (severity >= log_level)*/ message_data << message;
	}


	LogMessage::~LogMessage()
	{
		Flush();
		// if FATAL, then abort
		if (severity == LogSeverity::FATAL) abort();
	}

	std::ostream& chaos::LogMessage::stream()
	{
		return message_data;
	}

	void LogMessage::Flush()
	{
		mtx.lock();
		//if (severity >= log_level)
		{
			std::string message = message_data.str();

			//SetConsoleTextColor(ToColor(severity));
			std::cout << message << std::endl;
			//SetConsoleTextColor(0x07); // Reset to 0x07
		}
		mtx.unlock();
	}
}