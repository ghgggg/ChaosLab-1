#include "core/core.hpp"

#include <regex>
#include <io.h>
#include <Windows.h>

namespace chaos
{
	File::File(const std::string& _file)
	{
		if (_file.empty())
			return;

		buffer.Resize(_file.size() + 1);
		memset(buffer.data(), 0, _file.size() + 1);
		memcpy(buffer.data(), _file.data(), _file.size());

		// replace all '/' to '\\'
		for (int i = 0; i < buffer.size(); i++)
		{
			if (buffer[i] == '/') buffer[i] = '\\';
		}

		std::string_view file(buffer.data());

		spos = file.find_last_of('\\') + 1; // last slash pos

		auto remain = std::string_view(buffer.data() + spos);
		// ppos == 0 means that the string does not contain '.' 
		ppos = remain.find_last_of('.') + 1;

		std::string_view _name = 0 == ppos ? remain : remain.substr(0, ppos - 1);

		// if the first char is '.', the name will be empty
		CHECK(!_name.empty());
		auto valid = std::regex_match(std::string(_name), std::regex("[^\\|\\\\/:\\*\\?\"<>]+"));
		CHECK(valid) << "file name can not contain |\\/:*?\"<>";
	}
	File::File(const char* file) : File(std::string(file)) {}

	std::string_view File::GetPath() const
	{
		return std::string_view(buffer.data(), spos);
	}
	std::string_view File::GetName() const
	{
		return 0 == ppos ? std::string_view(buffer.data() + spos) : std::string_view(buffer.data() + spos, ppos - 1);
	}
	std::string_view File::GetType() const
	{
		return 0 == ppos ? std::string_view() : std::string_view(buffer.data() + spos + ppos);
	}

	const char* File::data() const noexcept
	{
		return buffer.data();
	}
	File::operator std::string() const noexcept
	{
		return std::string(buffer.data(), buffer.size());
	}
	bool File::empty() const noexcept { return buffer.size() == 0; }

	std::ostream& operator<<(std::ostream& stream, const File& file)
	{
		return stream << file.data();
	}


	void GetFileList(const std::string& folder, FileList& list, const std::string& types)
	{
		CHECK_EQ(0, _access(folder.c_str(), 6)) << "can not access to \"" << folder << "\"";

		//static std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

		HANDLE handle;
		WIN32_FIND_DATAA find_data;

		std::string root = folder;
		if (root.back() != '\\' || root.back() != '/') root.append("\\");

		static std::vector<std::string> type_list = Split(types, "\\|");

		//handle = FindFirstFile(converter.from_bytes((root + "*.*").data()).data(), &find_data);
		handle = FindFirstFileA((root + "*.*").data(), &find_data);
		if (handle != INVALID_HANDLE_VALUE)
		{
			do
			{
				if ('.' == find_data.cFileName[0])
				{
					continue;
				}
				else if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
				{
					GetFileList(root + find_data.cFileName, list, types);
				}
				else
				{
					std::string file_name = find_data.cFileName;

					size_t pos = file_name.find_last_of('.') + 1;
					std::string type = file_name.substr(pos);
					if ("*" == types || std::find(type_list.begin(), type_list.end(), type) != type_list.end())
					{
						list.push_back(File(root + file_name));
					}
				}
			} while (FindNextFileA(handle, &find_data));
		}

		FindClose(handle);
	}
}