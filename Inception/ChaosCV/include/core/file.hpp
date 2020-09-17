#pragma once

#include "def.hpp"
#include "allocator.hpp"

namespace chaos
{
	class CHAOS_API File
	{
	public:
		File() = default;
		File(const std::string& file);
		File(const char* file);

		std::string_view GetPath() const;
		std::string_view GetName() const;
		std::string_view GetType() const;

		const char* data() const noexcept;
		operator std::string() const noexcept; // copy data
		bool empty() const noexcept;

		__declspec(property(get = GetName)) std::string_view name; // read only, 0-copy Format("%.*s", name.size(), name.data());
		__declspec(property(get = GetPath)) std::string_view path; // read only, 0-copy Format("%.*s", path.size(), path.data());
		__declspec(property(get = GetType)) std::string_view type; // read only, 0-copy 

		CHAOS_API friend std::ostream& operator<<(std::ostream& steram, const File& file);
	private:
		AutoBuffer<char, 1024> buffer;
		size_t ppos; // last point pose
		size_t spos; // last slash pose
	};
	using FileList = std::vector<File>;

	CHAOS_API void GetFileList(const std::string& folder, FileList& list, const std::string& types = "*");
}