#pragma once

#include "core.hpp"
#include "allocator.hpp"
#include <numeric>

namespace chaos
{
	template<class Type> class VecConstIterator;
	template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
	class Vec : public AutoBuffer<Type, 8>
	{
	public:
		using Buffer = AutoBuffer<Type, 8>;
		using ConstIterator = VecConstIterator<Type>;

		Vec() : Buffer(0) {}
		virtual ~Vec() = default;

		template<class Tp, std::enable_if_t<std::is_convertible_v<Tp, Type>, bool> = true>
		Vec(const std::initializer_list<Tp>& list) : Buffer(list.size())
		{
			const Tp* data = list.begin();
			for (size_t i = 0; i < list.size(); i++)
			{
				ptr[i] = static_cast<Type>(data[i]);
			}
		}

		template<class Tp, std::enable_if_t<std::is_convertible_v<Tp, Type>, bool> = true>
		Vec(size_t size, const Tp* data) : Buffer(size)
		{
			for (size_t i = 0; data && i < size; i++)
			{
				ptr[i] = static_cast<Type>(data[i]);
			}
		}

		template<class Tp, std::enable_if_t<std::is_convertible_v<Type, Tp>, bool> = true>
		operator std::vector<Tp>() const
		{
			std::vector<Tp> data;
			data.reserve(sz);
			for (size_t i = 0; i < sz; i++)
			{
				data.emplace_back(ptr[i]);
			}
			return data;
		}

		ConstIterator begin() const { return ConstIterator(*this); }
		ConstIterator end() const { return ConstIterator(*this) + sz; }

		bool empty() const noexcept { return sz == 0 || ptr == nullptr; }

		const Type& back() const { return ptr[sz - 1]; }
		Type& back() { return ptr[sz - 1]; }

		friend std::ostream& operator<<(std::ostream& stream, const Vec<Type>& vec)
		{
			stream << "[";
			for (int i = 0; i < vec.sz; i++)
			{
				stream << Format(",%d" + !i, vec[i]);
			}
			return stream << "]";
		}

	protected:
		using Buffer::sz;
		using Buffer::ptr;
	};


	template<class Type>
	class VecConstIterator : public std::iterator_traits<Type*>
	{
	public:
		VecConstIterator(const Vec<Type>& vec) : vec(vec), index(0) {}

		const Type& operator*() const { return vec[index]; }

		VecConstIterator& operator++() noexcept
		{
			index++;
			return *this;
		}

		VecConstIterator& operator+(size_t step) noexcept
		{
			index += step;
			return *this;
		}
		VecConstIterator& operator-(size_t step) noexcept
		{
			index -= step;
			return *this;
		}

		bool operator!=(const VecConstIterator& iter) const noexcept { return index != iter.index; }
	protected:
		const Vec<Type>& vec;
		size_t index;
	private:
		friend Vec<Type>;
	};

	class CHAOS_API Shape : public Vec<uint>
	{
	public:
		using Vec<uint>::Vec;

		Shape(uint w) : Vec({w}) {}
		Shape(uint h, uint w) : Vec({ h, w }) {}
		Shape(uint c, uint h, uint w) : Vec({ c, h ,w }) {}
		Shape(uint n, uint c, uint h, uint w) : Vec({n, c, h, w}) {}

		~Shape() {}

		size_t vol() const noexcept { return std::accumulate(begin(), end(), 1, std::multiplies<uint>()); }

		template<class Tp, std::enable_if_t<std::is_convertible_v<Tp, uint>, bool> = true>
		void Insert(size_t pos, const Tp& val)
		{
			if (pos < sz)
			{
				size_t rest = sz - pos; // rest < vec.size
				Resize(sz + 1);
				memmove(ptr + pos + 1, ptr + pos, rest * sizeof(uint));
				ptr[pos] = val;
			}
			else
			{
				Resize(pos + 1);
				ptr[pos] = val;
			}
		}
		void Remove(size_t pos)
		{
			CHECK_LT(pos, sz) << "out of range";
			size_t rest = sz - pos;
			memmove(ptr + pos, ptr + pos + 1, rest * sizeof(uint));
			Resize(sz - 1);
		}

		friend bool operator==(const Shape& lhs, const Shape& rhs)
		{
			if (lhs.size() != rhs.size()) return false;
			for (size_t i = 0; i < lhs.size(); i++)
			{
				if (lhs[i] != rhs[i]) return false;
			}
			return true;
		}
	};

	class CHAOS_API Steps : public Vec<uint> // to save steps for shape
	{
	public:
		using Vec<uint>::Vec;

		Steps(const Shape& shape, uint elem_size)
		{
			Allocate(shape.size());
			ptr[sz - 1] = elem_size;
			for (int64 i = shape.size() - 2; i >= 0; i--)
			{
				ptr[i] = shape[i + 1] * ptr[i + 1];
			}
		}

		~Steps() {};
	};

	class CHAOS_API Point : Vec<float>
	{
	public:
		Point(float x, float y) : Vec({ x, y }) {}

		__declspec(property(get = GetX)) float x;
		float GetX() const
		{
			return ptr[0];
		}

		__declspec(property(get = GetY)) float y;
		float GetY() const
		{
			return ptr[1];
		}
	};
}