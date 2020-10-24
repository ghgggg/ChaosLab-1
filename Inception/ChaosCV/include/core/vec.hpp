#pragma once

#include "core.hpp"
#include "allocator.hpp"
#include <numeric>

namespace chaos
{
	template<class Type> class VecConstIterator;
	template<class Type, std::enable_if_t<std::is_arithmetic_v<Type>, bool> = true>
	class Vec
	{
	public:
		using ConstIterator = VecConstIterator<Type>;
		Vec() = default;
		
		virtual ~Vec() { Deallocate(); }

		template<class Tp, std::enable_if_t<std::is_convertible_v<Tp,Type>, bool> = true>
		Vec(const std::initializer_list<Tp>& list)
		{
			Allocate(list.size());
			size_t i = 0;
			for (const auto& elem : list)
			{
				buf[i++] = static_cast<Type>(elem);
			}
		}

		template<class Tp, std::enable_if_t<std::is_convertible_v<Tp, Type>, bool> = true>
		Vec(size_t size, const Tp* ptr)
		{
			Allocate(size);
			for (size_t i = 0; i < sz; i++)
			{
				buf[i] = ptr[i];
			}
		}

		Vec(const Vec& v)
		{
			Allocate(v.size());
			for (size_t i = 0; i < sz; i++)
			{
				buf[i] = v[i];
			}
		}
		const Vec& operator=(const Vec& v)
		{
			if (this == &v)
			{
				return *this;
			}

			Allocate(v.size());
			for (size_t i = 0; i < sz; i++)
			{
				buf[i] = v[i];
			}
			return *this;
		}

		void Resize(size_t size)
		{
			if (size < sz)
				return;

			size_t prev_size = sz;
			Type* prev_buf = buf;

			buf = new Type[size];
			sz = size;

			size_t i = 0;
			for (; i < prev_size; i++)
			{
				buf[i] = prev_buf[i];
			}
			for (; i < size; i++)
				buf[i] = Type();

			delete[] prev_buf;
		}

		void Allocate(size_t size)
		{
			if (size <= sz)
			{
				sz = size;
				return;
			}
			Deallocate();
			sz = size;
			buf = new Type[size];
		}

		void Deallocate()
		{
			if (buf)
			{
				delete[] buf;
				buf = nullptr;
			}
			sz = 0;
		}

		template<class Tp, std::enable_if_t<std::is_convertible_v<Tp, uint>, bool> = true>
		void Insert(size_t pos, const Tp& val)
		{
			if (pos < sz)
			{
				size_t rest = sz - pos; // rest < vec.size
				Resize(sz + 1);
				memmove(buf + pos + 1, buf + pos, rest * sizeof(Type));
				buf[pos] = val;
			}
			else
			{
				Resize(pos + 1);
				buf[pos] = val;
			}
		}
		void Remove(size_t pos)
		{
			CHECK_LT(pos, sz) << "out of range";
			size_t rest = sz - pos;
			memmove(buf + pos, buf + pos + 1, rest * sizeof(Type));
			Resize(sz - 1);
		}

		template<class Tp, std::enable_if_t<std::is_convertible_v<Type, Tp>, bool> =  true>
		operator std::vector<Tp>() const
		{
			std::vector<Tp> vec(sz);
			for (size_t i = 0; i < sz; i++) vec[i] = static_cast<Tp>(buf[i]);
			return vec;
		}

		inline size_t size() const noexcept { return sz; }
		inline const Type& operator[](size_t i) const { CHECK_LT(i, sz) << "out of range"; return buf[i]; }
		inline Type& operator[](size_t i) { CHECK_LT(i, sz) << "out of range"; return buf[i]; }
		inline const Type* data() const noexcept { return buf; }
		inline Type* data() noexcept { return buf; }
		inline const Type& back() const noexcept { return buf[sz-1]; }
		inline Type& back() noexcept { return buf[sz - 1]; }

		ConstIterator begin() const { return ConstIterator(*this); }
		ConstIterator end() const { return ConstIterator(*this) + sz; }

		bool empty() const noexcept { return sz == 0 || buf == nullptr; }

		friend std::ostream& operator<<(std::ostream& stream, const Vec& v)
		{
			stream << "[";
			for (int i = 0; i < v.sz; i++)
			{
				stream << Format(",%d" + !i, v[i]);
			}
			return stream << "]";
		}

		friend bool operator==(const Vec& lhs, const Vec& rhs) noexcept
		{
			if (lhs.size() != rhs.size()) return false;
			for (size_t i = 0; i < lhs.size(); i++)
			{
				if (lhs[i] != rhs[i]) return false;
			}
			return true;
		}

	protected:
		Type* buf = nullptr;
		size_t sz = 0;
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


	class Steps : public Vec<uint>
	{
	public:
		using Vec<uint>::Vec;

		Steps(size_t sz) { Allocate(sz); }
	};

	class Shape : public Vec<uint>
	{
	public:
		using Vec<uint>::Vec;

		Shape(uint w) 
		{ 
			Allocate(1); 
			buf[0] = w; 
		}
		Shape(uint h, uint w) 
		{
			Allocate(2); 
			buf[0] = h; buf[1] = w;
		}
		Shape(uint c, uint h, uint w) 
		{ 
			Allocate(3); 
			buf[0] = c; buf[1] = h; buf[2] = w;
		}
		Shape(uint n, uint c, uint h, uint w) 
		{ 
			Allocate(4); 
			buf[0] = n; buf[1] = c; buf[2] = h; buf[3] = w;
		}

		size_t vol() const noexcept { return std::accumulate(begin(), end(), 1, std::multiplies<uint>()); }

		Steps steps() const noexcept
		{
			Steps _steps(sz);
			_steps[sz - 1] = 1;
			for (int64 i = sz - 2; i >= 0; i--)
			{
				_steps[i] = buf[i + 1] * _steps[i + 1];
			}
			return _steps;
		}

		
	};

}