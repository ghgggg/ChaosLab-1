#pragma once

#include <intrin.h>

// exchange-add operation for atomic operations on reference counters
// Just for windows, reference to NCNN
#define CHAOS_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)

namespace chaos
{
	/// <summary>
	/// <para>Automatically Allocated Buffer Class</para>
	/// <para>The class is used for temporary buffers in functions and methods.</para>
	/// <para>If a temporary buffer is usually small (a few K's of memory),</para>
	/// <para>but its size depends on the parameters, it makes sense to create a small</para>
	/// <para>fixed-size array on stack and use it if it's large enough. If the required buffer size</para>
	/// <para>is larger than the fixed size, another buffer of sufficient size is allocated dynamically</para>
	/// <para>and released after the processing. Therefore, in typical cases, when the buffer size is small,</para>
	/// <para>there is no overhead associated with malloc()/free().</para>
	/// <para>At the same time, there is no limit on the size of processed data.</para>
	/// <para>This is what AutoBuffer does. The template takes 2 parameters - type of the buffer elements and</para>
	/// <para>the number of stack-allocated elements.</para>
	/// </summary>
	template<class Type, size_t fixed_size = 1024 / sizeof(Type) + 8> class AutoBuffer
	{
	public:
		//! the default constructor
		AutoBuffer()
		{
			ptr = buf;
			sz = fixed_size;
		}

		//! constructor taking the real buffer size
		explicit AutoBuffer(size_t _size)
		{
			ptr = buf;
			sz = fixed_size;
			Allocate(_size);
		}

		//! the copy constructor
		AutoBuffer(const AutoBuffer<Type, fixed_size>& abuf)
		{
			ptr = buf;
			sz = fixed_size;
			Allocate(abuf.size());
			for (size_t i = 0; i < sz; i++)
				ptr[i] = abuf.ptr[i];
		}
		//! the assignment operator
		AutoBuffer<Type, fixed_size>& operator=(const AutoBuffer<Type, fixed_size>& abuf)
		{
			if (this != &abuf)
			{
				Deallocate();
				Allocate(abuf.size());
				for (size_t i = 0; i < sz; i++)
					ptr[i] = abuf.ptr[i];
			}
			return *this;
		}

		//! destructor. calls deallocate()
		~AutoBuffer() { Deallocate(); }

		//! allocates the new buffer of size _size. if the _size is small enough, stack-allocated buffer is used
		void Allocate(size_t _size)
		{
			if (_size <= sz)
			{
				sz = _size;
				return;
			}
			Deallocate();
			sz = _size;
			if (_size > fixed_size)
			{
				ptr = new Type[_size];
			}
		}
		//! deallocates the buffer if it was dynamically allocated
		void Deallocate()
		{
			if (ptr != buf)
			{
				delete[] ptr;
				ptr = buf;
				sz = fixed_size;
			}
		}
		//! resizes the buffer and preserves the content
		void Resize(size_t _size)
		{
			if (_size <= sz)
			{
				sz = _size;
				return;
			}
			size_t i, prevsize = sz, minsize = std::min(prevsize, _size);
			Type* prevptr = ptr;

			ptr = _size > fixed_size ? new Type[_size] : buf;
			sz = _size;

			if (ptr != prevptr)
				for (i = 0; i < minsize; i++)
					ptr[i] = prevptr[i];
			for (i = prevsize; i < _size; i++)
				ptr[i] = Type();

			if (prevptr != buf)
				delete[] prevptr;
		}
		//! returns the current buffer size
		size_t size() const { return sz; }
		//! returns pointer to the real buffer, stack-allocated or heap-allocated
		inline Type* data() { return ptr; }
		//! returns read-only pointer to the real buffer, stack-allocated or heap-allocated
		inline const Type* data() const { return ptr; }

		//! returns a reference to the element at specified location. No bounds checking is performed in Release builds.
		inline Type& operator[] (size_t i) { CHECK_LT(i, sz) << "out of range"; return ptr[i]; }
		//! returns a reference to the element at specified location. No bounds checking is performed in Release builds.
		inline const Type& operator[] (size_t i) const { CHECK_LT(i, sz) << "out of range"; return ptr[i]; }

	protected:
		//! pointer to the real buffer, can point to buf if the buffer is small enough
		Type* ptr;
		//! size of the real buffer
		size_t sz;
		//! pre-allocated buffer. At least 1 element to confirm C++ standard requirements
		Type buf[(fixed_size > 0) ? fixed_size : 1];
	};

}