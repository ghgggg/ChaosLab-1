#pragma once

#include <intrin.h>

#define MALLOC_ALIGN 16

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
		virtual ~AutoBuffer() { Deallocate(); }

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
		inline size_t size() const noexcept { return sz; }
		//! returns pointer to the real buffer, stack-allocated or heap-allocated
		inline Type* data() noexcept { return ptr; }
		//! returns read-only pointer to the real buffer, stack-allocated or heap-allocated
		inline const Type* data() const noexcept { return ptr; }

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


	/// <summary>
	/// <para>Aligns a pointer to the specified number of bytes</para>
	/// <para>The function returns the aligned pointer of the same type as the input pointer:</para>
	/// <para>(_Tp*)(((size_t)ptr + n - 1) and -n)</para>
	/// </summary>
	/// <param name="ptr">Aligned pointer</param>
	/// <param name="n">Alignment size that must be a power of two</param>
	/// <return>The aligned pointer of the same type as the input pointer</return>
	template<typename _Tp> static inline _Tp* AlignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
	{
		CHECK((n & (n - 1)) == 0); // n is a power of 2
		return (_Tp*)(((size_t)ptr + n - 1) & -n);
	}

	/// <summary>
	/// <para>Aligns a buffer size to the specified number of bytes</para>
	/// <para>The function returns the minimum number that is greater than or equal to sz and is divisible by n :</para>
	/// <para>(sz + n - 1) and -n</para>
	/// </summary>
	/// <param name="sz">Buffer size to align</param>
	/// <param name="n">Alignment size that must be a power of two</param>
	/// <return>The minimum number that is greater than or equal to sz and is divisible by n</return>
	static inline size_t AlignSize(size_t sz, int n)
	{
		CHECK((n & (n - 1)) == 0); // n is a power of 2
		return (sz + n - 1) & -n;
	}

	static inline void* FastMalloc(size_t size)
	{
		return _aligned_malloc(size, MALLOC_ALIGN);
	}

	static inline void FastFree(void* ptr)
	{
		if (ptr) { _aligned_free(ptr); }
	}


	class CHAOS_API Allocator
	{
	public:
		virtual ~Allocator() = default;

		virtual void* FastMalloc(size_t size) = 0;
		virtual void FastFree(void* ptr) = 0;
	};

	class CHAOS_API PoolAllocator : public Allocator
	{
	public:
		PoolAllocator();
		~PoolAllocator();

		virtual void* FastMalloc(size_t size) override;
		virtual void FastFree(void* ptr) override;

		void SetSizeCompareRatio(float scr);
		__declspec(property(put = SetSizeCompareRatio)) uint size_compare_ratio;

		void Clear();
	private:
		std::mutex budgets_lock;
		std::mutex payouts_lock;
		uint _size_compare_ratio; // 0~256
		std::list<std::pair<size_t, void*>> budgets;
		std::list<std::pair<size_t, void*>> payouts;
	};

	class CHAOS_API UnlockedPoolAllocator : public Allocator
	{
	public:
		UnlockedPoolAllocator();
		~UnlockedPoolAllocator();

		virtual void* FastMalloc(size_t size) override;
		virtual void FastFree(void* ptr) override;

		void SetSizeCompareRatio(float scr);
		__declspec(property(put = SetSizeCompareRatio)) uint size_compare_ratio;

		void Clear();
	private:
		uint _size_compare_ratio; // 0~256
		std::list<std::pair<size_t, void*>> budgets;
		std::list<std::pair<size_t, void*>> payouts;
	};
}