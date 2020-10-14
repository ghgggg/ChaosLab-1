#include "core/core.hpp"

namespace chaos
{
	PoolAllocator::PoolAllocator()
	{
		_size_compare_ratio = 192; // 0.75f * 256
	}

	PoolAllocator::~PoolAllocator()
	{
		Clear();

		if (not payouts.empty())
		{
			LOG(ERROR) << "pool allocator destroyed too early";
			for (const auto& [size, ptr] : payouts)
			{
				LOG(ERROR) << Format("%p still in use", ptr);
			}
			LOG(FATAL) << "pool allocator destroyed too early";
		}
	}

	void PoolAllocator::Clear()
	{
		budgets_lock.lock();
		for (const auto& [size, ptr] : budgets)
		{
			chaos::FastFree(ptr);
		}
		budgets.clear();

		budgets_lock.unlock();
	}

	void PoolAllocator::SetSizeCompareRatio(float scr)
	{
		CHECK_GE(scr, 0.f) << "invalid size compare ratio " << scr;
		CHECK_LE(scr, 1.f) << "invalid size compare ratio " << scr;

		_size_compare_ratio = static_cast<uint>(scr * 256.f);
	}

	void* PoolAllocator::FastMalloc(size_t size)
	{
		budgets_lock.lock();

		// find free budget
		for (auto it = budgets.begin(); it != budgets.end(); ++it)
		{
			size_t bs = it->first;

			// size_compare_ratio ~ 100%
			if (bs >= size && ((bs * _size_compare_ratio) >> 8) <= size)
			{
				void* ptr = it->second;
				budgets.erase(it);
				budgets_lock.unlock();
				payouts_lock.lock();
				payouts.push_back(std::make_pair(bs, ptr));
				payouts_lock.unlock();
				return ptr;
			}
		}

		budgets_lock.unlock();

		// new
		void* ptr = chaos::FastMalloc(size);
		payouts_lock.lock();
		payouts.push_back(std::make_pair(size, ptr));
		payouts_lock.unlock();

		return ptr;
	}

	void PoolAllocator::FastFree(void* ptr)
	{
		payouts_lock.lock();

		// return to budgets
		for (auto it = payouts.begin(); it != payouts.end(); ++it)
		{
			if (it->second == ptr)
			{
				size_t size = it->first;
				payouts.erase(it);
				payouts_lock.unlock();
				budgets_lock.lock();
				budgets.push_back(std::make_pair(size, ptr));
				budgets_lock.unlock();
				return;
			}
		}

		payouts_lock.unlock();

		LOG(FATAL) << Format("pool allocator get wild %p", ptr);
		chaos::FastFree(ptr);
	}

	UnlockedPoolAllocator::UnlockedPoolAllocator()
	{
		_size_compare_ratio = 192; // 0.75 * 256
	}

	UnlockedPoolAllocator::~UnlockedPoolAllocator()
	{
		Clear();

		if (not payouts.empty())
		{
			LOG(ERROR) << "unlocked pool allocator destroyed too early";
			for (const auto& [size, ptr] : payouts)
			{
				LOG(ERROR) << Format("%p still in use", ptr);
			}
			LOG(FATAL) << "unlocked pool allocator destroyed too early";
		}
	}

	void UnlockedPoolAllocator::Clear()
	{
		for (const auto& [size, ptr] : budgets)
		{
			chaos::FastFree(ptr);
		}
		budgets.clear();
	}

	void UnlockedPoolAllocator::SetSizeCompareRatio(float scr)
	{
		CHECK_GE(scr, 0.f) << "invalid size compare ratio " << scr;
		CHECK_LE(scr, 1.f) << "invalid size compare ratio " << scr;

		_size_compare_ratio = static_cast<uint>(scr * 256.f);
	}

	void* UnlockedPoolAllocator::FastMalloc(size_t size)
	{
		// find free budget
		for (auto it = budgets.begin(); it != budgets.end(); ++it)
		{
			size_t bs = it->first;

			// size_compare_ratio ~ 100%
			if (bs >= size && ((bs * _size_compare_ratio) >> 8) <= size)
			{
				void* ptr = it->second;
				budgets.erase(it);
				payouts.push_back(std::make_pair(bs, ptr));
				return ptr;
			}
		}

		// new
		void* ptr = chaos::FastMalloc(size);
		payouts.push_back(std::make_pair(size, ptr));
		return ptr;
	}

	void UnlockedPoolAllocator::FastFree(void* ptr)
	{
		for (auto it = payouts.begin(); it != payouts.end(); ++it)
		{
			if (it->second == ptr)
			{
				size_t size = it->first;

				payouts.erase(it);

				budgets.push_back(std::make_pair(size, ptr));

				return;
			}
		}

		LOG(FATAL) << Format("unlocked pool allocator get wild %p", ptr);
		chaos::FastFree(ptr);
	}
}