#pragma once


template <class base_t> struct default_allocator
{
	base_t* operator()(size_t n) const
	{
		return new base_t[n];
	}
}; 

template <class base_t> struct default_releaser
{
	void operator()(base_t* p) const
	{
		delete[] p;
	}
};


template<class base_t, class allocator_t = default_allocator<base_t>, class releaser_t = default_releaser<base_t>> 
class cache_memory
{
public:
	cache_memory() : size_(0), allocator_(default_allocator<base_t>()), releaser_(default_releaser<base_t>()), memory_(0) {}

	cache_memory(allocator_t f_alloc, releaser_t f_release) 
		: size_(0), allocator_(f_alloc), releaser_(f_release), memory_(nullptr) {}
	
	~cache_memory()
	{
		releaser_(memory_);
	}

	bool reserve(size_t size)
	{
		if (size != size_)
		{
			releaser_(memory_);
			size_ = size;
			memory_ = allocator_(size);

			return true;
		}
		return false;
	}

	base_t* access()
	{
		return memory_;
	}

	const base_t* access() const
	{
		return memory_;
	}

	template <class type_> type_* access_as()
	{
		return (type_*)memory_;
	}

	base_t& operator[](size_t index)
	{
		return memory_[index];
	}

	operator base_t*()
	{
		return memory_;
	}

private:
	size_t size_;

	allocator_t allocator_;
	releaser_t releaser_;

	base_t* memory_;
};