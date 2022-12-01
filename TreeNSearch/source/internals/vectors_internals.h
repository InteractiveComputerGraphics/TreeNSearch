#pragma once
#include <array>
#include <vector>
#include <cstring>
#include <memory>
#include <iostream>
#include <cassert>
#include <immintrin.h>

#if !defined __AVX2__
#define __AVX2__
#endif

#ifdef __linux__
#include <malloc.h>
#endif


/*
	IMPORTANT: These are not general purpose data structures. They have been
	designed exclusively for TreeNSearch and might be unsafe if used in
	other applications.
*/

namespace tns
{
	namespace internals
	{
		/*
			Aligned allocator so that vectorized types can be used in std containers
			from: https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned
		*/
		template <typename T, std::size_t N = 32>
		class AlignmentAllocator {
		public:
			typedef T value_type;
			typedef std::size_t size_type;
			typedef std::ptrdiff_t difference_type;

			typedef T* pointer;
			typedef const T* const_pointer;

			typedef T& reference;
			typedef const T& const_reference;

		public:
			inline AlignmentAllocator() throw () { }

			template <typename T2>
			inline AlignmentAllocator(const AlignmentAllocator<T2, N>&) throw () { }

			inline ~AlignmentAllocator() throw () { }

			inline pointer adress(reference r) {
				return &r;
			}

			inline const_pointer adress(const_reference r) const {
				return &r;
			}

			inline pointer allocate(size_type n) {
#ifdef _WIN32
				return (pointer)_aligned_malloc(n * sizeof(value_type), N);
#elif __linux__
				// NB! Argument order is opposite from MSVC/Windows
				return (pointer)aligned_alloc(N, n * sizeof(value_type));
#else
#error "Unknown platform"
#endif
			}

			inline void deallocate(pointer p, size_type) {
#ifdef _WIN32
				_aligned_free(p);
#elif __linux__
				free(p);
#else
#error "Unknown platform"
#endif
			}

			inline void construct(pointer p, const value_type& wert) {
				new (p) value_type(wert);
			}

			inline void destroy(pointer p) {
				p->~value_type();
			}

			inline size_type max_size() const throw () {
				return size_type(-1) / sizeof(value_type);
			}

			template <typename T2>
			struct rebind {
				typedef AlignmentAllocator<T2, N> other;
			};

			bool operator!=(const AlignmentAllocator<T, N>& other) const {
				return !(*this == other);
			}

			// Returns true if and only if storage allocated from *this
			// can be deallocated from other, and vice versa.
			// Always returns true for stateless allocators.
			bool operator==(const AlignmentAllocator<T, N>& other) const {
				return true;
			}
		};

		/**
		* Alias for aligned vector
		*/
		template<typename T, size_t SIZE>
		using avector = std::vector<T, AlignmentAllocator<T, SIZE>>;

		/**
		* Collection of fixed sized dynamically allocated memory chunks, used
		* to store the neighbor lists.
		* It's purpose it to avoid the reallocations that would happen by using
		* an std::vector<> when the capacity is reached. In chunked_vector<>,
		* another memory chunk is allocated but no reallocations occur.
		*/
		template<typename T, size_t CHUNKSIZE = 1000>
		class chunked_vector
		{
		private:
			/* Fields */
			std::vector<std::array<T*, 2>> chunks; // [{begin, cursor}]
			int current = 0; // Currect chunk in use

		public:
			/* Methods */
			chunked_vector(const chunked_vector<T, CHUNKSIZE>& other) = delete;
			chunked_vector(chunked_vector<T, CHUNKSIZE>&& other) noexcept = default;
			chunked_vector()
			{
				T* ptr = new T[CHUNKSIZE];
				this->chunks.push_back({ ptr, ptr });
				this->current = 0;
			};
			~chunked_vector()
			{
				for (std::array<T*, 2>&chunk : this->chunks) {
					delete[] chunk[0];
				}
			}
			size_t get_chunk_size()
			{
				return CHUNKSIZE;
			}
			T* get_cursor_with_space_to_write(const size_t n)
			{
				if (n > CHUNKSIZE) {
					std::cout << "chunked_vector compare: Cannot allow_to_append n > CHUNKSIZE (" << n << " < " << CHUNKSIZE << ")." << std::endl;
					exit(-1);
				}

				const size_t space_left = CHUNKSIZE - (size_t)std::distance(this->chunks[this->current][0], this->chunks[this->current][1]) - 1;
				if (n > space_left) {
					this->current++;
					if (this->chunks.size() == this->current) {
						T* ptr = new T[CHUNKSIZE];
						this->chunks.push_back({ ptr, ptr });
					}
					this->chunks[this->current][1] = this->chunks[this->current][0];
				}

				T* return_cursor = this->chunks[this->current][1];
				this->chunks[this->current][1] += n;
				*this->chunks[this->current][1] = -1;

				return return_cursor;
			}
			//void set_cursor_back(T* cursor)
			//{
			//	*cursor = -1;
			//	this->chunks[this->current][1] = cursor;
			//}
			void clear()
			{
				this->chunks[0][1] = this->chunks[0][0];
				this->current = 0;
			}
			size_t n_bytes() const
			{
				return this->chunks.size() * CHUNKSIZE * static_cast<size_t>(sizeof(T));
			}
		};

		/**
		* Uninitialized vector. Analogous to std::vector<> but it allows to be resized
		* without initializing the newly allocated memory.
		*/
		template<typename T>
		class uvector
		{
		public:
			T* data = nullptr;
			T* end = nullptr;
			T* cursor = nullptr;  // Cursor is handled by the user

			uvector() = default;
			~uvector()
			{
				if (this->data != nullptr) {
					delete[] this->data;
				}
			}
			uvector(const uvector& other)
			{
				if (other.data != nullptr) {
					this->data = new T[std::distance(other.data, other.end)];
					this->end = this->data + std::distance(other.data, other.end);
					this->cursor = this->data + std::distance(other.data, other.cursor);
					memcpy(this->data, other.data, sizeof(T) * std::distance(other.data, other.end));
				}
				else {
					this->data = nullptr;
					this->end = nullptr;
					this->cursor = nullptr;
				}
			}

			// needed for std::sort
			uvector& operator=(uvector other)
			{
				std::swap(this->data, other.data);
				std::swap(this->end, other.end);
				std::swap(this->cursor, other.cursor);
				return *this;
			}
			void init(const int n)
			{
				assert(n >= 0);
				if (this->data != nullptr) {
					delete[] this->data;
					this->data = nullptr;
					this->end = nullptr;
					this->cursor = nullptr;
				}

				if (n > 0) {
					this->data = new T[n];
					this->cursor = this->data;
					this->end = this->data + n;
				}
			}

			void init_with_at_least_size(const int n, const double multiplier = 1.0)
			{
				assert(n >= 0);
				if (this->capacity() < n) {
					this->init((int)(multiplier * n));
				}
				else {
					this->cursor = this->data;
				}
			}

			void grow_while_keeping_data(const int n)
			{
				assert(n >= 0);
				if (n > this->capacity()) {
					size_t cursor_ = std::distance(this->data, this->cursor);
					T* data_ = new T[n];
					memcpy(data_, this->data, sizeof(T) * this->capacity());  // Cursor is handled by the user
					delete[] this->data;
					this->data = data_;
					this->cursor = this->data + cursor_;
					this->end = this->data + n;
				}
			}

			int capacity() const
			{
				return (int)std::distance(this->data, this->end);
			}

			int capacity_left() const
			{
				return (int)std::distance(this->cursor, this->end);
			}

			int size() const
			{
				return (int)std::distance(this->data, this->cursor);
			}

			template<typename INDEX_TYPE>
			T& operator[](INDEX_TYPE idx) {
				return this->data[idx];
			}

			template<typename INDEX_TYPE>
			T operator[](INDEX_TYPE idx) const {
				return this->data[idx];
			}
		};
	}
}
