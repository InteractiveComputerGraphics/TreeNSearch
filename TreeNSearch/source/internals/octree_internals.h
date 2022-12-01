#pragma once
#include <iostream>
#include <array>
#include <vector>
#include <cstring>
#include <algorithm>
#include <limits>

#include "vectors_internals.h"

/*
	IMPORTANT: These are not general purpose data structures. They have been
	designed exclusively for TreeNSearch and might be unsafe if used for
	other applications.
*/

namespace tns
{
	namespace internals
	{
		/**
		* Axis-aligned bounding box with minimal functionality to be used
		* in the octree nodes.
		*/
		template<typename T>
		class AABB
		{
		public:
			// Fields
			std::array<T, 3> bottom = { std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::max() };
			std::array<T, 3> top = { std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest() };

			// Methods
			inline AABB<T> get_extended(const T extension) const
			{
				AABB<T> extended;
				extended.bottom = { this->bottom[0] - extension, this->bottom[1] - extension, this->bottom[2] - extension };
				extended.top = { this->top[0] + extension, this->top[1] + extension, this->top[2] + extension };
				return extended;
			};
			inline std::array<AABB<T>, 2> split(const int dim, const T pivot) const
			{
				std::array<AABB<T>, 2> children;
				children[0] = *this;
				children[1] = *this;
				children[0].top[dim] = pivot;
				children[1].bottom[dim] = pivot;
				return children;
			};
			inline bool intersects(const AABB<T>& other) const
			{
				for (size_t dim = 0; dim < 3; dim++) {
					if ((this->bottom[dim] > this->top[dim]) || (other.top[dim] < other.bottom[dim])) {
						return false;
					}
				}
				return true;
			};
		};

		/**
		* Collection of cells in SIMD friendly container with self-managed memory.
		*/
		class CellList
		{
		public:
			int capacity = 0;
			int* offsets = nullptr;
			uint16_t* i = nullptr;
			uint16_t* j = nullptr;
			uint16_t* k = nullptr;
			float* radii = nullptr;
			int n_cells = 0;

			CellList() = default;
			~CellList()
			{
				if (this->offsets != nullptr) { delete[] this->offsets; }
				if (this->i != nullptr) { delete[] this->i; }
				if (this->j != nullptr) { delete[] this->j; }
				if (this->k != nullptr) { delete[] this->k; }
				if (this->radii != nullptr) { delete[] this->radii; }
			}
			CellList(const CellList& other)
			{
				this->capacity = other.capacity;
				this->n_cells = other.n_cells;
				if (other.offsets != nullptr) {
					this->offsets = new int[this->capacity + 1];
					this->i = new uint16_t[this->capacity];
					this->j = new uint16_t[this->capacity];
					this->k = new uint16_t[this->capacity];

					memcpy(this->offsets, other.offsets, sizeof(int) * (this->capacity + 1));
					memcpy(this->i, other.i, sizeof(uint16_t) * this->capacity);
					memcpy(this->j, other.j, sizeof(uint16_t) * this->capacity);
					memcpy(this->k, other.k, sizeof(uint16_t) * this->capacity);
				}
				if (other.radii != nullptr) {
					this->radii = new float[this->capacity];
					memcpy(this->radii, other.radii, sizeof(float) * this->capacity);
				}
			}
			void init_with_at_least_size(const int n)
			{
				this->n_cells = 0;
				if (this->offsets == nullptr) {
					this->offsets = new int[n + 1];
					this->i = new uint16_t[n];
					this->j = new uint16_t[n];
					this->k = new uint16_t[n];
					this->capacity = n;
					this->n_cells = 0;
				}
				else {
					if (this->capacity < n) {
						delete[] this->offsets; this->offsets = new int[n + 1];
						delete[] this->i; this->i = new uint16_t[n];
						delete[] this->j; this->j = new uint16_t[n];
						delete[] this->k; this->k = new uint16_t[n];
						this->capacity = n;
					}
				}
			}
			void grow_while_keeping_data(const int n)
			{
				if (n > this->capacity) {
					int* offsets_ = new int[n + 1];
					uint16_t* i_ = new uint16_t[n];
					uint16_t* j_ = new uint16_t[n];
					uint16_t* k_ = new uint16_t[n];

					memcpy(offsets_, this->offsets, sizeof(int) * (this->capacity + 1));
					memcpy(i_, this->i, sizeof(uint16_t) * this->capacity);
					memcpy(j_, this->j, sizeof(uint16_t) * this->capacity);
					memcpy(k_, this->k, sizeof(uint16_t) * this->capacity);

					delete[] this->offsets;
					delete[] this->i;
					delete[] this->j;
					delete[] this->k;

					this->offsets = offsets_;
					this->i = i_;
					this->j = j_;
					this->k = k_;

					this->capacity = n;
				}
			}
			void init_radii()
			{
				if (this->radii == nullptr) {
					this->radii = new float[this->n_cells];
				}
				else {
					delete[] this->radii; this->radii = new float[this->n_cells];
				}
			}
		};

		/**
		* Recursive structure that can hold any type of buffer. Used to store
		* the octree.
		*/
		template<typename NODE_BUFFER, size_t N>
		class RecursiveBuffer
		{
		public:
			// Fiedls
			NODE_BUFFER buffer;
			RecursiveBuffer* parent = nullptr;
			std::array<RecursiveBuffer*, N> children;

			// Methods
			inline RecursiveBuffer()
			{
				for (int i = 0; i < N; i++) {
					this->children[i] = nullptr;
				}
			};
			inline ~RecursiveBuffer()
			{
				this->delete_children();
			};
			inline RecursiveBuffer& get_child(const size_t i)
			{
				if (this->children[i] == nullptr) {
					this->children[i] = new RecursiveBuffer();
					this->children[i]->parent = this;
				}
				return *this->children[i];
			};
			inline void populate_children()
			{
				if (this->children[0] == nullptr) {
					for (int i = 0; i < N; i++) {
						this->children[i] = new RecursiveBuffer();
						this->children[i]->parent = this;
					}
				}
			}
			inline void delete_children()
			{
				if (this->children[0] != nullptr) {
					for (int i = 0; i < N; i++) {
						delete this->children[i];
						this->children[i] = nullptr;
					}
				}
			}
		};

		/**
		* Data corresponding to an octree node.
		*/
		struct OctreeNode
		{
			AABB<uint16_t> domain;
			uvector<int> cell_indices;
			float max_search_radius = -1.0;
		};
		using RecursiveOctreeNode = RecursiveBuffer<OctreeNode, 8>;

		/**
		* Container used to allocate the temporal data required to solve the brute force
		* of the neighborhood search.
		*/
		struct BruteforceBuffer
		{
			struct LeafPoints
			{
				std::vector<std::array<float, 3>> points;
				std::vector<uint8_t> inside;
				std::vector<std::vector<int>> inside_indices;
				std::vector<int> indices;
				std::vector<float> radii_sq;
				std::vector<int> set_offsets;
			};
			struct LeafPointsSIMD
			{
				avector<std::array<__m256, 3>, 32> xyz;
				avector<__m256, 32> x;
				avector<__m256, 32> y;
				avector<__m256, 32> z;
				avector<__m256i, 32> indices;
				avector<__m256, 32> radii_sq;

				std::vector<std::vector<int>> inside_indices;
				std::vector<int> set_offsets;
			};

			LeafPoints points;
			LeafPointsSIMD points_simd;
			std::vector<int> neighbors_buffer;
		};
	}
}
