#pragma once

namespace tns
{
	/**
	 * @brief Handle to a list of neighbors.
	*/
	class NeighborList
	{
	private:
		const int* ptr;
		NeighborList(const int* neighborlist_ptr)
			: ptr(neighborlist_ptr + 1) {}
		friend class TreeNSearch;

	public:
		/**
		* @return Number of neighbors.
		*/
		inline int size() const
		{
			return *(this->ptr - 1);
		}
		/**
		* @param i i-th neighbor of the list.
		* @return Index of the neighbor.
		*/
		inline int operator[](const size_t i) const
		{
			return this->ptr[i];
		}
		/**
		* @return Pointer to the neighborlist.
		*/
		inline const int* get_ptr() const
		{
			return this->ptr;
		}
	};
}
