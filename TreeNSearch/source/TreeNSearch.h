#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <omp.h>

#include "internals/vectors_internals.h"
#include "internals/octree_internals.h"
#include "NeighborList.h"

// Forward declaration to avoid including the whole taskflow interface
namespace tf
{
	class Executor;
	class Taskflow;
	class Subflow;
}

namespace tns
{
	/**
	 * @brief Neighborhood Search method based on the paper "Fast Octree Neighborhood 
	 * Search for SPH Simulations": https://animation.rwth-aachen.de/publication/0579/.
	 * 
	 * Given different collections of points, it finds which points are inside the search
	 * radius of each other. The search radius can be globally constant or variable per
	 * point.
	*/
	class TreeNSearch
	{
	public:
		// =========================================================================================================
		// ===============================================  METHODS  ===============================================
		// =========================================================================================================
		
		// -----------------------------------------------  CONSTRUCTORS  -----------------------------------------------
		TreeNSearch() = default;
		~TreeNSearch() = default;

		// -----------------------------------------------  MAIN INTERFACE  -----------------------------------------------
		/**
		 * @brief Adds a collection of points to search and/or to be found in **fixed radius** neighborhood search mode.
		 * 
		 * @warning It is not possible to mix fixed radius with variable radius search radius. If one point set has varaible search radius, all of them must be declared as variable.
		 * 
		 * @param points_begin Pointer to the points coordinates. Coordinates must be contiguous in memory and in `xyzxyzxyz` layout.
		 * @param n_points Number of points in the set.
		 * 
		 * @return Point set id.
		*/
		int add_point_set(const float* points_begin, const int n_points);

		/**
		 * @brief Adds a collection of points to search and/or to be found in **fixed radius** neighborhood search mode.
		 *
		 * @warning TreeNSearch works with type float internally. Declaring point sets as type double will incurr into data copying at the beginning of the neighborhood search process.
		 * @warning It is not possible to mix fixed radius with variable radius search radius. If one point set has varaible search radius, all of them must be declared as variable.
		 *
		 * @param points_begin Pointer to the points coordinates. Coordinates must be contiguous in memory and in `xyzxyzxyz` layout.
		 * @param n_points Number of points in the set.
		 *
		 * @return Point set id.
		*/
		int add_point_set(const double* points_begin, const int n_points);
		
		/**
		 * @brief Updates the coordinates data of a point set in **fixed radius** neighborhood search mode.
		 * 
		 * @param set_id Id of the point set to be resized.
		 * @param points_begin Pointer to the points coordinates. Coordinates must be contiguous in memory and in `xyzxyzxyz` layout.
		 * @param n_points Number of points in the set.
		*/
		void resize_point_set(const int set_id, const float* points_begin, const int n_points);

		/**
		 * @brief Updates the coordinates data of a point set in **fixed radius** neighborhood search mode.
		 * 
		 * @param set_id Id of the point set to be resized.
		 * @param points_begin Pointer to the points coordinates. Coordinates must be contiguous in memory and in `xyzxyzxyz` layout.
		 * @param n_points Number of points in the set.
		*/
		void resize_point_set(const int set_id, const double* points_begin, const int n_points);

		/**
		 * @brief Sets the global search radius in **fixed radius** neighborhood search mode.
		 * 
		 * @warning If one point set has variable search radius, all of them must be declared as variable and this method should not be used.
		 * 
		 * @param search_radius Global search radius to be used for all the points in the neighborhood search.
		*/
		void set_search_radius(const float search_radius);

		/**
		 * @brief Sets the global search radius in **fixed radius** neighborhood search mode.
		 * 
		 * @warning If one point set has variable search radius, all of them must be declared as variable and this method should not be used.
		 * 
		 * @param search_radius Global search radius to be used for all the points in the neighborhood search.
		*/
		void set_search_radius(const double search_radius);

		/**
		 * @brief Adds a collection of points to search and/or to be found in **variable radius** neighborhood search mode.
		 * 
		 * @warning It is not possible to mix fixed radius sets with variable search radius sets. If one point set has variable search radius, all of them must be declared as variable.
		 * 
		 * @param points_begin Pointer to the points coordinates. Coordinates must be contiguous in memory and in `xyzxyzxyz` layout.
		 * @param radii_begin Pointer to the search radii. Must be contiguous in memory.
		 * @param n_points Number of points in the set.
		 * 
		 * @return Point set id.
		*/
		int add_point_set(const float* points_begin, const float* radii_begin, const int n_points);

		/**
		 * @brief Adds a collection of points to search and/or to be found in **variable radius** neighborhood search mode.
		 * 
		 * @warning TreeNSearch works with type float internally. Declaring point sets as type double will incurr into data copying at the beginning of the neighborhood search process.
		 * @warning It is not possible to mix fixed radius sets with variable search radius sets. If one point set has variable search radius, all of them must be declared as variable.
		 * 
		 * @param points_begin Pointer to the points coordinates. Coordinates must be contiguous in memory and in `xyzxyzxyz` layout.
		 * @param radii_begin Pointer to the search radii. Must be contiguous in memory.
		 * @param n_points Number of points in the set.
		 * 
		 * @return Point set id.
		*/
		int add_point_set(const double* points_begin, const double* radii_begin, const int n_points);

		/**
		 * @brief Updates the coordinates data of a point set in **variable radius** neighborhood search mode.
		 *
		 * @param set_id Id of the point set to be resized.
		 * @param points_begin Pointer to the points coordinates. Coordinates must be contiguous in memory and in `xyzxyzxyz` layout.
		 * @param radii_begin Pointer to the search radii. Must be contiguous in memory.
		 * @param n_points Number of points in the set.
		*/
		void resize_point_set(const int set_id, const float* points_begin, const float* radii_begin, const int n_points);

		/**
		 * @brief Updates the coordinates data of a point set in **variable radius** neighborhood search mode.
		 *
		 * @param set_id Id of the point set to be resized.
		 * @param points_begin Pointer to the points coordinates. Coordinates must be contiguous in memory and in `xyzxyzxyz` layout.
		 * @param radii_begin Pointer to the search radii. Must be contiguous in memory.
		 * @param n_points Number of points in the set.
		*/
		void resize_point_set(const int set_id, const double* points_begin, const double* radii_begin, const int n_points);

		/**
		 * @brief Sets the cell size for the background grid acceleration structure. If not specified, it will be set to 1.5 times the minimum search radius by default.
		 *
		 * @note This is a very relevant parameter for the performance of this method. 
		 * In variable search radius neighborhood search mode, you might want to adapt the cell size of the background grid as your points change their search radii.
		 * 
		 * @param cell_size Cell size for the background grid.
		*/
		void set_cell_size(const float cell_size);

		/**
		 * @brief Sets the cell size for the background grid acceleration structure. If not specified, it will be set to 1.5 times the minimum search radius by default.
		 *
		 * @note This is a very relevant parameter for the performance of this method. 
		 * In variable search radius neighborhood search mode, you might want to adapt the cell size of the background grid as your points change their search radii.
		 * 
		 * @param cell_size Cell size for the background grid.
		*/
		void set_cell_size(const double cell_size);

		/**
		 * @brief Executes the neighborhood search and builds the neighborlists.
		*/
		void run();

		/**
		 * @brief Returns a handle to the neighborlist in set_j of a point_i from point set_i.
		 * 
		 * @param set_i Set id of the point searching.
		 * @param set_j Set id of the neighbors.
		 * @param point_i Index of the point searching.
		 * 
		 * @returns NeighborList, a handle to access the neighbors.
		*/
		NeighborList get_neighborlist(const int set_i, const int set_j, const int point_i) const;

		/**
		 * @brief Loops through the neighbors of a point and calls a callback function.
		 * The function should be inlined and therefore provide the same performance as manually looping through the neighborlist.
		 * The callback function should have a single argument of type `const int` which will be the neighbor index.
		 * 
		 * @param set_i Set id of the point searching.
		 * @param set_j Set id of the neighbors.
		 * @param point_i Index of the point searching.
		 * @param FUNC Callback function with signature `std::function<void(cosnt int)>`.
		*/
		template<typename FUNC>
		inline void for_each_neighbor(const int set_i, const int set_j, const int i, FUNC f);

		/**
		 * @brief Computes the new order of the points according to their Morton curve with respect to the background grid.
		 * Only the new indices are computed, no user data is actually reordered.
		 * To reorder arrays to the new order, use `TreeNsearch.apply_zsort(...)`.
		*/
		void prepare_zsort();

		/**
		* @brief Reorder an array with respect according to the Morton curve with respect to the background grid of a specific point set.
		* Use `TreeNsearch.prepare_zsort()` before calling this function to compute the new index order first, then use `TreeNsearch.apply_zsort()` to reorder one or more arrays.
		* 
		 * @param set_i Set id corresponding to the point set related to the array to be reordered.
		 * @param data_ptr Pointer to the data to be reordered.
		 * @param stride Number of items per point in the data array. 
		 * For example, when `float*` is passed as the begin of a coordinate array, stride should be set to 3 since the reorder has to happen in
		 * groups of 3 numbers at a time `xyz|xyz|xyz`.
		*/
		template<typename T>
		void apply_zsort(const int set_i, T * data_ptr, const int stride = 1) const;

		/**
		 * Activate/Deactivates symmetric neighborhoods in variable radius neighborhood search mode.
		 * When active, a point i will be in the neighborlist of point j if point j is inside the search radius of point i,
		 * even when the situation is not reciprocal.
		 * This option is irrelevant in fixed radius neighborhood search mode.
		 * 
		 * @param activate Whether to activate symmetric search or not.
		*/
		void set_symmetric_search(const bool activate);
		
		// -----------------------------------------------  SECONDARY METHODS  -----------------------------------------------

		/**
		 * @brief Executes the neighborhood search and builds the neighborlists without usin SIMD instructions.
		 * The result is identical to the one obtained by using `TreeNSearch.run()`, but the performance is much worse.
		*/
		void run_scalar();

		/**
		 * @brief Prints some metrics regarding the internal data structures and resulting neighborlists.
		 * Useful for debugging.
		*/
		void print_state() const;

		/**
		 * @brief Returns the number of bytes that the neighborlists take up in memory.
		 * 
		 * @return Number of bytes of the neighboring information.
		*/
		uint64_t get_neighborlist_n_bytes() const;


		// -----------------------------------------------  SETTERS AND GETTERS  -----------------------------------------------

		/**
		 * @brief Activate/deactivate searches for all the combinations possible between the all the point sets.
		 * 
		 * @param active Whether to activate or deactivate all searches.
		*/
		void set_all_searches(const bool active);

		/**
		 * @brief Activate/deactivate that set_i searches for points in set_j.
		 * 
		 * @param set_i Set id of the set searching.
		 * @param set_j Set id of the set to be searched.
		 * @param active Whether to activate or deactivate this particular search.
		*/
		void set_active_search(const int set_i, const int set_j, const bool active = true);

		/**
		 * @brief Activate/deactivate that set_i searches in all sets and/or that set_i is found by all sets.
		 * In the case that set_i should not search in any set, it will also not search in itself.
		 *
		 * @param set_i Set id of the set searching.
		 * @param search_in_all Whether to search in all sets.
		 * @param be_found_by_all Whether to be found by all sets.
		*/
		void set_active_search(const int set_i, const bool search_in_all = true, const bool be_found_by_all = true);

		// Other Setters
		/**
		 * @brief Set the number of threads to be used. If not set, it will use the return of `omp_get_max_threads()`.
		 * 
		 * @param n_threads Number of threads.
		*/
		void set_n_threads(const int n_threads);

		/**
		 * @brief Set the maximum number of points per octree node to stop the recursion and execute the brute force pair-wise distance comparisons.
		 * If nothing is set, TreeNSearch will use the default value of 1000.
		 * 
		 * @param cap Maximum number of points per octree node to stop the recursion.
		*/
		void set_recursion_cap(const int cap);

		/**
		 * @brief Set the minimum number of points for which the octree is built in parallel. For very low point counts, the overhead from the parallel 
		 * tree construction can be significantly slower than the sequential version.
		 * 
		 * @param n_points Minimum number of points for which the octree is built in parallel.
		*/
		void set_n_points_for_parallel_octree(const int n_points = 200000);

		/**
		* @return Number of sets added to TreeNSearch.
		*/
		int get_n_sets() const;

		/**
		* @return Number of threads being used.
		*/
		int get_n_threads() const;

		/**
		* @return Number of points in set_i.
		*/
		int get_n_points_in_set(const int set_i) const;

		/**
		* @return Total number of points in all sets.
		*/
		int get_total_n_points() const;

		/**
		* @return Whether set_i is searching for neighbors in set_j.
		*/
		bool is_search_active(const int set_i, const int set_j) const;

		/**
		* @return Whether there is a set with set_i id.
		*/
		bool does_set_exist(const int set_i) const;

		/**
		* @return Current re-mapping order of the particles of set_i according to the Morton space filling curve with respect to the background grid used by TreeNSearch.
		*/
		const std::vector<int>& get_zsort_order(const int set_i) const;

	private:
		// Helpers
		void _set_up();
		void _new_point_set(const int n_points);
		int _get_set_pair_id(const int set_i, const int set_j) const;

		// Method
		void _check();
		void _clear_neighborlists();

		void _update_world_AABB();
		void _update_world_AABB_simd();

		void _points_to_cells();
		void _points_to_cells_simd();

		void _prepare_root();

		void _build_octree_and_gather_leaves();
		void _run_octree_node(internals::RecursiveOctreeNode & node_buffer, const size_t depth, tf::Subflow* sf);
		void _build_octree_and_gather_leaves_simd();
		void _run_octree_node_simd(internals::RecursiveOctreeNode & node_buffer, const size_t depth, tf::Subflow* sf);

		void _solve_leaves(const bool use_simd);

		void _prepare_brute_force(internals::OctreeNode & leaf_buffer, const int thread_id);
		void _brute_force(internals::OctreeNode & leaf_buffer, const int thread_id);
		bool _prepare_brute_force_simd(internals::OctreeNode & leaf_buffer, const int thread_id);
		void _brute_force_simd(internals::OctreeNode & leaf_buffer, const int thread_id);

		void _compute_zsort_order_notree();

	private:
		// ========================================================================================================
		// ===============================================  FIELDS  ===============================================
		// ========================================================================================================
		
		// Set data
		int n_sets = 0;  // Number of sets
		std::vector<const float*> set_points;  // Points of each set
		std::vector<const float*> set_radii;  // Radii of each set (only initialized in variable search mode)
		std::vector<const double*> set_points_double;  // (In case declared as double)
		std::vector<const double*> set_radii_double;  // (In case declared as double)
		std::vector<int> n_points_per_set;  // Number of points per set
		std::vector<int> set_offsets;  // Assuming all point sets are concatenated, the offset ofa set is the index to the first point
		std::vector<std::vector<float>> set_points_buffers;  // When declared as double, point data is casted and store as float here
		std::vector<std::vector<float>> set_radii_buffers;  // When declared as double, point data is casted and store as float here

		// Neighborhood search
		bool symmetric_search = true;
		bool is_global_search_radius_set = false;
		float global_search_radius_sq = -1.0f;
		float global_search_radius = -1.0f;
		float max_search_radius = -1.0f;
		std::vector<std::vector<int>> active_searches;
		std::vector<std::vector<bool>> active_searches_table;

		// Neighborlist solution
		std::vector<internals::chunked_vector<int, 262144>> thread_neighborlists; // Neighborlist data. Each thread keeps its own lists.
		std::vector<std::vector<int*>> solution_ptr;  // Pointers to the solution neighborlists. Indexing: [set_i*n_sets + set_j][particle_i] -> [n_neighbors, neighbor_0, neighbor_1, ..., neighbor_n]

		// Octree / Grid
		float cell_size = -1.0f;
		float cell_size_inv = -1.0f;
		internals::AABB<float> domain_float;
		float domain_enlargment = 1.1f;
		int avg_points_per_cell = -1;
		int n_points_to_stop_recursion = 1000;
		int n_points_for_parallel_octree = 200000;
		int n_cells_in_node_for_switching_to_sequential = 2000;
		bool parallel_octree = false;
		internals::RecursiveOctreeNode octree_root;
		internals::CellList cells;
		std::vector<internals::CellList> thread_cells;  // Buffer per thread to create cells in parallel
		std::vector<internals::OctreeNode*> leaves;  // List of all octree leaf pointers
		std::vector<std::vector<internals::OctreeNode*>> thread_leaves; // Buffer per thread to gather leaves in parallel

		// Brute force
		std::vector<internals::BruteforceBuffer> thread_bruteforce_buffers;

		// Zsort
		std::vector<std::vector<int>> zsort_set_new_to_old_map;
		bool are_cells_valid = false;

		// Look up tables
		internals::avector<__m256i, 32> shift_lut_32;
		internals::avector<__m128i, 8> shift_lut_8;

		// Misc
		int n_threads = -1;
		tf::Executor* executor = nullptr;
	};



	// ==========================================================  HEADER DEFINITIONS  ==========================================================

	template<typename FUNC>
	inline void TreeNSearch::for_each_neighbor(const int set_i, const int set_j, const int i, FUNC f)
	{
		const tns::NeighborList neighbors = this->get_neighborlist(set_i, set_j, i);
		for (int loc_j = 0; loc_j < neighbors.size(); loc_j++) {
			const int j = neighbors[loc_j];
			f(j);
		}
	}

	template<typename T>
	inline void TreeNSearch::apply_zsort(const int set_i, T* data_ptr, const int stride) const
	{
		if (!this->does_set_exist(set_i)) {
			std::cout << "tns::TreeNSearch::apply_zsort error: set to z_sort does not exit." << std::endl;
			exit(-1);
		}

		if (set_i >= this->zsort_set_new_to_old_map.size()) {
			std::cout << "tns::TreeNSearch::apply_zsort error: no zsort order ready for set_i (" << set_i << ")." << std::endl;
			exit(-1);
		}

		const int n_points = (int)this->get_n_points_in_set(set_i);
		const std::vector<int>& new_to_old_map = this->zsort_set_new_to_old_map[set_i];
		std::vector<T> zsort_swap_buffer(n_points * stride);

		#pragma omp parallel for schedule(static) num_threads(this->n_threads)
		for (int i = 0; i < n_points * stride; i++) {
			zsort_swap_buffer[i] = data_ptr[i];
		}

		if (stride == 1) {
			#pragma omp parallel for schedule(static) num_threads(this->n_threads)
			for (int new_idx = 0; new_idx < n_points; new_idx++) {
				const int old_idx = new_to_old_map[new_idx];
				data_ptr[new_idx] = zsort_swap_buffer[old_idx];
			}
		}
		else {
			#pragma omp parallel for schedule(static) num_threads(this->n_threads)
			for (int new_idx = 0; new_idx < n_points; new_idx++) {
				const int old_idx = new_to_old_map[new_idx];
				for (int j = 0; j < stride; j++) {
					data_ptr[new_idx*stride + j] = zsort_swap_buffer[old_idx*stride + j];
				}
			}
		}
	}
}

