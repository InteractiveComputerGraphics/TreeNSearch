#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <omp.h>

#include <TreeNSearch>

/**
 * Simple brute force neighborhood search with the same features than TreeNSearch 
 * to compare their results. 
 * BruteforceNSearch has no acceleration structure and no SIMD optimizations. It 
 * compares all point pairs to build the neighborlists.
*/
class BruteforceNSearch
{
public:
	/* Fields */
	int n_sets = 0;
	std::vector<const float*> set_points;
	std::vector<std::vector<float>> set_radii;
	std::vector<int> n_points_per_set;
	std::vector<std::vector<int>> active_searches;
	std::vector<std::vector<bool>> active_searches_table;
	int n_threads = -1;
	std::vector<std::vector<std::vector<int>>> solution; // [set_i * n_sets + set_j][point_i][neighbor_j]
	bool symmetric_search = true;

	/* Methods */
	BruteforceNSearch() = default;
	~BruteforceNSearch() = default;

	int add_point_set(const float* points_begin, const float* radii_begin, const int n_points);
	int add_point_set(const float* points_begin, const float radius, const int n_points);
	void resize_point_set(const int set_i, const float* points_begin, const float* radii_begin, const int n_points);
	void resize_point_set(const int set_i, const float* points_begin, const float radius, const int n_points);
	void set_n_threads(const int n_threads);
	void set_active_search(const int set_i, const int set_j, const bool active = true); // Activate/Deactivate unidirectional search "set_i searches into set_j"
	void set_all_searches(const bool active); // Activate/Deactivate all point set pairs.
	void set_symmetric_search(const bool activate);

	void run();
	int get_n_neighbors(const int set_i, const int set_j, const int point_i) const;
	std::vector<std::vector<int>> get_neighbor_list_copy(const int set_i, const int set_j) const;

	bool compare(const int set_i, const int set_j, const tns::TreeNSearch& tree_n_search, const bool crash_at_first_discrepancy = false);
	bool compare(const tns::TreeNSearch& tree_n_search, const bool crash_at_first_discrepancy = false);
	void check_for_symmetry();
};
