#include "BruteforceNSearch.h"


int BruteforceNSearch::add_point_set(const float* points_begin, const float* radii_begin, const int n_points)
{
	const int set_id = this->n_sets;
	this->n_sets++;

	this->set_points.push_back(points_begin);
	this->set_radii.push_back(std::vector<float>(radii_begin, radii_begin + n_points));
	this->n_points_per_set.push_back(n_points);

	// Search logic. Default: dont search or be searched into
	for (std::vector<bool>& other_sets_searches : this->active_searches_table) {
		other_sets_searches.push_back(false);
	}
	this->active_searches_table.push_back(std::vector<bool>(this->n_sets, false));

	return set_id;
}

int BruteforceNSearch::add_point_set(const float* points_begin, const float radius, const int n_points)
{
	std::vector<float> radii(n_points, radius);
	return this->add_point_set(points_begin, radii.data(), n_points);
}

void BruteforceNSearch::resize_point_set(const int set_i, const float* points_begin, const float* radii_begin, const int n_points)
{
	this->set_points[set_i] = points_begin;
	this->set_radii[set_i] = std::vector<float>(radii_begin, radii_begin + n_points);
	this->n_points_per_set[set_i] = n_points;
}

void BruteforceNSearch::resize_point_set(const int set_i, const float* points_begin, const float radius, const int n_points)
{
	this->set_points[set_i] = points_begin;
	this->set_radii[set_i] = std::vector<float>(n_points, radius);
	this->n_points_per_set[set_i] = n_points;
}

void BruteforceNSearch::set_n_threads(const int n_threads)
{
	this->n_threads = n_threads;
}

void BruteforceNSearch::set_active_search(const int set_i, const int set_j, const bool active)
{
	this->active_searches_table[set_i][set_j] = active;
}

void BruteforceNSearch::set_all_searches(const bool active)
{
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		for (int set_j = 0; set_j < this->n_sets; set_j++) {
			this->active_searches_table[set_i][set_j] = active;
		}
	}
}

void BruteforceNSearch::set_symmetric_search(const bool activate)
{
	this->symmetric_search = activate;
}

void BruteforceNSearch::run()
{
	if (this->n_threads == -1) {
		this->n_threads = omp_get_max_threads();
	}

	this->solution.resize(this->n_sets*this->n_sets);
	
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		for (int set_j = 0; set_j < this->n_sets; set_j++) {
			if (!this->active_searches_table[set_i][set_j]) { continue; }
			this->solution[set_i*this->n_sets + set_j].resize(this->n_points_per_set[set_i]);

			#pragma omp parallel for num_threads(this->n_threads) schedule(static)
			for (int i = 0; i < this->n_points_per_set[set_i]; i++) {
				const float* p = this->set_points[set_i] + 3*i;
				const float r2i = this->set_radii[set_i][i] * this->set_radii[set_i][i];
				std::vector<int>& neighborlist = this->solution[set_i*this->n_sets + set_j][i];
				neighborlist.clear();
				for (int j = 0; j < this->n_points_per_set[set_j]; j++) {
					if (set_i == set_j && i == j) { continue; }
					const float* q = this->set_points[set_j] + 3*j;
					const float d2 = (p[0] - q[0])*(p[0] - q[0]) + (p[1] - q[1])*(p[1] - q[1]) + (p[2] - q[2])*(p[2] - q[2]);

					if (this->symmetric_search) {
						const float r2j = this->set_radii[set_j][j] * this->set_radii[set_j][j];
						if (d2 <= r2i || d2 <= r2j) {
							neighborlist.push_back(j);
						}
					}
					else {
						if (d2 <= r2i) {
							neighborlist.push_back(j);
						}
					}
				}
			}
		}
	}
}

int BruteforceNSearch::get_n_neighbors(const int set_i, const int set_j, const int point_i) const
{
	return (int)this->solution[set_i*this->n_sets + set_j][point_i].size();
}

std::vector<std::vector<int>> BruteforceNSearch::get_neighbor_list_copy(const int set_i, const int set_j) const
{
	return this->solution[set_i * this->n_sets + set_j];
}

bool BruteforceNSearch::compare(const int set_i, const int set_j, const tns::TreeNSearch& tree_n_search, const bool crash_at_first_discrepancy)
{
	const std::vector<std::vector<int>>& this_solution = this->solution[set_i*this->n_sets + set_j];
	std::vector<int> neighbors;

	int total_count = 0;
	int diff_count = 0;
	for (int i = 0; i < this->n_points_per_set[set_i]; i++) {
		total_count += (int)this_solution[i].size();
		diff_count += std::abs((int)this_solution[i].size() - tree_n_search.get_neighborlist(set_i, set_j, i).size());

		if (crash_at_first_discrepancy) {
			const tns::NeighborList neighborlist = tree_n_search.get_neighborlist(set_i, set_j, i);
			neighbors.resize(neighborlist.size());
			for (int j = 0; j < neighborlist.size(); j++) {
				neighbors[j] = neighborlist[j];
			}

			std::sort(neighbors.begin(), neighbors.end());

			if (this_solution[i] != neighbors) {

				std::cout << "Point: " << i << std::endl;
				std::cout << "BruteforceNSearch neighborlist: " << std::endl;
				for (auto j : this_solution[i]) { std::cout << j << ", "; }
				std::cout << std::endl;
				std::cout << "TreeNSearch neighborlist: " << std::endl;
				for (auto j : neighbors) { std::cout << j << ", "; }
				std::cout << std::endl;

				std::cout << "Relative distnace between the points wrt search radius: " << std::endl;
				std::vector<int> diff;
				std::set_difference(this_solution[i].begin(), this_solution[i].end(), neighbors.begin(), neighbors.end(), std::back_inserter(diff));
				std::set_difference(neighbors.begin(), neighbors.end(), this_solution[i].begin(), this_solution[i].end(), std::back_inserter(diff));
				for (int j : diff) {
					const float* p = this->set_points[set_i] + 3*i;
					const float* q = this->set_points[set_j] + 3*j;
					std::cout << i << " <- " << j << ": ";
					const float d = std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
					std::cout << d << "(" << 100.0 * (d - this->set_radii[set_i][i]) / this->set_radii[set_i][i] << "%)" << std::endl;
				}

				tree_n_search.print_state();
				exit(-1);
			}
		}
	}
	//std::cout << "Error: " << diff_count << " out of " << total_count << " neighbors are wrong (" << 100.0f * ((float)diff_count / (float)total_count) << "%)." << std::endl;
	return diff_count == 0;
}

bool BruteforceNSearch::compare(const tns::TreeNSearch& tree_n_search, const bool crash_at_first_discrepancy)
{
	bool success = true;
	for (int set_i = 0; set_i < tree_n_search.get_n_sets(); set_i++) {
		for (int set_j = 0; set_j < tree_n_search.get_n_sets(); set_j++) {
			if (tree_n_search.is_search_active(set_i, set_j)) {
				success = success && this->compare(set_i, set_j, tree_n_search, crash_at_first_discrepancy);
			}
		}
	}
	return success;
}

void BruteforceNSearch::check_for_symmetry()
{
	int total_non_symmetric = 0;
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		for (int set_j = 0; set_j < this->n_sets; set_j++) {
			if (!this->active_searches_table[set_i][set_j]) { continue; }
			if (!this->active_searches_table[set_j][set_i]) { continue; }

			#pragma omp parallel for num_threads(this->n_threads) schedule(static) reduction(+:total_non_symmetric)
			for (int i = 0; i < this->n_points_per_set[set_i]; i++) {
				for (const int j : this->solution[set_i * this->n_sets + set_j][i]) {
					bool found = false;
					for (const int k : this->solution[set_j * this->n_sets + set_i][j]) {
						if (k == i) {
							found = true;
							break;
						}
					}
					if (!found) {
						total_non_symmetric++;
					}
				}
			}

		}
	}
	std::cout << "Non symmetric neighbors found: " << total_non_symmetric << std::endl;
}
