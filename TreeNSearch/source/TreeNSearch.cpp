#include "TreeNSearch.h"

#include <libmorton/morton.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/sort.hpp>

#include "internals/shuffle_lut.h"

using namespace tns::internals;


void tns::TreeNSearch::set_search_radius(const float search_radius)
{
	if (this->set_radii.size() > 0) {
		std::cout << "tns::TreeNSearch::set_search_radius error: Cannot set a global search radius if a set with a radii array was already added." << std::endl;
		exit(-1);
	}

	this->is_global_search_radius_set = true;
	this->global_search_radius = search_radius;
	this->global_search_radius_sq = search_radius * search_radius;
}
void tns::TreeNSearch::set_search_radius(const double search_radius)
{
	this->set_search_radius((float)search_radius);
}
int tns::TreeNSearch::add_point_set(const float* points_begin, const int n_points)
{
	this->_new_point_set(n_points);
	this->set_points.push_back(points_begin);
	this->set_points_double.push_back(nullptr);
	return this->n_sets - 1;
}
int tns::TreeNSearch::add_point_set(const double* points_begin, const int n_points)
{
	this->_new_point_set(n_points);
	this->set_points.push_back(nullptr);
	this->set_points_double.push_back(points_begin);
	return this->n_sets - 1;
}
int tns::TreeNSearch::add_point_set(const float* points_begin, const float* radii_begin, const int n_points)
{
	this->_new_point_set(n_points);
	this->set_points.push_back(points_begin);
	this->set_radii.push_back(radii_begin);
	this->set_points_double.push_back(nullptr);
	this->set_radii_double.push_back(nullptr);
	return this->n_sets - 1;
}
int tns::TreeNSearch::add_point_set(const double* points_begin, const double* radii_begin, const int n_points)
{
	this->_new_point_set(n_points);
	this->set_points.push_back(nullptr);
	this->set_radii.push_back(nullptr);
	this->set_points_double.push_back(points_begin);
	this->set_radii_double.push_back(radii_begin);
	return this->n_sets - 1;
}
void tns::TreeNSearch::resize_point_set(const int set_id, const float* points_begin, const float* radii_begin, const int n_points)
{
	if (!this->does_set_exist(set_id)) {
		std::cout << "TreeNSearch::resize_point_set error: Cannot resize a set that was not previously added." << std::endl;
		exit(-1);
	}
	if (this->set_radii.size() == 0) {
		std::cout << "TreeNSearch::resize_point_set error: Cannot resize a set with a radii array if it previously didn't have one." << std::endl;
		exit(-1);
	}
	if (this->set_points[set_id] == points_begin && this->set_radii[set_id] == radii_begin && this->get_n_points_in_set(set_id) == n_points) {
		return;
	}
	this->resize_point_set(set_id, points_begin, n_points);
	this->set_radii[set_id] = radii_begin;
	this->set_radii_double[set_id] = nullptr;
}
void tns::TreeNSearch::resize_point_set(const int set_id, const double* points_begin, const double* radii_begin, const int n_points)
{
	if (!this->does_set_exist(set_id)) {
		std::cout << "TreeNSearch::resize_point_set error: Cannot resize a set that was not previously added." << std::endl;
		exit(-1);
	}
	if (this->set_points_double[set_id] == points_begin && this->set_radii_double[set_id] == radii_begin && this->get_n_points_in_set(set_id) == n_points) {
		return;
	}
	const float* points_begin_float = nullptr;
	const float* radii_begin_float = nullptr;
	this->resize_point_set(set_id, points_begin_float, radii_begin_float, n_points);
	this->set_points[set_id] = nullptr;
	this->set_radii[set_id] = nullptr;
	this->set_points_double[set_id] = points_begin;
	this->set_radii_double[set_id] = radii_begin;
}
void tns::TreeNSearch::resize_point_set(const int set_id, const float* points_begin, const int n_points)
{
	if (!this->does_set_exist(set_id)) {
		std::cout << "TreeNSearch::resize_point_set error: Cannot resize a set that was not previously added." << std::endl;
		exit(-1);
	}
	if (this->set_points[set_id] == points_begin && this->get_n_points_in_set(set_id) == n_points) {
		return;
	}
	this->n_points_per_set[set_id] = n_points;
	this->set_points[set_id] = points_begin;
	this->set_points_double[set_id] = nullptr;

	// Recompute set offsets
	std::inclusive_scan(this->n_points_per_set.begin(), this->n_points_per_set.end(), this->set_offsets.begin() + 1);

	// Prev octree is invalid for zsort
	this->are_cells_valid = false;
}
void tns::TreeNSearch::resize_point_set(const int set_id, const double* points_begin, const int n_points)
{
	if (!this->does_set_exist(set_id)) {
		std::cout << "TreeNSearch::resize_point_set error: Cannot resize a set that was not previously added." << std::endl;
		exit(-1);
	}
	if (this->set_points_double[set_id] == points_begin && this->get_n_points_in_set(set_id) == n_points) {
		return;
	}
	const float* points_begin_float = nullptr;
	this->resize_point_set(set_id, points_begin_float, n_points);
	this->set_points[set_id] = nullptr;
	this->set_points_double[set_id] = points_begin;
}
void tns::TreeNSearch::set_cell_size(const double cell_size)
{
	this->set_cell_size((float)cell_size);
}
void tns::TreeNSearch::run()
{
	if (this->get_total_n_points() < this->number_of_too_few_particles) {
		this->run_scalar();
	}
	else {
		// Run in SIMD mode
		this->_set_up();
		this->_check();
		this->_clear_neighborlists();
		this->_update_world_AABB_simd();
		this->_points_to_cells_simd();
		this->_build_octree_and_gather_leaves_simd();
		this->_solve_leaves(/* use_simd = */ true);
		this->are_cells_valid = true;
	}
}
void tns::TreeNSearch::run_scalar()
{
	this->_set_up();
	this->_check();
	this->_clear_neighborlists();
	this->_update_world_AABB();
	this->_points_to_cells();
	this->_build_octree_and_gather_leaves();
	this->_solve_leaves(/* use_simd = */ false);
	this->are_cells_valid = true;
}
void tns::TreeNSearch::set_recursion_cap(const int cap)
{
	this->n_points_to_stop_recursion = cap;
}
void tns::TreeNSearch::set_n_threads(const int n_threads)
{
	this->n_threads = n_threads;
}
void tns::TreeNSearch::set_symmetric_search(const bool activate)
{
	this->symmetric_search = activate;
}
void tns::TreeNSearch::set_cell_size(const float cell_size)
{
	this->cell_size = cell_size;
	this->cell_size_inv = 1.0f / cell_size;
}
int tns::TreeNSearch::_get_set_pair_id(const int set_i, const int set_j) const
{
	return set_i * this->n_sets + set_j;
}
void tns::TreeNSearch::set_n_points_for_parallel_octree(const int n_points)
{
	this->n_points_for_parallel_octree = n_points;
}
int tns::TreeNSearch::get_n_sets() const
{
	return this->n_sets;
}
int tns::TreeNSearch::get_n_threads() const
{
	return this->n_threads;
}
int tns::TreeNSearch::get_n_points_in_set(const int set_i) const
{
	return this->n_points_per_set[set_i];
}
int tns::TreeNSearch::get_total_n_points() const
{
	int total = 0;
	for (const int n : this->n_points_per_set) {
		total += n;
	}
	return total;
}
bool tns::TreeNSearch::is_search_active(const int set_i, const int set_j) const
{
	return this->active_searches_table[set_i][set_j];
}
bool tns::TreeNSearch::does_set_exist(const int set_i) const
{
	return set_i < this->n_sets;
}
void tns::TreeNSearch::set_active_search(const int set_i, const int set_j, const bool active)
{
	this->active_searches_table[set_i][set_j] = active;
}
void tns::TreeNSearch::set_active_search(const int set_i, const bool search_neighbors, const bool find_neighbors)
{
	// Note: The order is important: if search_neighbors is false, it will overwrite find_neighbors to avoid search on itself.
	for (int set_j = 0; set_j < this->n_sets; set_j++) {
		this->active_searches_table[set_j][set_i] = find_neighbors;
	}
	for (int set_j = 0; set_j < this->n_sets; set_j++) {
		this->active_searches_table[set_i][set_j] = search_neighbors;
	}
}
void tns::TreeNSearch::set_all_searches(const bool active)
{
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		for (int set_j = 0; set_j < this->n_sets; set_j++) {
			this->active_searches_table[set_i][set_j] = active;
		}
	}
}
tns::NeighborList tns::TreeNSearch::get_neighborlist(const int set_i, const int set_j, const int point_i) const
{
	assert(this->does_set_exist(set_i) && "TreeNSearch::get_neighborlist error: Set does not exist.");
	assert(this->does_set_exist(set_j) && "TreeNSearch::get_neighborlist error: Set does not exist.");
	assert(this->is_search_active(set_i, set_j) && "TreeNSearch::get_neighborlist error: Set pair not active.");
	assert(point_i < this->get_n_points_in_set(set_i) && "TreeNSearch::get_neighborlist error: point not in set.");
	
	return NeighborList(this->solution_ptr[this->_get_set_pair_id(set_i, set_j)][point_i]);
}
const std::vector<int>& tns::TreeNSearch::get_zsort_order(const int set_i) const
{
	return this->zsort_set_new_to_old_map[set_i];
}
uint64_t tns::TreeNSearch::get_neighborlist_n_bytes() const
{
	uint64_t n = 0;
	for (const auto& v : this->thread_neighborlists) {
		n += v.n_bytes();
	}
	return n;
}

void tns::TreeNSearch::_set_up()
{
	// Default number of threads
	if (this->n_threads == -1) {
		this->n_threads = omp_get_max_threads();
	}

	// Copy point data to float if declared as double
	/*
		Double type is supported by casting and writting all `const double*` user data into 
		internal `std::vector<float>` and setting those pointers as the ones to work with.
	*/
	this->set_points_buffers.resize(this->n_sets);
	this->set_radii_buffers.resize(this->n_sets);
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		if (this->set_points_double[set_i] != nullptr) {
			const int n = this->get_n_points_in_set(set_i);

			this->set_points_buffers[set_i].resize(3*n);
			#pragma omp parallel for schedule(static) num_threads(this->n_threads)
			for (int i = 0; i < 3*n; i++) {
				this->set_points_buffers[set_i][i] = (float)this->set_points_double[set_i][i];
			}
			this->set_points[set_i] = this->set_points_buffers[set_i].data();

			if (this->set_radii.size() > 0) {
				this->set_radii_buffers[set_i].resize(n);
				#pragma omp parallel for schedule(static) num_threads(this->n_threads)
				for (int i = 0; i < n; i++) {
					this->set_radii_buffers[set_i][i] = (float)this->set_radii_double[set_i][i];
				}
				this->set_radii[set_i] = this->set_radii_buffers[set_i].data();
			}
		}
	}

	// Default cell size
	if (this->cell_size < 0.0f) {
		if (this->is_global_search_radius_set) {
			this->set_cell_size(1.5f*this->global_search_radius);
		}
		else {
			float min_radius = std::numeric_limits<float>::max();
			for (int set_i = 0; set_i < this->n_sets; set_i++) {
				min_radius = std::min(min_radius, *std::min_element(this->set_radii[set_i], this->set_radii[set_i] + this->get_n_points_in_set(set_i)));
			}
			this->set_cell_size(1.5f*min_radius);
		}
	}


	// Allocate thread bruteforce buffers
	this->thread_bruteforce_buffers.resize(this->n_threads);

	// Active search table to vector<vector<int>>
	this->active_searches.resize(this->n_sets);
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		this->active_searches[set_i].clear();
		for (int set_j = 0; set_j < this->n_sets; set_j++) {
			if (this->active_searches_table[set_i][set_j]) {
				this->active_searches[set_i].push_back(set_j);
			}
		}
	}

	// Make SIMD look up tables aligned
	if (this->shift_lut_32.size() == 0) {
		this->shift_lut_32.resize(256);
		for (int i = 0; i < 256; i++) {
			this->shift_lut_32[i] = _mm256_loadu_si256((__m256i*) & tns::internals::shift_lut_32[i][0]);
		}
		this->shift_lut_8.resize(256);
		for (int i = 0; i < 256; i++) {
			this->shift_lut_8[i] = _mm_loadu_si128((__m128i*) & tns::internals::shift_lut_8[i][0]);
		}

	}
}
void tns::TreeNSearch::_new_point_set(const int n_points)
{
	const int set_id = this->n_sets;
	this->n_sets++;

	if (set_id == 0) {
		this->set_offsets.push_back(0);
	}
	this->set_offsets.push_back(this->set_offsets.back() + n_points);
	this->n_points_per_set.push_back(n_points);

	// Search logic. Default: dont search or be searched into
	for (std::vector<bool>& other_sets_searches : this->active_searches_table) {
		other_sets_searches.push_back(false);
	}
	this->active_searches_table.push_back(std::vector<bool>(this->n_sets, false));

	// Prev octree is invalid for zsort
	this->are_cells_valid = false;
}
void tns::TreeNSearch::_check()
{
	if (this->cell_size <= 0.0) {
		std::cout << "TreeNSearch error: cell_size is not set. Use TreeNSearch::set_cell_size()." << std::endl;
		exit(-1);
	}

	if (this->n_points_to_stop_recursion <= 0) {
		std::cout << "TreeNSearch error: n_points_to_stop_recursion <= 0." << std::endl;
		exit(-1);
	}

	if (this->is_global_search_radius_set && this->global_search_radius <= 0.0f) {
		std::cout << "TreeNSearch error: global_search_radius <= 0." << std::endl;
		exit(-1);
	}

	if (this->is_global_search_radius_set && this->set_radii.size() > 0) {
		std::cout << "TreeNSearch error: global search radius and per-point variable search radii specified." << std::endl;
		exit(-1);
	}

	if (!this->is_global_search_radius_set && this->set_radii.size() != this->n_sets) {
		std::cout << "TreeNSearch error: not all point sets have per-point search radius specified." << std::endl;
		exit(-1);
	}
}
void tns::TreeNSearch::_clear_neighborlists()
{
	// Solution pointers
	this->solution_ptr.resize(this->n_sets * this->n_sets);
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		for (int set_j = 0; set_j < this->n_sets; set_j++) {
			if (this->active_searches_table[set_i][set_j]) {
				this->solution_ptr[this->_get_set_pair_id(set_i, set_j)].resize(this->get_n_points_in_set(set_i));
			}
			else {
				this->solution_ptr[this->_get_set_pair_id(set_i, set_j)] = std::vector<int*>(); // deallocate
			}
		}
	}

	// Neighborlist integer data
	this->thread_neighborlists.resize(this->n_threads);
	for (auto& neighborlists : this->thread_neighborlists) {
		neighborlists.clear();
	}
}

void tns::TreeNSearch::_update_world_AABB()
{
	/*
		Updating the world AABB changes the zsort alignment significatly slowing 
		down the tree construction (since points are not almost-zsorted anymore).
		Therefore, we enlarge the world AABB by small factor to be able to reuse it
		during multiple time steps.

		In the end, we make the world perfectly cubical for agreement between zsort and octree.
	*/
	constexpr float MAXf = std::numeric_limits<float>::max();
	constexpr float MINf = std::numeric_limits<float>::lowest();
	std::array<float, 3> new_bottom = { MAXf, MAXf, MAXf };
	std::array<float, 3> new_top = { MINf, MINf, MINf };

	// Find points AABB
	// Note: Paralelized across points and sets
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		const float* points = this->set_points[set_i];
		const int n_points = this->get_n_points_in_set(set_i);

		#pragma omp parallel num_threads(this->n_threads)
		{
			const int thread_id = omp_get_thread_num();
			std::array<float, 3> thread_bottom = { MAXf, MAXf, MAXf };
			std::array<float, 3> thread_top = { MINf, MINf, MINf };

			const int chunksize = n_points / this->n_threads;
			const int begin = thread_id * chunksize;
			const int end = (thread_id == this->n_threads - 1) ? n_points : (thread_id + 1) * chunksize;

			// Ignore empty chunks
			const int thread_n_points = end - begin;
			if (thread_n_points > 0) {

				for (int point_i = begin; point_i < end; point_i++) {
					thread_bottom[0] = std::min(thread_bottom[0], points[3 * point_i]);
					thread_bottom[1] = std::min(thread_bottom[1], points[3 * point_i + 1]);
					thread_bottom[2] = std::min(thread_bottom[2], points[3 * point_i + 2]);

					thread_top[0] = std::max(thread_top[0], points[3 * point_i]);
					thread_top[1] = std::max(thread_top[1], points[3 * point_i + 1]);
					thread_top[2] = std::max(thread_top[2], points[3 * point_i + 2]);
				}

				#pragma omp critical
				{
					new_bottom[0] = std::min(new_bottom[0], thread_bottom[0]);
					new_bottom[1] = std::min(new_bottom[1], thread_bottom[1]);
					new_bottom[2] = std::min(new_bottom[2], thread_bottom[2]);

					new_top[0] = std::max(new_top[0], thread_top[0]);
					new_top[1] = std::max(new_top[1], thread_top[1]);
					new_top[2] = std::max(new_top[2], thread_top[2]);
				}
			}
		}
	}

	// If the new AABB is fully contained in the prev one, we don't change it
	std::array<float, 3>& bottom = this->domain_float.bottom;
	std::array<float, 3>& top = this->domain_float.top;
	if (bottom[0] <= new_bottom[0] && new_top[0] <= top[0] &&
		bottom[1] <= new_bottom[1] && new_top[1] <= top[1] &&
		bottom[2] <= new_bottom[2] && new_top[2] <= top[2]) {

		return;
	}
	else {
		bottom = new_bottom;
		top = new_top;
		this->are_cells_valid = false; // Old cells no longer valid
	}

	// Make the AABB cubical
	//// World center
	std::array<float, 3> center;
	center[0] = 0.5f * (top[0] + bottom[0]);
	center[1] = 0.5f * (top[1] + bottom[1]);
	center[2] = 0.5f * (top[2] + bottom[2]);

	//// World length
	float length = 0.0f;
	for (int dim = 0; dim < 3; dim++) {
		length = std::max(length, top[dim] - bottom[dim]);
	}
	length += 100.0f * std::numeric_limits<float>::epsilon();
	length *= this->domain_enlargment;

	//// length must be power of two multiple of cell size so the grid and the octree overlay perfectly
	const int n_cells = (int)(length / this->cell_size) + 1;
	int n_cells_pow2 = 1;
	while (n_cells_pow2 < n_cells) { n_cells_pow2 *= 2; }
	length = this->cell_size * n_cells_pow2;

	if (n_cells_pow2 > 32768) {
		std::cout << "TreeNSearch error: Max allowed cells per dimension is 32768 (2^15)." << std::endl;
		std::cout << "                   Use TreeNSearch.set_cell_size() to set a larger value." << std::endl;
		this->print_state();
		exit(-1);
	}

	//// World AABB
	for (int dim = 0; dim < 3; dim++) {
		bottom[dim] = center[dim] - 0.5f * length;
		top[dim] = center[dim] + 0.5f * length;
	}
}
void tns::TreeNSearch::_update_world_AABB_simd()
{
	/*
		In the SIMD version of _update_world_AABB() two points coordinates are processed 
		at the same time to find the max and min extend of the world AABB.
	*/
	constexpr float MAXf = std::numeric_limits<float>::max();
	constexpr float MINf = std::numeric_limits<float>::lowest();

	// Tight world domain computation (SIMD)
	std::vector<std::array<__m256, 2>> thread_domain_simd(this->n_threads);
	for (std::array<__m256, 2>&domain : thread_domain_simd) {
		domain[0] = _mm256_set1_ps(MAXf);
		domain[1] = _mm256_set1_ps(MINf);
	}

	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		const float* points = this->set_points[set_i];
		const int n_points = this->get_n_points_in_set(set_i);

		#pragma omp parallel num_threads(this->n_threads)
		{
			const int thread_id = omp_get_thread_num();
			__m256 b_simd = _mm256_set1_ps(MAXf);
			__m256 t_simd = _mm256_set1_ps(MINf);

			const int chunksize = n_points / this->n_threads;
			const int begin = thread_id * chunksize;
			const int end = (thread_id == this->n_threads - 1) ? n_points : (thread_id + 1) * chunksize;

			// Ignore empty chunks
			const int thread_n_points = end - begin;
			if (thread_n_points > 0) {

				// We compute two points at the same time: [x, y, z, z, y, z, ., .]
				// Therefore, to not overflow, the remainder is 3
				for (int point_i = begin; point_i < end - 3; point_i += 2) {
					const __m256 p = _mm256_loadu_ps(&points[3 * point_i]);
					b_simd = _mm256_min_ps(b_simd, p);
					t_simd = _mm256_max_ps(t_simd, p);
				}

				for (int point_i = end - 3; point_i < end; point_i++) {
					const int base = 3 * point_i;
					const __m256 p = _mm256_setr_ps(points[base + 0], points[base + 1], points[base + 2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
					b_simd = _mm256_min_ps(b_simd, p);
					t_simd = _mm256_max_ps(t_simd, p);
				}

				thread_domain_simd[thread_id][0] = _mm256_min_ps(thread_domain_simd[thread_id][0], b_simd);
				thread_domain_simd[thread_id][1] = _mm256_max_ps(thread_domain_simd[thread_id][1], t_simd);
			}

			// AABB is the first point
			else {
				const __m256 p = _mm256_loadu_ps(&points[3 * begin]);
				thread_domain_simd[thread_id][0] = _mm256_min_ps(thread_domain_simd[thread_id][0], p);
				thread_domain_simd[thread_id][1] = _mm256_max_ps(thread_domain_simd[thread_id][1], p);
			}
		}
	}

	//// Reduce results from each thread to the result in thread 0
	for (int thread_id = 1; thread_id < this->n_threads; thread_id++) {
		thread_domain_simd[0][0] = _mm256_min_ps(thread_domain_simd[0][0], thread_domain_simd[thread_id][0]);
		thread_domain_simd[0][1] = _mm256_max_ps(thread_domain_simd[0][1], thread_domain_simd[thread_id][1]);
	}

	//// Reduce the 2 point results from thread 0 into a single AABB
	std::array<float, 3> new_bottom;
	std::array<float, 3> new_top;
	const float* b = reinterpret_cast<const float*>(&thread_domain_simd[0][0]);
	const float* t = reinterpret_cast<const float*>(&thread_domain_simd[0][1]);
	for (int dim = 0; dim < 3; dim++) {
		new_bottom[dim] = std::min(b[dim], b[3 + dim]);
		new_top[dim] = std::max(t[dim], t[3 + dim]);
	}

	// (From here on is the same as the scalar version)

	// If the new AABB is fully contained in the prev one, we don't change it
	std::array<float, 3>& bottom = this->domain_float.bottom;
	std::array<float, 3>& top = this->domain_float.top;
	if (bottom[0] <= new_bottom[0] && new_top[0] <= top[0] &&
		bottom[1] <= new_bottom[1] && new_top[1] <= top[1] &&
		bottom[2] <= new_bottom[2] && new_top[2] <= top[2]) {

		return;
	}
	else {
		bottom = new_bottom;
		top = new_top;
		this->are_cells_valid = false; // Old cells no longer valid
	}

	// Make the AABB cubical
	//// World center
	std::array<float, 3> center;
	center[0] = 0.5f * (top[0] + bottom[0]);
	center[1] = 0.5f * (top[1] + bottom[1]);
	center[2] = 0.5f * (top[2] + bottom[2]);

	//// World length
	float length = 0.0f;
	for (int dim = 0; dim < 3; dim++) {
		length = std::max(length, top[dim] - bottom[dim]);
	}
	length += 100.0f * std::numeric_limits<float>::epsilon();
	length *= this->domain_enlargment;

	//// length must be power of two multiple of cell size so the grid and the octree overlay perfectly
	const int n_cells = (int)(length / this->cell_size) + 1;
	int n_cells_pow2 = 1;
	while (n_cells_pow2 < n_cells) { n_cells_pow2 *= 2; }
	length = this->cell_size * n_cells_pow2;

	if (n_cells_pow2 > 32768) {
		std::cout << "TreeNSearch error: Max allowed cells per dimension is 32768 (2^15)." << std::endl;
		std::cout << "                   Use TreeNSearch.set_cell_size() to set a larger value." << std::endl;
		this->print_state();
		exit(-1);
	}

	//// World AABB
	for (int dim = 0; dim < 3; dim++) {
		bottom[dim] = center[dim] - 0.5f * length;
		top[dim] = center[dim] + 0.5f * length;
	}
}
void tns::TreeNSearch::_points_to_cells()
{
	/*
		Classify points into cells.

		Linearly loop over the points and create a new cell every time a new point does not belong to
		the cell of the previous point.

		In the variable radius neighborhood search mode, each cell must know the largest search radius
		of the points it contains.
	*/

	// Compute cells in thread buffers
	// Note: We distribute the work to the threads from all the point sets simulataneously
	this->thread_cells.resize(this->n_threads);
	#pragma omp parallel num_threads(this->n_threads)
	{
		const int thread_id = omp_get_thread_num();
		CellList& cells = this->thread_cells[thread_id];
		cells.init_with_at_least_size(std::max(10000, (int)(0.1*this->get_total_n_points())));
		int cell_i = 0;

		const int n_points = this->get_total_n_points();
		const int chunksize = n_points / n_threads;
		const int begin_thread_point = chunksize * thread_id;
		const int end_thread_point = (thread_id == n_threads - 1) ? n_points : chunksize * (thread_id + 1);

		// Do not process if there are no points
		const int thread_n_points = end_thread_point - begin_thread_point;
		if (thread_n_points > 0) {

			// Find the set of the first point
			int begin_set = 0;
			while (this->set_offsets[begin_set + 1] < begin_thread_point) {
				begin_set++;
			}

			// Find the end set (set of the last point + 1)
			//const int last_point = end_thread_point - 1;
			int end_set = begin_set;
			while (this->set_offsets[end_set + 1] < end_thread_point) {
				end_set++;
			}
			end_set++;

			// For each set
			for (int set_i = begin_set; set_i < end_set; set_i++) {

				const int set_offset = this->set_offsets[set_i];
				const int end_set_point = this->set_offsets[set_i + 1];

				// Clip the range of the points to process with the set range
				const int begin_point = std::max(set_offset, begin_thread_point);
				const int end_point = std::min(end_set_point, end_thread_point);

				const float* points = this->set_points[set_i];

				// Insert first cell
				// Note: For every set, we create a new cell to avoid points from different sets sharing the same cell
				const int point_idx = begin_point - set_offset;
				const float* p = points + 3 * point_idx;
				cells.offsets[cell_i] = begin_point;
				cells.i[cell_i] = (uint16_t)((p[0] - this->domain_float.bottom[0]) * this->cell_size_inv);
				cells.j[cell_i] = (uint16_t)((p[1] - this->domain_float.bottom[1]) * this->cell_size_inv);
				cells.k[cell_i] = (uint16_t)((p[2] - this->domain_float.bottom[2]) * this->cell_size_inv);
				cell_i++;

				// cells computation
				int point_i = begin_point + 1;
				for (; point_i < end_point; point_i++) {

					if (cells.capacity <= cell_i) {
						cells.grow_while_keeping_data(2 * cells.capacity);
					}

					// Load
					const int point_idx = point_i - set_offset;
					const float* p = points + 3 * point_idx;

					// Cell coords
					const uint16_t i = (uint16_t)((p[0] - this->domain_float.bottom[0]) * this->cell_size_inv);
					const uint16_t j = (uint16_t)((p[1] - this->domain_float.bottom[1]) * this->cell_size_inv);
					const uint16_t k = (uint16_t)((p[2] - this->domain_float.bottom[2]) * this->cell_size_inv);

					// Comparison
					const bool different = cells.i[cell_i - 1] != i || cells.j[cell_i - 1] != j || cells.k[cell_i - 1] != k;
					if (different) {
						cells.offsets[cell_i] = point_i;
						cells.i[cell_i] = i;
						cells.j[cell_i] = j;
						cells.k[cell_i] = k;
						cell_i++;
					}
				}
			}
		}

		cells.n_cells = cell_i;
	}

	// Merge thread cell lists into the global vector
	//// Thread cells offsets
	std::vector<int> thread_cells_offsets(this->n_threads + 1);
	thread_cells_offsets[0] = 0;
	for (int thread_id = 0; thread_id < this->n_threads; thread_id++) {
		thread_cells_offsets[thread_id + 1] = thread_cells_offsets[thread_id] + this->thread_cells[thread_id].n_cells;
	}
	const int n_cells = thread_cells_offsets.back();

	//// Parallel write
	if (this->cells.capacity < n_cells) {
		this->cells.init_with_at_least_size((int)(1.1*n_cells));
	}
	#pragma omp parallel num_threads(this->n_threads)
	{
		const int thread_id = omp_get_thread_num();
		const CellList& thread_cells = this->thread_cells[thread_id];
		const int begin = thread_cells_offsets[thread_id];
		const int n = thread_cells_offsets[thread_id + 1] - thread_cells_offsets[thread_id];

		memcpy(&this->cells.i[begin], thread_cells.i, sizeof(uint16_t) * n);
		memcpy(&this->cells.j[begin], thread_cells.j, sizeof(uint16_t) * n);
		memcpy(&this->cells.k[begin], thread_cells.k, sizeof(uint16_t) * n);
		memcpy(&this->cells.offsets[begin], thread_cells.offsets, sizeof(int) * n);
	}
	this->cells.n_cells = n_cells;
	this->cells.offsets[n_cells] = this->get_total_n_points();
	this->avg_points_per_cell = this->get_total_n_points() / n_cells;


	// Variable radius
	if (!this->is_global_search_radius_set) {
		this->cells.init_radii();

		float max_radius = 0.0f;
		int begin_set_cell = 0;
		int end_set_cell = 0;
		for (int set_i = 0; set_i < this->n_sets; set_i++) {
			const int set_offset = this->set_offsets[set_i];
			const float* set_radii = this->set_radii[set_i];

			if (set_i + 1 == this->n_sets) {
				end_set_cell = this->cells.n_cells;
			}
			else {
				end_set_cell = (int)std::distance(this->cells.offsets, std::lower_bound(this->cells.offsets + begin_set_cell, this->cells.offsets + this->cells.n_cells, this->set_offsets[set_i + 1]));
			}

			#pragma omp parallel num_threads(this->n_threads)
			{
				const int thread_id = omp_get_thread_num();

				const int n = end_set_cell - begin_set_cell;
				const int chunksize = n / this->n_threads;
				const int begin = thread_id * chunksize;
				const int end = (thread_id + 1 == this->n_threads) ? n : (thread_id + 1) * chunksize;

				float thread_max_radius = 0.0f;

				for (int cell_i = begin_set_cell + begin; cell_i < begin_set_cell + end; cell_i++) {

					const int begin_point = this->cells.offsets[cell_i] - set_offset;
					const int end_point = this->cells.offsets[cell_i + 1] - set_offset;

					float cell_radius = set_radii[begin_point];
					for (int point_i = begin_point + 1; point_i < end_point; point_i++) {
						cell_radius = std::max(cell_radius, set_radii[point_i]);
					}
					this->cells.radii[cell_i] = cell_radius;
					thread_max_radius = std::max(thread_max_radius, cell_radius);
				}

				#pragma omp critical
				{
					max_radius = std::max(max_radius, thread_max_radius);
				}
			}

			begin_set_cell = end_set_cell;
		}
		this->max_search_radius = max_radius;
	}
	else {
		this->max_search_radius = this->global_search_radius;
	}
}
void tns::TreeNSearch::_points_to_cells_simd()
{
	/*
		In this SIMD implementation of _points_to_cells() eight points are processed at the
		same time. The eight ijk cell coordiantes are computed and points have different
		cell indices are identified. Then, SIMD permutations are used to append the cell
		boundaries to the global array without using branches.
	*/

	// Compute cells in thread buffers
	// Note: We distribute the work to the threads from all the point sets simulataneously
	this->thread_cells.resize(this->n_threads);
	#pragma omp parallel num_threads(this->n_threads)
	{
		const int thread_id = omp_get_thread_num();
		CellList& cells = this->thread_cells[thread_id];
		cells.init_with_at_least_size(std::max(10000, (int)(0.1*this->get_total_n_points())));
		int cell_i = 0;

		const int n_points = this->get_total_n_points();
		const int chunksize = n_points / n_threads;
		const int begin_thread_point = chunksize * thread_id;
		const int end_thread_point = (thread_id == n_threads - 1) ? n_points : chunksize * (thread_id + 1);

		// Do not process if there are no points
		const int thread_n_points = end_thread_point - begin_thread_point;
		if (thread_n_points > 0) {

			// Find the set of the first point
			int begin_set = 0;
			while (this->set_offsets[begin_set + 1] < begin_thread_point) {
				begin_set++;
			}

			// Find the end set (set of the last point + 1)
			//const int last_point = end_thread_point - 1;
			int end_set = begin_set;
			while (this->set_offsets[end_set + 1] < end_thread_point) {
				end_set++;
			}
			end_set++;

			// For each set
			for (int set_i = begin_set; set_i < end_set; set_i++) {

				const int set_offset = this->set_offsets[set_i];
				const int end_set_point = this->set_offsets[set_i + 1];

				// Clip the range of the points to process with the set range
				const int begin_point = std::max(set_offset, begin_thread_point);
				const int end_point = std::min(end_set_point, end_thread_point);

				const float* points = this->set_points[set_i];

				// Insert first cell
				// Note: For every set, we create a new cell to avoid points from different sets sharing the same cell
				const int point_idx = begin_point - set_offset;
				const float* p = points + 3 * point_idx;
				cells.offsets[cell_i] = begin_point;
				cells.i[cell_i] = (uint16_t)((p[0] - this->domain_float.bottom[0]) * this->cell_size_inv);
				cells.j[cell_i] = (uint16_t)((p[1] - this->domain_float.bottom[1]) * this->cell_size_inv);
				cells.k[cell_i] = (uint16_t)((p[2] - this->domain_float.bottom[2]) * this->cell_size_inv);
				cell_i++;

				uint16_t current_i = cells.i[cell_i - 1];
				uint16_t current_j = cells.j[cell_i - 1];
				uint16_t current_k = cells.k[cell_i - 1];

				// SIMD preparation
				const __m256 bx = _mm256_set1_ps(this->domain_float.bottom[0]);
				const __m256 by = _mm256_set1_ps(this->domain_float.bottom[1]);
				const __m256 bz = _mm256_set1_ps(this->domain_float.bottom[2]);
				const __m256 cell_size_inv = _mm256_set1_ps(this->cell_size_inv);
				const __m128i rotate = _mm_setr_epi8(0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);

				// SIMD cells computation
				int point_i = begin_point + 1;
				for (; point_i < end_point - 8; point_i += 8) {

					if (cells.capacity - cell_i <= 16) { // 16 so it can write a full 8SIMD line, then 7 remainder and then 1 from a new set
						cells.grow_while_keeping_data(16 + 2 * cells.capacity);
					}

					// Load
					const int point_idx = point_i - set_offset;
					const float* p = points + 3 * point_idx;

					// xyzxyz -> xxyyzz
					// https://www.intel.com/content/dam/develop/external/us/en/documents/normvec-181650.pdf
					__m256 m03;
					__m256 m14;
					__m256 m25;
					m03 = _mm256_castps128_ps256(_mm_loadu_ps(p)); // load lower halves
					m14 = _mm256_castps128_ps256(_mm_loadu_ps(p + 4));
					m25 = _mm256_castps128_ps256(_mm_loadu_ps(p + 8));
					m03 = _mm256_insertf128_ps(m03, _mm_loadu_ps(p + 12), 1); // load upper halves
					m14 = _mm256_insertf128_ps(m14, _mm_loadu_ps(p + 16), 1);
					m25 = _mm256_insertf128_ps(m25, _mm_loadu_ps(p + 20), 1);

					__m256 xy = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE(2, 1, 3, 2)); // upper x's and y's 
					__m256 yz = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE(1, 0, 2, 1)); // lower y's and z's 
					__m256 x = _mm256_shuffle_ps(m03, xy, _MM_SHUFFLE(2, 0, 3, 0));
					__m256 y = _mm256_shuffle_ps(yz, xy, _MM_SHUFFLE(3, 1, 2, 0));
					__m256 z = _mm256_shuffle_ps(yz, m25, _MM_SHUFFLE(3, 0, 3, 1));

					// point -> cell
					const __m256i i32 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_sub_ps(x, bx), cell_size_inv));
					const __m256i j32 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_sub_ps(y, by), cell_size_inv));
					const __m256i k32 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_sub_ps(z, bz), cell_size_inv));

					// int32 to int16
					/*
						There is no instruction to downcast int32 to int16. We need to use _mm256_shuffle_epi8 which can only
						shuffle bytes locally in 128bit lanes. See https://stackoverflow.com/questions/49721807/what-is-the-inverse-of-mm256-cvtepi16-epi32
					*/
					const __m256i i16_two_128 = _mm256_shuffle_epi8(i32, epi32_to_epi16_mask128); // epi16 in each 128bit line
					const __m256i i16_one_128 = _mm256_permute4x64_epi64(i16_two_128, 0x58);
					const __m128i i = _mm256_castsi256_si128(i16_one_128);

					const __m256i j16_two_128 = _mm256_shuffle_epi8(j32, epi32_to_epi16_mask128); // epi16 in each 128bit line
					const __m256i j16_one_128 = _mm256_permute4x64_epi64(j16_two_128, 0x58);
					const __m128i j = _mm256_castsi256_si128(j16_one_128);

					const __m256i k16_two_128 = _mm256_shuffle_epi8(k32, epi32_to_epi16_mask128); // epi16 in each 128bit line
					const __m256i k16_one_128 = _mm256_permute4x64_epi64(k16_two_128, 0x58);
					const __m128i k = _mm256_castsi256_si128(k16_one_128);

					// load comparison -> _mm_setr_epi16(cells.i[cell_i - 1], i[0], i[1], [2], i[3], i[4], i[5], i[6]);
					const __m128i ic = _mm_insert_epi16(_mm_shuffle_epi8(i, rotate), current_i, 0);
					const __m128i jc = _mm_insert_epi16(_mm_shuffle_epi8(j, rotate), current_j, 0);
					const __m128i kc = _mm_insert_epi16(_mm_shuffle_epi8(k, rotate), current_k, 0);

					// compare
					const __m128i cmp_not = _mm_and_si128(_mm_and_si128(_mm_cmpeq_epi16(i, ic), _mm_cmpeq_epi16(j, jc)), _mm_cmpeq_epi16(k, kc));
					const __m128i cmp16 = _mm_andnot_si128(cmp_not, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));

					// create shuffle mask id
					const __m128i cmp8 = _mm_shuffle_epi8(cmp16, epi16_to_epi8_cmp_mask);
					const int mask_id = _mm_movemask_epi8(cmp8);

					// Make new cells
					const __m128i shuffle_mask = this->shift_lut_8[mask_id];
					_mm_storeu_si128((__m128i*) & cells.i[cell_i], _mm_shuffle_epi8(i, shuffle_mask));
					_mm_storeu_si128((__m128i*) & cells.j[cell_i], _mm_shuffle_epi8(j, shuffle_mask));
					_mm_storeu_si128((__m128i*) & cells.k[cell_i], _mm_shuffle_epi8(k, shuffle_mask));

					const __m256i offsets = _mm256_add_epi32(this->shift_lut_32[mask_id], _mm256_set1_epi32(point_i));
					_mm256_storeu_si256((__m256i*) & cells.offsets[cell_i], offsets);

					cell_i += _mm_popcnt_u32(mask_id);

					current_i = cells.i[cell_i - 1];
					current_j = cells.j[cell_i - 1];
					current_k = cells.k[cell_i - 1];
				}

				// Remainder
				for (; point_i < end_point; point_i++) {
					const int point_idx = point_i - set_offset;
					const float* p = points + 3 * point_idx;

					const uint16_t i = (uint16_t)((p[0] - this->domain_float.bottom[0]) * this->cell_size_inv);
					const uint16_t j = (uint16_t)((p[1] - this->domain_float.bottom[1]) * this->cell_size_inv);
					const uint16_t k = (uint16_t)((p[2] - this->domain_float.bottom[2]) * this->cell_size_inv);

					const bool cmp = (i == cells.i[cell_i - 1] && j == cells.j[cell_i - 1] && k == cells.k[cell_i - 1]);
					if (!cmp) {
						cells.offsets[cell_i] = point_i;
						cells.i[cell_i] = i;
						cells.j[cell_i] = j;
						cells.k[cell_i] = k;
						cell_i++;
					}
				}
			}
		}

		cells.n_cells = cell_i;
	}

	// (From here on the implementation is identical to the scalar mode)

	// Merge thread cell lists into the global vector
	//// Thread cells offsets
	std::vector<int> thread_cells_offsets(this->n_threads + 1);
	thread_cells_offsets[0] = 0;
	for (int thread_id = 0; thread_id < this->n_threads; thread_id++) {
		thread_cells_offsets[thread_id + 1] = thread_cells_offsets[thread_id] + this->thread_cells[thread_id].n_cells;
	}
	const int n_cells = thread_cells_offsets.back();

	//// Parallel write
	if (this->cells.capacity < n_cells) {
		this->cells.init_with_at_least_size((int)(1.1*n_cells));
	}
	#pragma omp parallel num_threads(this->n_threads)
	{
		const int thread_id = omp_get_thread_num();
		const CellList& thread_cells = this->thread_cells[thread_id];
		const int begin = thread_cells_offsets[thread_id];
		const int n = thread_cells_offsets[thread_id + 1] - thread_cells_offsets[thread_id];

		memcpy(&this->cells.i[begin], thread_cells.i, sizeof(uint16_t) * n);
		memcpy(&this->cells.j[begin], thread_cells.j, sizeof(uint16_t) * n);
		memcpy(&this->cells.k[begin], thread_cells.k, sizeof(uint16_t) * n);
		memcpy(&this->cells.offsets[begin], thread_cells.offsets, sizeof(int) * n);
	}
	this->cells.n_cells = n_cells;
	this->cells.offsets[n_cells] = this->get_total_n_points();
	this->avg_points_per_cell = this->get_total_n_points() / n_cells;


	// Variable radius
	if (!this->is_global_search_radius_set) {
		this->cells.init_radii();

		float max_radius = 0.0f;
		int begin_set_cell = 0;
		int end_set_cell = 0;
		for (int set_i = 0; set_i < this->n_sets; set_i++) {
			const int set_offset = this->set_offsets[set_i];
			const float* set_radii = this->set_radii[set_i];

			if (set_i + 1 == this->n_sets) {
				end_set_cell = this->cells.n_cells;
			}
			else {
				end_set_cell = (int)std::distance(this->cells.offsets, std::lower_bound(this->cells.offsets + begin_set_cell, this->cells.offsets + this->cells.n_cells, this->set_offsets[set_i + 1]));
			}

			#pragma omp parallel num_threads(this->n_threads)
			{
				const int thread_id = omp_get_thread_num();

				const int n = end_set_cell - begin_set_cell;
				const int chunksize = n / this->n_threads;
				const int begin = thread_id * chunksize;
				const int end = (thread_id + 1 == this->n_threads) ? n : (thread_id + 1) * chunksize;

				float thread_max_radius = 0.0f;

				for (int cell_i = begin_set_cell + begin; cell_i < begin_set_cell + end; cell_i++) {

					const int begin_point = this->cells.offsets[cell_i] - set_offset;
					const int end_point = this->cells.offsets[cell_i + 1] - set_offset;

					float cell_radius = set_radii[begin_point];
					for (int point_i = begin_point + 1; point_i < end_point; point_i++) {
						cell_radius = std::max(cell_radius, set_radii[point_i]);
					}
					this->cells.radii[cell_i] = cell_radius;
					thread_max_radius = std::max(thread_max_radius, cell_radius);
				}

				#pragma omp critical
				{
					max_radius = std::max(max_radius, thread_max_radius);
				}
			}

			begin_set_cell = end_set_cell;
		}
		this->max_search_radius = max_radius;
	}
	else {
		this->max_search_radius = this->global_search_radius;
	}
}
void tns::TreeNSearch::_prepare_root()
{
	// Cell world bounds
	const float world_length = this->domain_float.top[0] - this->domain_float.bottom[0];
	const uint16_t end_cell = (uint16_t)std::round(world_length / this->cell_size);
	this->octree_root.buffer.domain.bottom = { 0, 0, 0 };
	this->octree_root.buffer.domain.top = { end_cell, end_cell, end_cell };

	// Cell indices of root are all the cells
	this->octree_root.buffer.cell_indices.init_with_at_least_size(this->cells.n_cells, 1.1);
	this->octree_root.buffer.cell_indices.cursor = this->octree_root.buffer.cell_indices.data + this->cells.n_cells;
	std::iota(this->octree_root.buffer.cell_indices.data, this->octree_root.buffer.cell_indices.data + this->cells.n_cells, 0);

	// Max radius
	this->octree_root.buffer.max_search_radius = this->max_search_radius;

	// Parallel
	this->parallel_octree = this->get_total_n_points() > this->n_points_for_parallel_octree;
}
void tns::TreeNSearch::_build_octree_and_gather_leaves()
{
	this->_prepare_root();

	// this->thread_leaves: Make enough space to avoid reallocations
	int n_leaves = 0;
	for (std::vector<OctreeNode*>& leafs : this->thread_leaves) {
		n_leaves += (int)leafs.size();
	}

	this->thread_leaves.resize(this->n_threads);
	for (std::vector<OctreeNode*>& leafs : this->thread_leaves) {
		leafs.clear();
		leafs.reserve(n_leaves);
	}

	if (this->parallel_octree) {
		tf::Taskflow taskflow;
		taskflow.emplace([this](tf::Subflow& sf)
			{
				this->_run_octree_node(this->octree_root, 0, &sf);
			}
		);
		this->executor = new tf::Executor(this->n_threads);
		this->executor->run(taskflow).wait();
		delete this->executor;
	}
	else
	{
		this->_run_octree_node(this->octree_root, 0, nullptr);
	}
}
void tns::TreeNSearch::_build_octree_and_gather_leaves_simd()
{
	this->_prepare_root();

	// this->thread_leaves: Make enough space to avoid reallocations
	int n_leaves = 0;
	for (std::vector<OctreeNode*>& leafs : this->thread_leaves) {
		n_leaves += (int)leafs.size();
	}

	this->thread_leaves.resize(this->n_threads);
	for (std::vector<OctreeNode*>& leafs : this->thread_leaves) {
		leafs.clear();
		leafs.reserve(n_leaves);
	}

	if (this->parallel_octree) {
		tf::Taskflow taskflow;
		taskflow.emplace([this](tf::Subflow& sf)
			{
				this->_run_octree_node_simd(this->octree_root, 0, &sf);
			}
		);
		this->executor = new tf::Executor(this->n_threads);
		this->executor->run(taskflow).wait();
		delete this->executor;
	}
	else
	{
		this->_run_octree_node_simd(this->octree_root, 0, nullptr);
	}
}
void tns::TreeNSearch::_run_octree_node(RecursiveOctreeNode& node_buffer, const size_t depth, tf::Subflow* sf)
{
	/*
		Recursively classify cells into the octree children nodes that they overlap with.
	*/
	auto& buffer = node_buffer.buffer;
	const int n_cells = buffer.cell_indices.size();
	const int n_approx_points = n_cells * this->avg_points_per_cell;

	// Early exit: Impossible to find neighbors
	if (n_cells == 0) {
		std::cout << "compare TreeNSearch: node with zero cells." << std::endl;
		exit(-1);
	}

	// Ghost cells
	uint16_t ghost_cells;
	const float ghost_cells_float = buffer.max_search_radius / this->cell_size;
	const float remainder = ghost_cells_float - std::round(ghost_cells_float);
	if (std::abs(remainder) < 2 * std::numeric_limits<float>::epsilon()) {
		ghost_cells = (uint16_t)std::round(ghost_cells_float);
	}
	else {
		ghost_cells = (uint16_t)(ghost_cells_float)+1;
	}

	// Early exit: Too small AABB
	if (buffer.domain.top[0] - buffer.domain.bottom[0] <= ghost_cells) {
		node_buffer.delete_children();
		this->thread_leaves[(this->parallel_octree) ? this->executor->this_worker_id() : 0].push_back(&buffer);
		return;
	}

	// Exit: Cap criteria
	if (n_approx_points <= this->n_points_to_stop_recursion) {
		node_buffer.delete_children();
		this->thread_leaves[(this->parallel_octree) ? this->executor->this_worker_id() : 0].push_back(&buffer);
	}
	else {
		// Classification
		/*
			Leaves order is imposed by libmorton to have a proper zsort:
				000, 100, 010, 110, 001, 101, 011, 111
		*/
		std::array<uint16_t, 3>& bottom = buffer.domain.bottom;
		std::array<uint16_t, 3>& top = buffer.domain.top;

		// Create children
		node_buffer.populate_children();

		// Pivot
		const std::array<uint16_t, 3> pivot = {
			(uint16_t)((bottom[0] + top[0]) / 2),
			(uint16_t)((bottom[1] + top[1]) / 2),
			(uint16_t)((bottom[2] + top[2]) / 2) };
		const std::array<uint16_t, 3> pivot_plus = {
			(uint16_t)(pivot[0] + ghost_cells),
			(uint16_t)(pivot[1] + ghost_cells),
			(uint16_t)(pivot[2] + ghost_cells) };
		const std::array<uint16_t, 3> pivot_minus = {
			(uint16_t)(pivot[0] - ghost_cells),
			(uint16_t)(pivot[1] - ghost_cells),
			(uint16_t)(pivot[2] - ghost_cells) };

		// Children AABB
		for (int i = 0; i < 8; i++) {
			node_buffer.children[i]->buffer.domain = node_buffer.buffer.domain;
		}

		node_buffer.children[0]->buffer.domain.top = pivot;

		node_buffer.children[4]->buffer.domain.bottom[2] = pivot[2];
		node_buffer.children[4]->buffer.domain.top[0] = pivot[0];
		node_buffer.children[4]->buffer.domain.top[1] = pivot[1];

		node_buffer.children[2]->buffer.domain.bottom[1] = pivot[1];
		node_buffer.children[2]->buffer.domain.top[0] = pivot[0];
		node_buffer.children[2]->buffer.domain.top[2] = pivot[2];

		node_buffer.children[6]->buffer.domain.bottom[1] = pivot[1];
		node_buffer.children[6]->buffer.domain.bottom[2] = pivot[2];
		node_buffer.children[6]->buffer.domain.top[0] = pivot[0];

		node_buffer.children[1]->buffer.domain.bottom[0] = pivot[0];
		node_buffer.children[1]->buffer.domain.top[1] = pivot[1];
		node_buffer.children[1]->buffer.domain.top[2] = pivot[2];

		node_buffer.children[5]->buffer.domain.bottom[0] = pivot[0];
		node_buffer.children[5]->buffer.domain.bottom[2] = pivot[2];
		node_buffer.children[5]->buffer.domain.top[1] = pivot[1];

		node_buffer.children[3]->buffer.domain.bottom[0] = pivot[0];
		node_buffer.children[3]->buffer.domain.bottom[1] = pivot[1];
		node_buffer.children[3]->buffer.domain.top[2] = pivot[2];

		node_buffer.children[7]->buffer.domain.bottom = pivot;


		// Children
		//// Count how many cells goes to each child
		std::array<int, 8> children_n_cell = { 0, 0, 0, 0, 0, 0, 0, 0 };
		for (int cell_i = 0; cell_i < n_cells; cell_i++) {
			const int c = buffer.cell_indices[cell_i];
			const std::array<uint16_t, 3> ijk = { this->cells.i[c], this->cells.j[c], this->cells.k[c] };

			if (ijk[0] < pivot_plus[0]) {
				if (ijk[1] < pivot_plus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						children_n_cell[0]++;
					}
					if (ijk[2] >= pivot_minus[2]) {
						children_n_cell[4]++;
					}
				}
				if (ijk[1] >= pivot_minus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						children_n_cell[2]++;
					}
					if (ijk[2] >= pivot_minus[2]) {
						children_n_cell[6]++;
					}
				}
			}
			if (ijk[0] >= pivot_minus[0]) {
				if (ijk[1] < pivot_plus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						children_n_cell[1]++;
					}
					if (ijk[2] >= pivot_minus[2]) {
						children_n_cell[5]++;
					}
				}
				if (ijk[1] >= pivot_minus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						children_n_cell[3]++;
					}
					if (ijk[2] >= pivot_minus[2]) {
						children_n_cell[7]++;
					}
				}
			}
		}

		//// Allocation of the children cells
		std::array<int*, 8> children_cell_indices_cursor;
		for (int child_i = 0; child_i < 8; child_i++) {
			node_buffer.children[child_i]->buffer.cell_indices.init_with_at_least_size(children_n_cell[child_i], 1.1);
			children_cell_indices_cursor[child_i] = node_buffer.children[child_i]->buffer.cell_indices.cursor;
		}

		//// Copying the cell indices to the relevant child cell lists
		for (int cell_i = 0; cell_i < n_cells; cell_i++) {
			const int c = buffer.cell_indices[cell_i];
			const std::array<uint16_t, 3> ijk = { this->cells.i[c], this->cells.j[c], this->cells.k[c] };

			if (ijk[0] < pivot_plus[0]) {
				if (ijk[1] < pivot_plus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						*children_cell_indices_cursor[0]++ = c;
					}
					if (ijk[2] >= pivot_minus[2]) {
						*children_cell_indices_cursor[4]++ = c;
					}
				}
				if (ijk[1] >= pivot_minus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						*children_cell_indices_cursor[2]++ = c;
					}
					if (ijk[2] >= pivot_minus[2]) {
						*children_cell_indices_cursor[6]++ = c;
					}
				}
			}
			if (ijk[0] >= pivot_minus[0]) {
				if (ijk[1] < pivot_plus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						*children_cell_indices_cursor[1]++ = c;
					}
					if (ijk[2] >= pivot_minus[2]) {
						*children_cell_indices_cursor[5]++ = c;
					}
				}
				if (ijk[1] >= pivot_minus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						*children_cell_indices_cursor[3]++ = c;
					}
					if (ijk[2] >= pivot_minus[2]) {
						*children_cell_indices_cursor[7]++ = c;
					}
				}
			}
		}

		//// Set the cursor at the end of the cell lists
		for (int child_i = 0; child_i < 8; child_i++) {
			node_buffer.children[child_i]->buffer.cell_indices.cursor = children_cell_indices_cursor[child_i];
		}

		// Variable of fixed search radius
		if (!this->is_global_search_radius_set) {
			for (int child_i = 0; child_i < 8; child_i++) {
				auto& child_cell_indices = node_buffer.children[child_i]->buffer.cell_indices;
				const int n_cells = child_cell_indices.size();
				float max_radius = 0.0f;
				for (int cell_i = 0; cell_i < n_cells; cell_i++) {
					const int c = child_cell_indices[cell_i];
					max_radius = std::max(max_radius, this->cells.radii[c]);
				}
				node_buffer.children[child_i]->buffer.max_search_radius = max_radius;
			}
		}
		else {
			for (int child_i = 0; child_i < 8; child_i++) {
				node_buffer.children[child_i]->buffer.max_search_radius = this->global_search_radius;
			}
		}

		// Run children
		if (this->parallel_octree && n_cells > this->n_cells_in_node_for_switching_to_sequential) {
			for (int child_i = 0; child_i < 8; child_i++) {
				if (node_buffer.children[child_i]->buffer.cell_indices.size() > 0) {
					auto child_task = sf->emplace([depth, &node_buffer, child_i, this](tf::Subflow& sf) {
						this->_run_octree_node(*node_buffer.children[child_i], depth + 1, &sf);
						}
					);
				}
				else {
					node_buffer.children[child_i]->delete_children();
				}
			}
		}
		else {
			for (int child_i = 0; child_i < 8; child_i++) {
				if (node_buffer.children[child_i]->buffer.cell_indices.size() > 0) {
					this->_run_octree_node(*node_buffer.children[child_i], depth + 1, sf);
				}
				else {
					node_buffer.children[child_i]->delete_children();
				}
			}
		}
	}
}
void tns::TreeNSearch::_run_octree_node_simd(RecursiveOctreeNode& node_buffer, const size_t depth, tf::Subflow* sf)
{
	/*
		In the SIMD implementation of _run_octree_node() both counting and appending
		cells to the children lists is done in batches of eight cells.
		Counting is done with the _popocnt instruction to get the true values in a mask.
		Pushing is done by using SIMD permutations to selectively copying the indices of the cells
		corresponding to true comparisons.
	*/
	auto& buffer = node_buffer.buffer;
	const int n_cells = buffer.cell_indices.size();
	const int n_approx_points = n_cells * this->avg_points_per_cell;

	// Early exit: Impossible to find neighbors
	if (n_cells == 0) {
		std::cout << "compare TreeNSearch: node with zero cells." << std::endl;
		exit(-1);
	}

	// Ghost cells
	uint16_t ghost_cells;
	const float ghost_cells_float = buffer.max_search_radius / this->cell_size;
	const float remainder = ghost_cells_float - std::round(ghost_cells_float);
	if (std::abs(remainder) < 2 * std::numeric_limits<float>::epsilon()) {
		ghost_cells = (uint16_t)std::round(ghost_cells_float);
	}
	else {
		ghost_cells = (uint16_t)(ghost_cells_float)+1;
	}

	// Early exit: Too small AABB
	if (buffer.domain.top[0] - buffer.domain.bottom[0] <= ghost_cells) {
		node_buffer.delete_children();
		this->thread_leaves[(this->parallel_octree) ? this->executor->this_worker_id() : 0].push_back(&buffer);
		return;
	}

	// Exit: Cap criteria
	if (n_approx_points <= this->n_points_to_stop_recursion) {
		node_buffer.delete_children();
		this->thread_leaves[(this->parallel_octree) ? this->executor->this_worker_id() : 0].push_back(&buffer);
	}
	else {
		// Classification
		/*
			Leaves order is imposed by libmorton to have a proper zsort:
				000, 100, 010, 110, 001, 101, 011, 111
		*/

		std::array<uint16_t, 3>& bottom = node_buffer.buffer.domain.bottom;
		std::array<uint16_t, 3>& top = node_buffer.buffer.domain.top;

		// Create children
		node_buffer.populate_children();

		// Pivot
		const std::array<uint16_t, 3> pivot = {
			(uint16_t)((bottom[0] + top[0]) / 2),
			(uint16_t)((bottom[1] + top[1]) / 2),
			(uint16_t)((bottom[2] + top[2]) / 2) };
		const std::array<uint16_t, 3> pivot_plus = {
			(uint16_t)(pivot[0] + ghost_cells),
			(uint16_t)(pivot[1] + ghost_cells),
			(uint16_t)(pivot[2] + ghost_cells) };
		const std::array<uint16_t, 3> pivot_minus = {
			(uint16_t)(pivot[0] - ghost_cells),
			(uint16_t)(pivot[1] - ghost_cells),
			(uint16_t)(pivot[2] - ghost_cells) };

		// Children AABB
		for (int i = 0; i < 8; i++) {
			node_buffer.children[i]->buffer.domain = node_buffer.buffer.domain;
		}

		/*
			Our order to libmorton order mapping:
				0 -> 0
				1 -> 4
				2 -> 2
				3 -> 6
				4 -> 1
				5 -> 5
				6 -> 3
				7 -> 7
		*/
		node_buffer.children[0]->buffer.domain.top = pivot;

		node_buffer.children[4]->buffer.domain.bottom[2] = pivot[2];
		node_buffer.children[4]->buffer.domain.top[0] = pivot[0];
		node_buffer.children[4]->buffer.domain.top[1] = pivot[1];

		node_buffer.children[2]->buffer.domain.bottom[1] = pivot[1];
		node_buffer.children[2]->buffer.domain.top[0] = pivot[0];
		node_buffer.children[2]->buffer.domain.top[2] = pivot[2];

		node_buffer.children[6]->buffer.domain.bottom[1] = pivot[1];
		node_buffer.children[6]->buffer.domain.bottom[2] = pivot[2];
		node_buffer.children[6]->buffer.domain.top[0] = pivot[0];

		node_buffer.children[1]->buffer.domain.bottom[0] = pivot[0];
		node_buffer.children[1]->buffer.domain.top[1] = pivot[1];
		node_buffer.children[1]->buffer.domain.top[2] = pivot[2];

		node_buffer.children[5]->buffer.domain.bottom[0] = pivot[0];
		node_buffer.children[5]->buffer.domain.bottom[2] = pivot[2];
		node_buffer.children[5]->buffer.domain.top[1] = pivot[1];

		node_buffer.children[3]->buffer.domain.bottom[0] = pivot[0];
		node_buffer.children[3]->buffer.domain.bottom[1] = pivot[1];
		node_buffer.children[3]->buffer.domain.top[2] = pivot[2];

		node_buffer.children[7]->buffer.domain.bottom = pivot;


		// Children
		const int* cells = buffer.cell_indices.data;
		const uint16_t* ii = this->cells.i;
		const uint16_t* jj = this->cells.j;
		const uint16_t* kk = this->cells.k;

		const std::array<__m128i, 3> pivot_plus_simd = { _mm_set1_epi16(pivot_plus[0]), _mm_set1_epi16(pivot_plus[1]), _mm_set1_epi16(pivot_plus[2]) };
		const std::array<__m128i, 3> pivot_minus_simd = { _mm_set1_epi16(pivot_minus[0]), _mm_set1_epi16(pivot_minus[1]), _mm_set1_epi16(pivot_minus[2]) };

		//// Count how many cells goes to each child
		std::array<int, 8> children_n_cell = { 16, 16, 16, 16, 16, 16, 16, 16 }; // Do not deal with remainder (8 extra). Everything is divided by 2 in the end (16).
		for (int cell_i = 0; cell_i < n_cells - 8; cell_i += 8) {

			// Fetch ijk
			const int* c = &cells[cell_i];
			const __m128i i = _mm_setr_epi16(ii[c[0]], ii[c[1]], ii[c[2]], ii[c[3]], ii[c[4]], ii[c[5]], ii[c[6]], ii[c[7]]);
			const __m128i j = _mm_setr_epi16(jj[c[0]], jj[c[1]], jj[c[2]], jj[c[3]], jj[c[4]], jj[c[5]], jj[c[6]], jj[c[7]]);
			const __m128i k = _mm_setr_epi16(kk[c[0]], kk[c[1]], kk[c[2]], kk[c[3]], kk[c[4]], kk[c[5]], kk[c[6]], kk[c[7]]);

			// 0: ijk[0] < pivot_plus[0] && ijk[1] < pivot_plus[1] && ijk[2] < pivot_plus[2];
			const __m128i cmp0 = _mm_and_si128(_mm_and_si128(
				_mm_cmplt_epi16(i, pivot_plus_simd[0]),
				_mm_cmplt_epi16(j, pivot_plus_simd[1])),
				_mm_cmplt_epi16(k, pivot_plus_simd[2])
			);
			children_n_cell[0] += _mm_popcnt_u32(_mm_movemask_epi8(cmp0));

			// 4: ijk[0] < pivot_plus[0] && ijk[1] < pivot_plus[1] && ijk[2] >= pivot_minus[2];
			const __m128i cmp4 = _mm_and_si128(_mm_and_si128(
				_mm_cmplt_epi16(i, pivot_plus_simd[0]),
				_mm_cmplt_epi16(j, pivot_plus_simd[1])),
				_mm_cmpge_epi16(k, pivot_minus_simd[2])
			);
			children_n_cell[4] += _mm_popcnt_u32(_mm_movemask_epi8(cmp4));

			// 2: ijk[0] < pivot_plus[0] && ijk[1] >= pivot_minus[1] && ijk[2] < pivot_plus[2];
			const __m128i cmp2 = _mm_and_si128(_mm_and_si128(
				_mm_cmplt_epi16(i, pivot_plus_simd[0]),
				_mm_cmpge_epi16(j, pivot_minus_simd[1])),
				_mm_cmplt_epi16(k, pivot_plus_simd[2])
			);
			children_n_cell[2] += _mm_popcnt_u32(_mm_movemask_epi8(cmp2));

			// 6: ijk[0] < pivot_plus[0] && ijk[1] >= pivot_minus[1] && ijk[2] >= pivot_minus[2];
			const __m128i cmp6 = _mm_and_si128(_mm_and_si128(
				_mm_cmplt_epi16(i, pivot_plus_simd[0]),
				_mm_cmpge_epi16(j, pivot_minus_simd[1])),
				_mm_cmpge_epi16(k, pivot_minus_simd[2])
			);
			children_n_cell[6] += _mm_popcnt_u32(_mm_movemask_epi8(cmp6));

			// 1: ijk[0] >= pivot_minus[0] && ijk[1] < pivot_plus[1] && ijk[2] < pivot_plus[2];
			const __m128i cmp1 = _mm_and_si128(_mm_and_si128(
				_mm_cmpge_epi16(i, pivot_minus_simd[0]),
				_mm_cmplt_epi16(j, pivot_plus_simd[1])),
				_mm_cmplt_epi16(k, pivot_plus_simd[2])
			);
			children_n_cell[1] += _mm_popcnt_u32(_mm_movemask_epi8(cmp1));

			// 5: ijk[0] >= pivot_minus[0] && ijk[1] < pivot_plus[1] && ijk[2] >= pivot_minus[2];
			const __m128i cmp5 = _mm_and_si128(_mm_and_si128(
				_mm_cmpge_epi16(i, pivot_minus_simd[0]),
				_mm_cmplt_epi16(j, pivot_plus_simd[1])),
				_mm_cmpge_epi16(k, pivot_minus_simd[2])
			);
			children_n_cell[5] += _mm_popcnt_u32(_mm_movemask_epi8(cmp5));

			// 3: ijk[0] >= pivot_minus[0] && ijk[1] >= pivot_minus[1] && ijk[2] < pivot_plus[2];
			const __m128i cmp3 = _mm_and_si128(_mm_and_si128(
				_mm_cmpge_epi16(i, pivot_minus_simd[0]),
				_mm_cmpge_epi16(j, pivot_minus_simd[1])),
				_mm_cmplt_epi16(k, pivot_plus_simd[2])
			);
			children_n_cell[3] += _mm_popcnt_u32(_mm_movemask_epi8(cmp3));

			// 7: ijk[0] >= pivot_minus[0] && ijk[1] >= pivot_minus[1] && ijk[2] >= pivot_minus[2];
			const __m128i cmp7 = _mm_and_si128(_mm_and_si128(
				_mm_cmpge_epi16(i, pivot_minus_simd[0]),
				_mm_cmpge_epi16(j, pivot_minus_simd[1])),
				_mm_cmpge_epi16(k, pivot_minus_simd[2])
			);
			children_n_cell[7] += _mm_popcnt_u32(_mm_movemask_epi8(cmp7));
		}

		//// Allocation of the children cells
		std::array<uvector<int>*, 8> children_cell_indices;
		for (int child_i = 0; child_i < 8; child_i++) {
			children_cell_indices[child_i] = &node_buffer.children[child_i]->buffer.cell_indices;
			children_cell_indices[child_i]->init_with_at_least_size((int)(children_n_cell[child_i] / 2.0), 1.1);
		}

		//// SIMD push to children
		auto pusher = [children_cell_indices, this](const int child_i, const __m256i cell_indices, const __m128i cmp)
		{
			const __m128i cmp_8 = _mm_shuffle_epi8(cmp, epi16_to_epi8_cmp_mask);
			const int mask_id = _mm_movemask_epi8(cmp_8);
			const __m256i cell_indices_shuffled = _mm256_permutevar8x32_epi32(cell_indices, this->shift_lut_32[mask_id]);
			_mm256_storeu_si256((__m256i*) children_cell_indices[child_i]->cursor, cell_indices_shuffled);
			children_cell_indices[child_i]->cursor += _mm_popcnt_u32(mask_id);
		};

		//// Copying the cell indices to the relevant child cell lists
		/*
			AVX2 integer comparisons are only equality and greater than. Therefore
			we need to reformulate each octant test into that.
		*/
		const __m256i iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
		int cell_i = 0;
		for (; cell_i < n_cells - 8; cell_i += 8) {

			// Fetch ijk
			const int* c = &cells[cell_i];
			const __m128i i = _mm_setr_epi16(ii[c[0]], ii[c[1]], ii[c[2]], ii[c[3]], ii[c[4]], ii[c[5]], ii[c[6]], ii[c[7]]);
			const __m128i j = _mm_setr_epi16(jj[c[0]], jj[c[1]], jj[c[2]], jj[c[3]], jj[c[4]], jj[c[5]], jj[c[6]], jj[c[7]]);
			const __m128i k = _mm_setr_epi16(kk[c[0]], kk[c[1]], kk[c[2]], kk[c[3]], kk[c[4]], kk[c[5]], kk[c[6]], kk[c[7]]);
			const __m256i cell_indices = _mm256_loadu_si256((__m256i*) & cells[cell_i]);

			// 0: ijk[0] < pivot_plus[0] && ijk[1] < pivot_plus[1] && ijk[2] < pivot_plus[2];
			const __m128i cmp0 = _mm_and_si128(_mm_and_si128(
				_mm_cmplt_epi16(i, pivot_plus_simd[0]),
				_mm_cmplt_epi16(j, pivot_plus_simd[1])),
				_mm_cmplt_epi16(k, pivot_plus_simd[2])
			);
			pusher(0, cell_indices, cmp0);

			// 4: ijk[0] < pivot_plus[0] && ijk[1] < pivot_plus[1] && ijk[2] >= pivot_minus[2];
			const __m128i cmp4 = _mm_and_si128(_mm_and_si128(
				_mm_cmplt_epi16(i, pivot_plus_simd[0]),
				_mm_cmplt_epi16(j, pivot_plus_simd[1])),
				_mm_cmpge_epi16(k, pivot_minus_simd[2])
			);
			pusher(4, cell_indices, cmp4);

			// 2: ijk[0] < pivot_plus[0] && ijk[1] >= pivot_minus[1] && ijk[2] < pivot_plus[2];
			const __m128i cmp2 = _mm_and_si128(_mm_and_si128(
				_mm_cmplt_epi16(i, pivot_plus_simd[0]),
				_mm_cmpge_epi16(j, pivot_minus_simd[1])),
				_mm_cmplt_epi16(k, pivot_plus_simd[2])
			);
			pusher(2, cell_indices, cmp2);

			// 6: ijk[0] < pivot_plus[0] && ijk[1] >= pivot_minus[1] && ijk[2] >= pivot_minus[2];
			const __m128i cmp6 = _mm_and_si128(_mm_and_si128(
				_mm_cmplt_epi16(i, pivot_plus_simd[0]),
				_mm_cmpge_epi16(j, pivot_minus_simd[1])),
				_mm_cmpge_epi16(k, pivot_minus_simd[2])
			);
			pusher(6, cell_indices, cmp6);

			// 1: ijk[0] >= pivot_minus[0] && ijk[1] < pivot_plus[1] && ijk[2] < pivot_plus[2];
			const __m128i cmp1 = _mm_and_si128(_mm_and_si128(
				_mm_cmpge_epi16(i, pivot_minus_simd[0]),
				_mm_cmplt_epi16(j, pivot_plus_simd[1])),
				_mm_cmplt_epi16(k, pivot_plus_simd[2])
			);
			pusher(1, cell_indices, cmp1);

			// 5: ijk[0] >= pivot_minus[0] && ijk[1] < pivot_plus[1] && ijk[2] >= pivot_minus[2];
			const __m128i cmp5 = _mm_and_si128(_mm_and_si128(
				_mm_cmpge_epi16(i, pivot_minus_simd[0]),
				_mm_cmplt_epi16(j, pivot_plus_simd[1])),
				_mm_cmpge_epi16(k, pivot_minus_simd[2])
			);
			pusher(5, cell_indices, cmp5);

			// 3: ijk[0] >= pivot_minus[0] && ijk[1] >= pivot_minus[1] && ijk[2] < pivot_plus[2];
			const __m128i cmp3 = _mm_and_si128(_mm_and_si128(
				_mm_cmpge_epi16(i, pivot_minus_simd[0]),
				_mm_cmpge_epi16(j, pivot_minus_simd[1])),
				_mm_cmplt_epi16(k, pivot_plus_simd[2])
			);
			pusher(3, cell_indices, cmp3);

			// 7: ijk[0] >= pivot_minus[0] && ijk[1] >= pivot_minus[1] && ijk[2] >= pivot_minus[2];
			const __m128i cmp7 = _mm_and_si128(_mm_and_si128(
				_mm_cmpge_epi16(i, pivot_minus_simd[0]),
				_mm_cmpge_epi16(j, pivot_minus_simd[1])),
				_mm_cmpge_epi16(k, pivot_minus_simd[2])
			);
			pusher(7, cell_indices, cmp7);
		}

		//// Remainder
		for (; cell_i < n_cells; cell_i++) {
			const int c = buffer.cell_indices[cell_i];
			const std::array<uint16_t, 3> ijk = { this->cells.i[c], this->cells.j[c], this->cells.k[c] };

			if (ijk[0] < pivot_plus[0]) {
				if (ijk[1] < pivot_plus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						*children_cell_indices[0]->cursor++ = c;
						
					}
					if (ijk[2] >= pivot_minus[2]) {
						*children_cell_indices[4]->cursor++ = c;
					}
				}
				if (ijk[1] >= pivot_minus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						*children_cell_indices[2]->cursor++ = c;
					}
					if (ijk[2] >= pivot_minus[2]) {
						*children_cell_indices[6]->cursor++ = c;
					}
				}
			}
			if (ijk[0] >= pivot_minus[0]) {
				if (ijk[1] < pivot_plus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						*children_cell_indices[1]->cursor++ = c;
					}
					if (ijk[2] >= pivot_minus[2]) {
						*children_cell_indices[5]->cursor++ = c;
					}
				}
				if (ijk[1] >= pivot_minus[1]) {
					if (ijk[2] < pivot_plus[2]) {
						*children_cell_indices[3]->cursor++ = c;
					}
					if (ijk[2] >= pivot_minus[2]) {
						*children_cell_indices[7]->cursor++ = c;
					}
				}
			}
		}

		// Variable of fixed search radius
		if (!this->is_global_search_radius_set) {
			for (int child_i = 0; child_i < 8; child_i++) {
				auto& child_cell_indices = node_buffer.children[child_i]->buffer.cell_indices;
				const int n_cells = child_cell_indices.size();
				float max_radius = 0.0f;
				for (int cell_i = 0; cell_i < n_cells; cell_i++) {
					const int c = child_cell_indices[cell_i];
					max_radius = std::max(max_radius, this->cells.radii[c]);
				}
				node_buffer.children[child_i]->buffer.max_search_radius = max_radius;
			}
		}
		else {
			for (int child_i = 0; child_i < 8; child_i++) {
				node_buffer.children[child_i]->buffer.max_search_radius = this->global_search_radius;
			}
		}

		// Run children
		if (this->parallel_octree && n_cells > this->n_cells_in_node_for_switching_to_sequential) {
			for (int child_i = 0; child_i < 8; child_i++) {
				if (node_buffer.children[child_i]->buffer.cell_indices.size() > 0) {
					auto child_task = sf->emplace([depth, &node_buffer, child_i, this](tf::Subflow& sf) {
						this->_run_octree_node_simd(*node_buffer.children[child_i], depth + 1, &sf);
						}
					);
				}
				else {
					node_buffer.children[child_i]->delete_children();
				}
			}
		}
		else {
			for (int child_i = 0; child_i < 8; child_i++) {
				if (node_buffer.children[child_i]->buffer.cell_indices.size() > 0) {
					this->_run_octree_node_simd(*node_buffer.children[child_i], depth + 1, sf);
				}
				else {
					node_buffer.children[child_i]->delete_children();
				}
			}
		}
	}
}
void tns::TreeNSearch::_solve_leaves(const bool use_simd)
{
	/*
		At this point, we have a list of octree leaf pointers per thread containing clusters of neighboring cells.

		The leaf pointers are concatenated into a single list and the bruteforce pair-wise distance
		comparisons are called.
	*/

	// Concatenate leaf pointers
	//// Prefix sum the thread leaf buffers
	std::vector<int> thread_leaves_offsets(this->n_threads + 1);
	thread_leaves_offsets[0] = 0;
	std::transform_inclusive_scan(
		this->thread_leaves.begin(), this->thread_leaves.end(), thread_leaves_offsets.begin() + 1, std::plus<int>{},
		[](const std::vector<OctreeNode*>& leaves) { return (int)leaves.size(); }
	);
	const int n_leaves = thread_leaves_offsets.back();

	//// Parallel concatenate
	this->leaves.clear();
	this->leaves.resize(n_leaves);
	#pragma omp parallel num_threads(this->n_threads)
	{
		const int thread_id = omp_get_thread_num();
		const std::vector<OctreeNode*>& thread_leaves = this->thread_leaves[thread_id];

		const int n = thread_leaves_offsets[thread_id + 1] - thread_leaves_offsets[thread_id];
		const int begin_leaf = thread_leaves_offsets[thread_id];
		for (int leaf_i = 0; leaf_i < n; leaf_i++) {
			this->leaves[begin_leaf + leaf_i] = thread_leaves[leaf_i];
		}
	}

	// Execute bruteforce
	if (use_simd) {
		#pragma omp parallel for schedule(dynamic) num_threads(this->n_threads)
		for (int leaf_i = 0; leaf_i < n_leaves; leaf_i++) {
			const int thread_id = omp_get_thread_num();
			this->_brute_force_simd(*this->leaves[leaf_i], thread_id);
		}
	}
	else {
		#pragma omp parallel for schedule(dynamic) num_threads(this->n_threads)
		for (int leaf_i = 0; leaf_i < n_leaves; leaf_i++) {
			const int thread_id = omp_get_thread_num();
			this->_brute_force(*this->leaves[leaf_i], thread_id);
		}
	}
}
void tns::TreeNSearch::_prepare_brute_force(OctreeNode& leaf_buffer, const int thread_id)
{
	/*
		Once we have all the cells relevant to a octree leaf, we have to gather all the point data
		to perform the bruteforce pair-wise distance comparisons.

		In this function, the internal and relevant external particles to the octree leaf domain
		are gathered and flagged accordingly. A point is inside if the cell to which the point
		belongs is inside.

		The implementation is cumbersome due to points belonging to different sets must be fetched
		correctly.
	*/
	// Common stuff / name shortening
	BruteforceBuffer& bruteforce_buffer = this->thread_bruteforce_buffers[thread_id];

	// Domain
	const std::array<uint16_t, 3> b = leaf_buffer.domain.bottom;
	const std::array<uint16_t, 3> t = leaf_buffer.domain.top;
	AABB<float> leaf_float_domain;
	for (int i = 0; i < 3; i++) {
		leaf_float_domain.bottom[i] = this->domain_float.bottom[i] + ((float)b[i]) * this->cell_size;
		leaf_float_domain.top[i] = this->domain_float.bottom[i] + ((float)t[i]) * this->cell_size;
	}
	const AABB<float> extended_leaf_float_domain = leaf_float_domain.get_extended(leaf_buffer.max_search_radius);
	const std::array<float, 3>& eb = extended_leaf_float_domain.bottom;
	const std::array<float, 3>& et = extended_leaf_float_domain.top;

	// Gather points
	//// Gather Cells
	int max_inside_points = 0;
	const int n_cells = leaf_buffer.cell_indices.size();
	for (int loc_cell_i = 0; loc_cell_i < n_cells; loc_cell_i++) {
		const int cell_i = leaf_buffer.cell_indices[loc_cell_i];
		max_inside_points += this->cells.offsets[cell_i + 1] - this->cells.offsets[cell_i];
	}

	//// Gather points, their indices, and flag them inside/outside according to their cell inside/outside-ness
	bruteforce_buffer.points.points.resize(max_inside_points);
	bruteforce_buffer.points.indices.resize(max_inside_points);
	bruteforce_buffer.points.radii_sq.resize(max_inside_points);
	bruteforce_buffer.points.inside_indices.resize(this->n_sets);

	{
		int cursor = 0;
		int cell_begin_set = 0;
		int cell_end_set = -1;
		for (int set_i = 0; set_i < this->n_sets; set_i++) {
			const int begin_set_cursor = cursor;

			const float* set_points = this->set_points[set_i];
			const int set_offset = this->set_offsets[set_i];
			bruteforce_buffer.points.inside_indices[set_i].clear();

			// Binary search the set offsets in the cells (fewer but equivalent to the points)
			auto it = std::lower_bound(leaf_buffer.cell_indices.data + cell_begin_set, leaf_buffer.cell_indices.data + n_cells, this->set_offsets[set_i + 1], [this](const int& cell_idx, const int v) { return this->cells.offsets[cell_idx] < v; });
			cell_end_set = (int)std::distance(leaf_buffer.cell_indices.data, it);

			// Loop over the cells of the current set
			for (int loc_cell_i = cell_begin_set; loc_cell_i < cell_end_set; loc_cell_i++) {
				const int cell_i = leaf_buffer.cell_indices[loc_cell_i];
				const uint16_t i = this->cells.i[cell_i];
				const uint16_t j = this->cells.j[cell_i];
				const uint16_t k = this->cells.k[cell_i];
				const uint8_t inside_cell =
					(b[0] <= i && i < t[0] &&
						b[1] <= j && j < t[1] &&
						b[2] <= k && k < t[2]);

				const int point_begin = this->cells.offsets[cell_i];
				const int point_end = this->cells.offsets[cell_i + 1];

				// If cell is inside, all the points are inside. Gather and flag inside.
				if (inside_cell) {
					for (int point_i = point_begin; point_i < point_end; point_i++) {
						const int set_point_i = point_i - set_offset;
						memcpy(&bruteforce_buffer.points.points[cursor], &set_points[3 * set_point_i], sizeof(float) * 3);
						bruteforce_buffer.points.indices[cursor] = point_i;
						//if (cursor == 8) {
						//	std::cout << "trouble";
						//}
						bruteforce_buffer.points.inside_indices[set_i].push_back(cursor);
						cursor++;
					}
				}

				// If the cell is outside, only the points inside the extended leaf are gathered and flagged outside.
				else {
					const std::array<float, 3>& b = extended_leaf_float_domain.bottom;
					const std::array<float, 3>& t = extended_leaf_float_domain.top;

					for (int point_i = this->cells.offsets[cell_i]; point_i < this->cells.offsets[cell_i + 1]; point_i++) {
						const int set_point_i = point_i - set_offset;
						const float* p = set_points + 3 * set_point_i;
						const bool in = b[0] <= p[0] && b[1] <= p[1] && b[2] <= p[2] && p[0] <= t[0] && p[1] <= t[1] && p[2] <= t[2];

						if (in) {
							memcpy(&bruteforce_buffer.points.points[cursor], &set_points[3 * set_point_i], sizeof(float) * 3);
							bruteforce_buffer.points.indices[cursor] = point_i;
							cursor++;
						}
					}
				}
			}
			const int end_set_cursor = cursor;

			// Radii
			if (this->is_global_search_radius_set) {
				// ignore
			}
			else {
				const float* set_radii = this->set_radii[set_i];
				for (int i = begin_set_cursor; i < end_set_cursor; i++) {
					const float radius = set_radii[bruteforce_buffer.points.indices[i] - set_offset];
					bruteforce_buffer.points.radii_sq[i] = radius * radius;
				}
			}

			cell_begin_set = cell_end_set;
		}
		bruteforce_buffer.points.points.resize(cursor);
		bruteforce_buffer.points.indices.resize(cursor);
		bruteforce_buffer.points.radii_sq.resize(cursor);
		bruteforce_buffer.points.inside.resize(cursor);
	}


	const int n_points = (int)bruteforce_buffer.points.points.size();
	if (n_points == 0) {
		return;
	}


	// Offsets
	bruteforce_buffer.points.set_offsets.clear();
	bruteforce_buffer.points.set_offsets.push_back(0);
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		auto it = std::lower_bound(bruteforce_buffer.points.indices.begin() + bruteforce_buffer.points.set_offsets.back(), bruteforce_buffer.points.indices.end(), this->set_offsets[set_i + 1]);
		bruteforce_buffer.points.set_offsets.push_back((int)std::distance(bruteforce_buffer.points.indices.begin(), it));
	}

	// Remove index offset per set
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		const int offset = this->set_offsets[set_i];
		for (int i = bruteforce_buffer.points.set_offsets[set_i]; i < bruteforce_buffer.points.set_offsets[set_i + 1]; i++) {
			bruteforce_buffer.points.indices[i] -= offset;
		}
	}
}
void tns::TreeNSearch::_brute_force(OctreeNode& leaf_buffer, const int thread_id)
{
	/*
		Performs the pair-wise distance comparison between the relevant points inside
		an octree leaf.
	*/
	this->_prepare_brute_force(leaf_buffer, thread_id); // Gathers relevant point data
	
	// Common stuff / name shortening
	BruteforceBuffer& bruteforce_buffer = this->thread_bruteforce_buffers[thread_id];
	auto& neighborlist = this->thread_neighborlists[thread_id];
	const int n_points = (int)bruteforce_buffer.points.points.size();
	BruteforceBuffer::LeafPoints& buffer = bruteforce_buffer.points;

	if (n_points == 0) {
		return;
	}

	// Symmetric search
	const bool perform_symmetric_check = !this->is_global_search_radius_set && this->symmetric_search;

	// Enough space in the neighbors buffer
	bruteforce_buffer.neighbors_buffer.resize(n_points + this->n_sets); // n_points + n_neighs int per set

	// Error
	if (neighborlist.get_chunk_size() < n_points + this->n_sets) {
		#pragma omp critical
		{
			std::cout << "TreeNSearch compare: Too many potential number of neighbors for the neighbor lists data structure (" << n_points + this->n_sets << ")" << std::endl;
			std::cout << "You can increase the chunksize in TreeNSearch.thread_neighborlists." << std::endl;
			std::cout << "However, for typical applications, these are too many neighbors. Since, something probably went wrong here you some useful internal state variables:" << std::endl << std::endl;
			this->print_state();
			exit(-1);
		}
	}

	// Bruteforce
	if (!perform_symmetric_check) {
		for (int set_i = 0; set_i < this->n_sets; set_i++) {
			if (this->active_searches[set_i].size() == 0) { continue; }

			for (const int loc_i : buffer.inside_indices[set_i]) {
				const int set_idx_i = buffer.indices[loc_i];
				const float search_radius_sq = (this->is_global_search_radius_set) ? this->global_search_radius_sq : buffer.radii_sq[loc_i];

				// Fetch current point float coords
				const std::array<float, 3> point_i = buffer.points[loc_i];

				// Point - Point distance
				//// Make it so that the point cannot be its own neighbor
				buffer.points[loc_i][0] = std::numeric_limits<float>::max();

				//// Distance computation to other points
				for (const int set_j : this->active_searches[set_i]) {
					int* neighborlist_cursor = bruteforce_buffer.neighbors_buffer.data();
					int* neighborlist_begin = neighborlist_cursor;
					*neighborlist_cursor++ = -1; // Leave space for total n_neighbors written

					for (int loc_j = buffer.set_offsets[set_j]; loc_j < buffer.set_offsets[set_j + 1]; loc_j++) {
						const std::array<float, 3>& point_j = buffer.points[loc_j];

						double dx = point_i[0] - point_j[0];
						double dist_sq = dx * dx;
						dx = point_i[1] - point_j[1];
						dist_sq += dx * dx;
						dx = point_i[2] - point_j[2];
						dist_sq += dx * dx;

						bool cmp = dist_sq <= search_radius_sq;

						if (cmp) {
							*neighborlist_cursor++ = buffer.indices[loc_j];
						}
					}
					const int n_neighbors = (int)std::distance(neighborlist_begin, neighborlist_cursor) - 1;
					*neighborlist_begin = n_neighbors;

					// Copy neighborlist to destination
					int* neighborlist_dest = neighborlist.get_cursor_with_space_to_write(n_neighbors + 1);
					this->solution_ptr[this->_get_set_pair_id(set_i, set_j)][set_idx_i] = neighborlist_dest;
					memcpy(neighborlist_dest, neighborlist_begin, sizeof(int) * (n_neighbors + 1));
				}

				// Restore the point i coords
				buffer.points[loc_i][0] = point_i[0];
			}
		}
	}
	else {
		for (int set_i = 0; set_i < this->n_sets; set_i++) {
			if (this->active_searches[set_i].size() == 0) { continue; }

			for (const int loc_i : buffer.inside_indices[set_i]) {
				const int set_idx_i = buffer.indices[loc_i];
				const float search_radius_sq = buffer.radii_sq[loc_i];

				// Fetch current point float coords
				const std::array<float, 3> point_i = buffer.points[loc_i];

				// Point - Point distance
				//// Make it so that the point cannot be its own neighbor
				buffer.points[loc_i][0] = std::numeric_limits<float>::max();

				//// Distance computation to other points
				for (const int set_j : this->active_searches[set_i]) {
					int* neighborlist_cursor = bruteforce_buffer.neighbors_buffer.data();
					int* neighborlist_begin = neighborlist_cursor;
					*neighborlist_cursor++ = -1; // Leave space for total n_neighbors written

					for (int loc_j = buffer.set_offsets[set_j]; loc_j < buffer.set_offsets[set_j + 1]; loc_j++) {
						const std::array<float, 3>& point_j = buffer.points[loc_j];

						double dx = point_i[0] - point_j[0];
						double dist_sq = dx * dx;
						dx = point_i[1] - point_j[1];
						dist_sq += dx * dx;
						dx = point_i[2] - point_j[2];
						dist_sq += dx * dx;

						bool cmp = dist_sq <= search_radius_sq;
						const bool cmp2 = dist_sq <= buffer.radii_sq[loc_j];
						cmp = cmp || cmp2;

						if (cmp) {
							*neighborlist_cursor++ = buffer.indices[loc_j];
						}
					}
					const int n_neighbors = (int)std::distance(neighborlist_begin, neighborlist_cursor) - 1;
					*neighborlist_begin = n_neighbors;

					// Copy neighborlist to destination
					int* neighborlist_dest = neighborlist.get_cursor_with_space_to_write(n_neighbors + 1);
					this->solution_ptr[this->_get_set_pair_id(set_i, set_j)][set_idx_i] = neighborlist_dest;
					memcpy(neighborlist_dest, neighborlist_begin, sizeof(int) * (n_neighbors + 1));
					
					//if (set_i == 0 && set_j == 1 && set_idx_i == 7) {
					//	std::cout << "trouble";
					//}

					//std::cout << "[("<< set_i << ", " << set_j << "): " << set_idx_i << " ]";
					//for (int i = 0; i < n_neighbors; i++) {
					//	std::cout << " " << neighborlist_dest[i + 1];
					//}
					//std::cout << std::endl;
				}

				// Restore the point i coords
				buffer.points[loc_i][0] = point_i[0];
			}
		}
	}
}
bool tns::TreeNSearch::_prepare_brute_force_simd(OctreeNode& leaf_buffer, const int thread_id)
{
	/*
		The SIMD implementation of _prepare_brute_force() takes advantage of the fast
		shuffling from the xyzxyz to xxyyzz coordinate layout and checks eight points
		against the bounding box for the external points.
	*/

	// Common stuff / name shortening
	BruteforceBuffer::LeafPointsSIMD& buffer = this->thread_bruteforce_buffers[thread_id].points_simd;

	// Domain
	const std::array<uint16_t, 3> b = leaf_buffer.domain.bottom;
	const std::array<uint16_t, 3> t = leaf_buffer.domain.top;
	AABB<float> leaf_float_domain;
	for (int i = 0; i < 3; i++) {
		leaf_float_domain.bottom[i] = this->domain_float.bottom[i] + ((float)b[i]) * this->cell_size;
		leaf_float_domain.top[i] = this->domain_float.bottom[i] + ((float)t[i]) * this->cell_size;
	}
	const AABB<float> extended_leaf_float_domain = leaf_float_domain.get_extended(leaf_buffer.max_search_radius);
	const std::array<float, 3>& eb = extended_leaf_float_domain.bottom;
	const std::array<float, 3>& et = extended_leaf_float_domain.top;


	// Count points
	int max_inside_points = 0;
	const int n_cells = leaf_buffer.cell_indices.size();
	for (int loc_cell_i = 0; loc_cell_i < n_cells; loc_cell_i++) {
		const int cell_i = leaf_buffer.cell_indices[loc_cell_i];
		max_inside_points += this->cells.offsets[cell_i + 1] - this->cells.offsets[cell_i];
	}

	// Gather
	bool any_cell = false;
	const int n_safe = max_inside_points + 8 * this->n_sets; // Enough for padding
	const int n_simd_safe = n_safe / 8 + 1;

	//// Allocation
	buffer.x.resize(n_simd_safe);
	buffer.y.resize(n_simd_safe);
	buffer.z.resize(n_simd_safe);
	buffer.indices.resize(n_simd_safe);
	buffer.radii_sq.resize(n_simd_safe);
	buffer.inside_indices.resize(this->n_sets);
	buffer.set_offsets.clear();
	buffer.set_offsets.push_back(0);

	//// Scalar views
	float* xs = (float*)buffer.x.data();
	float* ys = (float*)buffer.y.data();
	float* zs = (float*)buffer.z.data();
	float* radii_sq_s = (float*)buffer.radii_sq.data();
	int* indices_s = (int*)buffer.indices.data();

	//// SIMD
	const __m256i iota8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	const std::array<__m256, 3> b_simd = { _mm256_set1_ps(eb[0]), _mm256_set1_ps(eb[1]), _mm256_set1_ps(eb[2]) };
	const std::array<__m256, 3> t_simd = { _mm256_set1_ps(et[0]), _mm256_set1_ps(et[1]), _mm256_set1_ps(et[2]) };
	__m256 x, y, z;

	// Gather loop
	int cursor = 0;
	int cell_begin_set = 0;
	int cell_end_set = -1;
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		const int begin_set_cursor = cursor;

		const float* set_points = this->set_points[set_i];
		const int set_offset = this->set_offsets[set_i];

		buffer.inside_indices[set_i].resize(n_safe);
		int inside_indices_cursor = 0;

		// Binary search the set offsets in the cells (fewer but equivalent to the points)
		auto it = std::lower_bound(leaf_buffer.cell_indices.data + cell_begin_set, leaf_buffer.cell_indices.data + n_cells, this->set_offsets[set_i + 1], [this](const int& cell_idx, const int v) { return this->cells.offsets[cell_idx] < v; });
		cell_end_set = (int)std::distance(leaf_buffer.cell_indices.data, it);

		// Loop over the cells of the current set
		for (int loc_cell_i = cell_begin_set; loc_cell_i < cell_end_set; loc_cell_i++) {
			const int cell_i = leaf_buffer.cell_indices[loc_cell_i];
			const int points_begin = this->cells.offsets[cell_i];
			const int points_end = this->cells.offsets[cell_i + 1];
			const uint16_t i = this->cells.i[cell_i];
			const uint16_t j = this->cells.j[cell_i];
			const uint16_t k = this->cells.k[cell_i];
			const uint8_t inside_cell = (b[0] <= i && i < t[0]) && (b[1] <= j && j < t[1]) && (b[2] <= k && k < t[2]);

			// If cell is inside, all the points are inside
			if (inside_cell) {
				any_cell = true;
				int point_i = points_begin;
				for (; point_i < points_end - 7; point_i += 8) {
					const int set_point_i = point_i - set_offset;

					xyzxyz_to_xxyyzz(x, y, z, &set_points[3 * set_point_i]);

					_mm256_storeu_ps(xs + cursor, x);
					_mm256_storeu_ps(ys + cursor, y);
					_mm256_storeu_ps(zs + cursor, z);
					_mm256_storeu_si256((__m256i*)(indices_s + cursor), _mm256_add_epi32(_mm256_set1_epi32(set_point_i), iota8));
					_mm256_storeu_si256((__m256i*)(buffer.inside_indices[set_i].data() + inside_indices_cursor), _mm256_add_epi32(_mm256_set1_epi32(cursor), iota8));

					cursor += 8;
					inside_indices_cursor += 8;
				}

				// Remainder
				const int remainder = points_end - point_i;
				const int set_point_i = point_i - set_offset;
				xyzxyz_to_xxyyzz(x, y, z, &set_points[3 * set_point_i], remainder, std::numeric_limits<float>::max());
				_mm256_storeu_ps(xs + cursor, x);
				_mm256_storeu_ps(ys + cursor, y);
				_mm256_storeu_ps(zs + cursor, z);
				_mm256_storeu_si256((__m256i*)(indices_s + cursor), _mm256_add_epi32(_mm256_set1_epi32(set_point_i), iota8));
				_mm256_storeu_si256((__m256i*)(buffer.inside_indices[set_i].data() + inside_indices_cursor), _mm256_add_epi32(_mm256_set1_epi32(cursor), iota8));

				cursor += remainder;
				inside_indices_cursor += remainder;
			}

			// If the cell is outside, only the points inside the extended leaf are gathered
			else {
				int point_i = points_begin;
				for (; point_i < points_end - 7; point_i += 8) {
					const int set_point_i = point_i - set_offset;

					// Load
					xyzxyz_to_xxyyzz(x, y, z, &set_points[3 * set_point_i]);

					// Point in AABB
					__m256 cmp = _mm256_cmp_ps(b_simd[0], x, _CMP_LE_OS);
					cmp = _mm256_and_ps(cmp, _mm256_cmp_ps(b_simd[1], y, _CMP_LE_OS));
					cmp = _mm256_and_ps(cmp, _mm256_cmp_ps(b_simd[2], z, _CMP_LE_OS));
					cmp = _mm256_and_ps(cmp, _mm256_cmp_ps(x, t_simd[0], _CMP_LE_OS));
					cmp = _mm256_and_ps(cmp, _mm256_cmp_ps(y, t_simd[1], _CMP_LE_OS));
					cmp = _mm256_and_ps(cmp, _mm256_cmp_ps(z, t_simd[2], _CMP_LE_OS));

					// Push back
					const int mask = _mm256_movemask_ps(cmp);
					const __m256i& shift_mask = this->shift_lut_32[mask];
					const int count = _mm_popcnt_u32(mask);

					_mm256_storeu_ps(xs + cursor, _mm256_permutevar8x32_ps(x, shift_mask));
					_mm256_storeu_ps(ys + cursor, _mm256_permutevar8x32_ps(y, shift_mask));
					_mm256_storeu_ps(zs + cursor, _mm256_permutevar8x32_ps(z, shift_mask));
					_mm256_storeu_si256((__m256i*)(indices_s + cursor), _mm256_permutevar8x32_epi32(_mm256_add_epi32(_mm256_set1_epi32(set_point_i), iota8), shift_mask));

					cursor += count;
				}

				// Remainder
				const int remainder = points_end - point_i;
				const int set_point_i = point_i - set_offset;

				// Load
				xyzxyz_to_xxyyzz(x, y, z, &set_points[3 * set_point_i], remainder, std::numeric_limits<float>::max());

				// Point in AABB
				__m256 cmp = _mm256_cmp_ps(b_simd[0], x, _CMP_LE_OS);
				cmp = _mm256_and_ps(cmp, _mm256_cmp_ps(b_simd[1], y, _CMP_LE_OS));
				cmp = _mm256_and_ps(cmp, _mm256_cmp_ps(b_simd[2], z, _CMP_LE_OS));
				cmp = _mm256_and_ps(cmp, _mm256_cmp_ps(x, t_simd[0], _CMP_LE_OS));
				cmp = _mm256_and_ps(cmp, _mm256_cmp_ps(y, t_simd[1], _CMP_LE_OS));
				cmp = _mm256_and_ps(cmp, _mm256_cmp_ps(z, t_simd[2], _CMP_LE_OS));

				// Push back
				const int mask = _mm256_movemask_ps(cmp);
				const __m256i& shift_mask = this->shift_lut_32[mask];
				const int count = _mm_popcnt_u32(mask);

				_mm256_storeu_ps(xs + cursor, _mm256_permutevar8x32_ps(x, shift_mask));
				_mm256_storeu_ps(ys + cursor, _mm256_permutevar8x32_ps(y, shift_mask));
				_mm256_storeu_ps(zs + cursor, _mm256_permutevar8x32_ps(z, shift_mask));
				_mm256_storeu_si256((__m256i*)(indices_s + cursor), _mm256_permutevar8x32_epi32(_mm256_add_epi32(_mm256_set1_epi32(set_point_i), iota8), shift_mask));

				cursor += count;
			}
		}
		const int end_set_cursor = cursor;

		// Radii
		if (this->is_global_search_radius_set) {
			// ignore
		}
		else {
			float* radii_sq_cursor = radii_sq_s + begin_set_cursor;
			int* indices_cursor = indices_s + begin_set_cursor;

			const float* set_radii = this->set_radii[set_i];
			for (int i = begin_set_cursor; i < end_set_cursor; i++) {
				const float radius = set_radii[*indices_cursor++];
				*radii_sq_cursor++ = radius * radius;
			}
		}

		// SIMD padding
		const int remainder = cursor % 8;
		if (remainder > 0) {
			const int padding = 8 - remainder;
			for (int i = 0; i < padding; i++) {
				xs[cursor] = std::numeric_limits<float>::max();
				ys[cursor] = std::numeric_limits<float>::max();
				ys[cursor] = std::numeric_limits<float>::max();
				indices_s[cursor] = -1;

				if (!this->is_global_search_radius_set) {
					radii_sq_s[cursor] = -1.0f;
				}
				cursor++;
			}
		}

		cell_begin_set = cell_end_set;
		buffer.set_offsets.push_back(cursor/8);
		buffer.inside_indices[set_i].resize(inside_indices_cursor);
	}

	// Resize down
	const int n_simd = cursor / 8;
	buffer.x.resize(n_simd);
	buffer.y.resize(n_simd);
	buffer.z.resize(n_simd);
	buffer.indices.resize(n_simd);
	buffer.radii_sq.resize(n_simd);

	if (!any_cell) {
		return false;
	}

	// x, y, z -> xyz
	buffer.xyz.resize(n_simd);
	for (int i = 0; i < n_simd; i++) {
		buffer.xyz[i][0] = buffer.x[i];
		buffer.xyz[i][1] = buffer.y[i];
		buffer.xyz[i][2] = buffer.z[i];
	}

	return true;
}
void tns::TreeNSearch::_brute_force_simd(OctreeNode& leaf_buffer, const int thread_id)
{
	/*
		The differences in the SIMD implementation of _brute_force() with respect to
		the scalar version is that eight points are compared at the same time and
		that the neighborlists are writen using SIMD permutations instead of
		conditional branches.
	*/
	const int not_empty = this->_prepare_brute_force_simd(leaf_buffer, thread_id);

	if (!not_empty) {
		return;
	}
	
	// Common stuff / name shortening
	BruteforceBuffer& bruteforce_buffer = this->thread_bruteforce_buffers[thread_id];
	auto& neighborlist = this->thread_neighborlists[thread_id];
	BruteforceBuffer::LeafPointsSIMD& buffer = bruteforce_buffer.points_simd;
	const int n_simd_lines = (int)buffer.indices.size();
	const int n_points = 8 * n_simd_lines;

	if (n_points == 0) {
		return;
	}

	// Scalar views
	std::array<std::array<float, 8>, 3>* xyzs = reinterpret_cast<std::array<std::array<float, 8>, 3>*>(buffer.xyz[0].data());
	float* radii_sq_s = (float*)buffer.radii_sq.data();
	int* indices_s = (int*)buffer.indices.data();

	// Symmetric search
	const bool perform_symmetric_check = !this->is_global_search_radius_set && this->symmetric_search;

	// Enough space in the neighbors buffer for the SIMD permutation
	bruteforce_buffer.neighbors_buffer.resize(n_points + this->n_sets); // n_points + n_neighs

	// Error
	if (neighborlist.get_chunk_size() < n_points + this->n_sets) {
		#pragma omp critical
		{
			std::cout << "TreeNSearch compare: Too many potential number of neighbors for the neighbor lists data structure (" << n_points + this->n_sets << ")" << std::endl;
			std::cout << "You can increase the chunksize in TreeNSearch.thread_neighborlists." << std::endl;
			std::cout << "However, for typical applications, these are too many neighbors. Since, something probably went wrong here you some useful internal state variables:" << std::endl << std::endl;
			this->print_state();
			exit(-1);
		}
	}

	// Bruteforce
	if (!perform_symmetric_check) {
		for (int set_i = 0; set_i < this->n_sets; set_i++) {
			if (this->active_searches[set_i].size() == 0) { continue; }

			for (const int loc_i : buffer.inside_indices[set_i]) {
				const int set_point_i = indices_s[loc_i];
				const float search_radius_sq = (this->is_global_search_radius_set) ? this->global_search_radius_sq : radii_sq_s[loc_i];

				// Fetch current point float coords
				const int line_i = loc_i / 8;
				const int in_line_i = loc_i % 8;
				const __m256 px = _mm256_set1_ps(xyzs[line_i][0][in_line_i]);
				const __m256 py = _mm256_set1_ps(xyzs[line_i][1][in_line_i]);
				const __m256 pz = _mm256_set1_ps(xyzs[line_i][2][in_line_i]);

				// Make it so that the point cannot be its own neighbor
				const float tmp_x = xyzs[line_i][0][in_line_i];
				xyzs[line_i][0][in_line_i] = std::numeric_limits<float>::max();

				// SIMD distance computation to other points
				for (const int set_j : this->active_searches[set_i]) {
					int* neighborlist_cursor = bruteforce_buffer.neighbors_buffer.data();
					int* neighborlist_begin = neighborlist_cursor;
					*neighborlist_cursor++ = -1; // Leave space for total n_neighbors written

					for (int line_j = buffer.set_offsets[set_j]; line_j < buffer.set_offsets[set_j + 1]; line_j++) {
						auto& q = buffer.xyz[line_j];

						// Squared distance
						__m256 dist_sq = _mm256_sub_ps(px, q[0]);
						dist_sq = _mm256_mul_ps(dist_sq, dist_sq);
						__m256 tmp = _mm256_sub_ps(py, q[1]);
						dist_sq = _mm256_add_ps(dist_sq, _mm256_mul_ps(tmp, tmp));
						tmp = _mm256_sub_ps(pz, q[2]);
						dist_sq = _mm256_add_ps(dist_sq, _mm256_mul_ps(tmp, tmp));

						// Comparison
						__m256 cmp = _mm256_cmp_ps(dist_sq, _mm256_set1_ps(search_radius_sq), _CMP_LE_OS);
						
						// Push back
						const int mask = _mm256_movemask_ps(cmp);
						const __m256i shuffled_indices = _mm256_permutevar8x32_epi32(buffer.indices[line_j], this->shift_lut_32[mask]);
						_mm256_storeu_si256((__m256i*)neighborlist_cursor, shuffled_indices);
						neighborlist_cursor += _mm_popcnt_u32(mask);
					}
					const int n_neighbors = (int)std::distance(neighborlist_begin, neighborlist_cursor) - 1;
					*neighborlist_begin = n_neighbors;

					// Copy neighborlist to destination
					int* neighborlist_dest = neighborlist.get_cursor_with_space_to_write(n_neighbors + 1);
					this->solution_ptr[this->_get_set_pair_id(set_i, set_j)][set_point_i] = neighborlist_dest;
					memcpy(neighborlist_dest, neighborlist_begin, sizeof(int) * (n_neighbors + 1));
				}

				// Restore the point i coords
				xyzs[line_i][0][in_line_i] = tmp_x;
			}
		}
	}
	else {
		for (int set_i = 0; set_i < this->n_sets; set_i++) {
			if (this->active_searches[set_i].size() == 0) { continue; }

			for (const int loc_i : buffer.inside_indices[set_i]) {
				const int set_point_i = indices_s[loc_i];
				const float search_radius_sq = radii_sq_s[loc_i];

				// Fetch current point float coords
				const int line_i = loc_i / 8;
				const int in_line_i = loc_i % 8;
				const __m256 px = _mm256_set1_ps(xyzs[line_i][0][in_line_i]);
				const __m256 py = _mm256_set1_ps(xyzs[line_i][1][in_line_i]);
				const __m256 pz = _mm256_set1_ps(xyzs[line_i][2][in_line_i]);

				// Make it so that the point cannot be its own neighbor
				const float tmp_x = xyzs[line_i][0][in_line_i];
				xyzs[line_i][0][in_line_i] = std::numeric_limits<float>::max();

				// SIMD distance computation to other points
				for (const int set_j : this->active_searches[set_i]) {
					int* neighborlist_cursor = bruteforce_buffer.neighbors_buffer.data();
					int* neighborlist_begin = neighborlist_cursor;
					*neighborlist_cursor++ = -1; // Leave space for total n_neighbors written

					for (int line_j = buffer.set_offsets[set_j]; line_j < buffer.set_offsets[set_j + 1]; line_j++) {
						auto& q = buffer.xyz[line_j];

						// Squared distance
						__m256 dist_sq = _mm256_sub_ps(px, q[0]);
						dist_sq = _mm256_mul_ps(dist_sq, dist_sq);
						__m256 tmp = _mm256_sub_ps(py, q[1]);
						dist_sq = _mm256_add_ps(dist_sq, _mm256_mul_ps(tmp, tmp));
						tmp = _mm256_sub_ps(pz, q[2]);
						dist_sq = _mm256_add_ps(dist_sq, _mm256_mul_ps(tmp, tmp));

						// Comparison
						__m256 cmp = _mm256_cmp_ps(dist_sq, _mm256_set1_ps(search_radius_sq), _CMP_LE_OS);
						const __m256 cmp2 = _mm256_cmp_ps(dist_sq, buffer.radii_sq[line_j], _CMP_LE_OS);
						cmp = _mm256_or_ps(cmp, cmp2);

						// Push back
						const int mask = _mm256_movemask_ps(cmp);
						const __m256i shuffled_indices = _mm256_permutevar8x32_epi32(buffer.indices[line_j], this->shift_lut_32[mask]);
						_mm256_storeu_si256((__m256i*)neighborlist_cursor, shuffled_indices);
						neighborlist_cursor += _mm_popcnt_u32(mask);
					}
					const int n_neighbors = (int)std::distance(neighborlist_begin, neighborlist_cursor) - 1;
					*neighborlist_begin = n_neighbors;

					// Copy neighborlist to destination
					int* neighborlist_dest = neighborlist.get_cursor_with_space_to_write(n_neighbors + 1);
					this->solution_ptr[this->_get_set_pair_id(set_i, set_j)][set_point_i] = neighborlist_dest;
					memcpy(neighborlist_dest, neighborlist_begin, sizeof(int) * (n_neighbors + 1));
				}

				// Restore the point i coords
				xyzs[line_i][0][in_line_i] = tmp_x;
			}
		}
	}
}

void tns::TreeNSearch::prepare_zsort()
{
	/*
		Computes the ordering of the particles corresponding to the Morton space filling curve
		with respect to the cells of the background grid.

		If no octree has been constructed yet (no neighborhood search performed) the sort
		is done from scratch using merge sort on the points.
		If the octree exists, we can use merge sort on the cells (instead of on the points)
		and then find the new point indices concatenating the cell's points in order.
	*/

	// Minimum number of points for zsort
	if (this->get_total_n_points() < this->number_of_too_few_particles) {
		this->are_cells_valid = false; // Ensures we do a global zsort
	}

	// Set up
	this->_set_up();

	// Prepare solution z index map
	this->zsort_set_new_to_old_map.resize(this->n_sets);
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		this->zsort_set_new_to_old_map[set_i].resize(this->get_n_points_in_set(set_i));
	}

	// Tree has never been built. Since we cannot assume almost-zsortness, we do a global zsort
	if (!this->are_cells_valid) {
		this->_compute_zsort_order_notree();
	}
	
	// The tree was built and we can reuse its structure
	else {
		// Allocate
		std::vector<std::vector<std::pair<int, uint_fast32_t>>> thread_zorder_buffer(this->n_threads);
		std::vector<std::pair<int, uint_fast32_t>> cell_idx_zidx_pairs;
		int begin_cell = 0;
		int end_cell = -1;
		for (int set_i = 0; set_i < this->n_sets; set_i++) {
			const int set_offset = this->set_offsets[set_i];
			std::vector<int>& zsort_new_to_old_map = this->zsort_set_new_to_old_map[set_i];

			// Binary search set bounds in the cell list
			end_cell = (int)std::distance(this->cells.offsets, std::lower_bound(this->cells.offsets + begin_cell, this->cells.offsets + this->cells.n_cells, this->set_offsets[set_i + 1]));
			const int set_n_cells = end_cell - begin_cell;

			// Z index of the cells
			cell_idx_zidx_pairs.resize(set_n_cells);
			#pragma omp parallel for schedule(static) num_threads(this->n_threads)
			for (int cell_i = 0; cell_i < set_n_cells; cell_i++) {
				const int c = begin_cell + cell_i;
				const uint_fast32_t z_index = libmorton::morton3D_32_encode(this->cells.i[c], this->cells.j[c], this->cells.k[c]);
				cell_idx_zidx_pairs[cell_i] = { cell_i, z_index };
			}

			// Sort the cells
			tf::Executor executor(this->n_threads);
			tf::Taskflow taskflow;
			taskflow.sort(cell_idx_zidx_pairs.begin(), cell_idx_zidx_pairs.end(), [](const std::pair<int, uint_fast32_t>& a, const std::pair<int, uint_fast32_t>& b) {return a.second < b.second; });
			executor.run(taskflow).wait();

			// Prefix sum
			std::vector<int> cell_offsets(set_n_cells + 1);
			cell_offsets[0] = 0;
			std::transform_inclusive_scan(
				cell_idx_zidx_pairs.begin(), cell_idx_zidx_pairs.end(), cell_offsets.begin() + 1, std::plus<int>{},
				[begin_cell, this](const std::pair<int, uint_fast32_t>& pair)
				{ 
					const int c = begin_cell + pair.first;
					return this->cells.offsets[c + 1] - this->cells.offsets[c];
				}
			);
			
			// Insert the points indices in order
			#pragma omp parallel for schedule(static) num_threads(this->n_threads)
			for (int new_cell_i = 0; new_cell_i < set_n_cells; new_cell_i++) {
				const int old_cell_i = cell_idx_zidx_pairs[new_cell_i].first;
				const int c = begin_cell + old_cell_i;
				const int points_begin = this->cells.offsets[c];
				const int points_end = this->cells.offsets[c+1];
				const int n_points = points_end - points_begin;
				for (int loc_point_i = 0; loc_point_i < n_points; loc_point_i++) {
					const int set_point_i = points_begin + loc_point_i - set_offset;
					const int old_idx = set_point_i;
					const int new_idx = cell_offsets[new_cell_i] + loc_point_i;

					zsort_new_to_old_map[new_idx] = old_idx;
				}
			}

			begin_cell = end_cell;
		}

		// Since the cells are no longer idx sorted, we have to destroy them
		this->are_cells_valid = false;
	}
}
void tns::TreeNSearch::_compute_zsort_order_notree()
{
	// World AABB
	this->_update_world_AABB_simd();
	const std::array<float, 3>& bottom = this->domain_float.bottom;
	const std::array<float, 3>& top = this->domain_float.top;
	const float world_size = top[0] - bottom[0];

	std::vector<std::pair<int, uint_fast64_t>> point_idx_cell_zidx_pairs;
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		const float* points = this->set_points[set_i];
		const int n_points = this->get_n_points_in_set(set_i);
		point_idx_cell_zidx_pairs.resize(n_points);

		// Cell size
		//// To match the octree boundaries, the quantization size for the zsort has to
		//// be power of two multiple of the cell size. Essentially the same octree but
		//// with finer resolution.
		constexpr uint32_t END_CELL = 2097152; // 2^(64//3)
		constexpr uint32_t N_CELLS_PER_DIMESION = END_CELL - 1;
		const float world_size = this->domain_float.top[0] - this->domain_float.bottom[0];
		float cell_size = this->cell_size;
		while (world_size / (cell_size / 2.0f) < (float)N_CELLS_PER_DIMESION) {
			cell_size /= 2.0f;
		}
		const float cell_size_inv = 1.0f / cell_size;
		
		// z index
		#pragma omp parallel for schedule(static) num_threads(this->n_threads)
		for (int point_i = 0; point_i < n_points; point_i++) {
			const uint_fast64_t z_index = libmorton::morton3D_64_encode(
				(uint_fast32_t)((points[3 * point_i + 0] - bottom[0]) * cell_size_inv),
				(uint_fast32_t)((points[3 * point_i + 1] - bottom[1]) * cell_size_inv),
				(uint_fast32_t)((points[3 * point_i + 2] - bottom[2]) * cell_size_inv)
			);
			point_idx_cell_zidx_pairs[point_i] = { point_i, z_index };
		}

		// Sort
		tf::Executor executor(this->n_threads);
		tf::Taskflow taskflow;
		taskflow.sort(point_idx_cell_zidx_pairs.begin(), point_idx_cell_zidx_pairs.end(), [](const std::pair<int, uint_fast64_t>& a, const std::pair<int, uint_fast64_t>& b) {return a.second < b.second; });
		executor.run(taskflow).wait();

		// Write the mapping
		std::vector<int>& new_to_old_map = this->zsort_set_new_to_old_map[set_i];
		#pragma omp parallel for schedule(static) num_threads(this->n_threads)
		for (int point_i = 0; point_i < n_points; point_i++) {
			const int new_idx = point_i;
			const int old_idx = point_idx_cell_zidx_pairs[point_i].first;
			new_to_old_map[new_idx] = old_idx;
		}
	}
}

void tns::TreeNSearch::print_state() const
{
	auto print3f = [](const float* p) {std::cout << "[" << p[0] << ", " << p[1] << ", " << p[2] << "]" << std::endl; };
	auto print3u16 = [](const uint16_t* p) {std::cout << "[" << p[0] << ", " << p[1] << ", " << p[2] << "]" << std::endl; };
	auto print3int = [](const int* p) {std::cout << "[" << p[0] << ", " << p[1] << ", " << p[2] << "]" << std::endl; };
	auto print_bool = [](const bool b) { std::cout << ((b) ? "true" : "false") << std::endl; };
	auto min_max_sum_init = []() { return std::array<int, 3>({ std::numeric_limits<int>::max(), std::numeric_limits<int>::lowest(), 0 }); };
	auto min_max_sum = [](std::array<int, 3>& arr, const int n) {arr[0] = std::min(arr[0], n); arr[1] = std::max(arr[1], n); arr[2] += n; };
	auto min_max_sum_print = [](std::array<int, 3>& arr, const int n) { std::cout << "[" << arr[0] << ", " << arr[1] << ", " << (double)arr[2] / (double)n << "]" << std::endl; };

	std::cout << "\n ================ OPTIONS ================ " << std::endl;
	std::cout << "symmetric_search: "; print_bool(this->symmetric_search);
	std::cout << "is_global_search_radius_set: "; print_bool(this->is_global_search_radius_set);
	std::cout << "domain_enlargment: " << this->domain_enlargment << std::endl;
	std::cout << "n_points_to_stop_recursion: " << this->n_points_to_stop_recursion << std::endl;
	std::cout << "n_threads: " << this->n_threads << std::endl;

	std::cout << "\n ================ GRID/OCTREE ================ " << std::endl;
	std::cout << "World AABB float" << std::endl;
	print3f(&this->domain_float.bottom[0]);
	print3f(&this->domain_float.top[0]);
	std::cout << std::endl;

	std::cout << "World AABB uint16" << std::endl;
	print3u16(&this->octree_root.buffer.domain.bottom[0]);
	print3u16(&this->octree_root.buffer.domain.top[0]);
	std::cout << std::endl;

	if (this->is_global_search_radius_set) {
		std::cout << "global_search_radius: " << this->global_search_radius << std::endl;
	}
	else {
		std::cout << "max_search_radius: " << this->max_search_radius << std::endl;
	}
	std::cout << "cell_size: " << this->cell_size << std::endl;

	// Cells
	std::array<int, 3> points_in_cells = min_max_sum_init();
	for (int cell_i = 0; cell_i < this->cells.n_cells; cell_i++) {
		const int n = this->cells.offsets[cell_i + 1] - this->cells.offsets[cell_i];
		min_max_sum(points_in_cells, n);
	}
	std::cout << "# cells: " << this->cells.n_cells << std::endl;
	std::cout << "points_per_cell [min, max, avg]: "; min_max_sum_print(points_in_cells, (int)this->cells.n_cells);
	
	// Leaves
	std::array<int, 3> leaf_sizes = min_max_sum_init();
	std::array<int, 3> cells_in_leaf = min_max_sum_init();
	std::array<int, 3> points_in_leaf = min_max_sum_init();
	std::array<int, 3> interior_points_in_leaf = min_max_sum_init();
	for (const OctreeNode* leaf : this->leaves) {
		const int n_cells = leaf->cell_indices.size();

		min_max_sum(leaf_sizes, (int)(leaf->domain.top[0] - leaf->domain.bottom[0]));
		min_max_sum(cells_in_leaf, n_cells);

		int n_total = 0;
		int n_interior = 0;
		for (int cell_i = 0; cell_i < n_cells; cell_i++) {
			const int c = leaf->cell_indices[cell_i];
			const int n_points = this->cells.offsets[c + 1] - this->cells.offsets[c];
			const uint8_t inside_cell =
				(leaf->domain.bottom[0] <= this->cells.i[c] && this->cells.i[c] < leaf->domain.top[0] &&
				 leaf->domain.bottom[1] <= this->cells.j[c] && this->cells.j[c] < leaf->domain.top[1] &&
				 leaf->domain.bottom[2] <= this->cells.k[c] && this->cells.k[c] < leaf->domain.top[2]);
			n_total += n_points;
			if (inside_cell) {
				n_interior += n_points;
			}
		}
		min_max_sum(points_in_leaf, n_total);
		min_max_sum(interior_points_in_leaf, n_interior);
	}
	std::cout << "# leaves: " << this->leaves.size() << std::endl;
	std::cout << "leaf_sizes [min, max, avg]: "; min_max_sum_print(leaf_sizes, (int)this->leaves.size());
	std::cout << "cells_in_leaf [min, max, avg]: ";  min_max_sum_print(cells_in_leaf, (int)this->leaves.size());
	std::cout << "points_in_leaf [min, max, avg]: ";  min_max_sum_print(points_in_leaf, (int)this->leaves.size());
	std::cout << "interior_points_in_leaf [min, max, avg]: ";  min_max_sum_print(interior_points_in_leaf, (int)this->leaves.size());

	// Global ghost cell
	uint16_t max_ghost_cells;
	const float ghost_cells_float = this->max_search_radius / this->cell_size;
	const float remainder = ghost_cells_float - std::round(ghost_cells_float);
	if (std::abs(remainder) < 2 * std::numeric_limits<float>::epsilon()) {
		max_ghost_cells = (uint16_t)std::round(ghost_cells_float);
	}
	else {
		max_ghost_cells = (uint16_t)(ghost_cells_float)+1;
	}
	std::cout << "max_ghost_cell_length: " << max_ghost_cells << std::endl;


	std::cout << "\n ================ NEIGHBORLISTS ================ " << std::endl;
	std::cout << "Active searches: " << std::endl;
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		for (const int set_j : this->active_searches[set_i]) {
			std::cout << "\t" << "set_" << set_i << " -> " << "set_" << set_j << std::endl;
		}
	}
	std::cout << "Total memory (MB): " << (double)this->get_neighborlist_n_bytes()/1024.0/1024.0 << std::endl;


	std::cout << "\n ================ PER SET DATA ================ " << std::endl;
	for (int set_i = 0; set_i < this->n_sets; set_i++) {
		std::cout << "\n ---------------- set_" << set_i << " ---------------- " << std::endl;
		std::cout << "# points: " << this->get_n_points_in_set(set_i) << std::endl;

		// Set AABB
		std::array<float, 3> b = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
		std::array<float, 3> t = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };
		for (int i = 0; i < this->get_n_points_in_set(set_i); i++) {
			for (int dim = 0; dim < 3; dim++) {
				b[dim] = std::min(b[dim], this->set_points[set_i][3*i + dim]);
				t[dim] = std::max(t[dim], this->set_points[set_i][3*i + dim]);
			}
		}
		std::cout << "AABB float" << std::endl;
		print3f(&b[0]);
		print3f(&t[0]);

		// Search radius
		if (!this->is_global_search_radius_set) {
			float min_search_radius = std::numeric_limits<float>::max();
			float max_search_radius = std::numeric_limits<float>::lowest();
			double sum_search_radius = 0.0;
			for (int i = 0; i < this->get_n_points_in_set(set_i); i++) {
				min_search_radius = std::min(min_search_radius, this->set_radii[set_i][i]);
				max_search_radius = std::max(max_search_radius, this->set_radii[set_i][i]);
				sum_search_radius += (double)this->set_radii[set_i][i];
			}
			std::cout << "search_radius [min, max, avg]: [" << min_search_radius << ", " << max_search_radius << ", " << sum_search_radius / (double)this->get_n_points_in_set(set_i) << "]" << std::endl;
		}

		// Neighbors count
		for (const int set_j : this->active_searches[set_i]) {
			bool id_valid_neighborlist = true;
			for (int i = 0; i < this->get_n_points_in_set(set_i); i++) {
				const int* neighborlist = this->solution_ptr[this->_get_set_pair_id(set_i, set_j)][i];
				if (neighborlist == nullptr) {
					id_valid_neighborlist = false;
					break;
				}
			}

			if (id_valid_neighborlist) {
				std::array<int, 3> n_neighbors = min_max_sum_init();
				for (int i = 0; i < this->get_n_points_in_set(set_i); i++) {
					min_max_sum(n_neighbors, this->get_neighborlist(set_i, set_j, i).size());
				}
				std::cout << "n_neighbors set_" << set_i << " -> " << "set_" << set_j << " [min, max, avg]: "; min_max_sum_print(n_neighbors, this->get_n_points_in_set(set_i));
			}
			else {
				std::cout << "n_neighbors set_" << set_i << " -> " << "set_" << set_j << " [min, max, avg]: Invalid neighborlist. Probably not finished computing." << std::endl;
			}
		}
	}
}

