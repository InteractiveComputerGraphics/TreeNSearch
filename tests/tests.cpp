#include "tests.h"

struct Points
{
	float particle_radius;
	float search_radius;
	std::vector<std::array<float, 3>> points;
	int n = 0;
};

Points generate_point_grid_as_SPH(const std::array<float, 3>& bottom, const std::array<float, 3>& top, const float sampling_distance)
{
	Points points;
	const float particle_diameter = sampling_distance;
	points.particle_radius = sampling_distance/2.0f;
	points.search_radius = 1.99f * particle_diameter;

	for (float x = bottom[0]; x <= top[0]; x += particle_diameter) {
		for (float y = bottom[1]; y <= top[1]; y += particle_diameter) {
			for (float z = bottom[2]; z <= top[2]; z += particle_diameter) {
				points.points.push_back({ x, y, z });
			}
		}
	}
	points.n = (int)points.points.size();
	return points;
}

void _compare_tns_with_bruteforce(Points& points, tns::TreeNSearch& nsearch, BruteforceNSearch& bruteforce, bool report_when_a_test_fails)
{
	// Comparison
	bruteforce.run();
	nsearch.run_scalar();
	std::cout << "\tNeighborhood search scalar... " << ((bruteforce.compare(nsearch, report_when_a_test_fails)) ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;
	nsearch.run();
	std::cout << "\tNeighborhood search SIMD... " << ((bruteforce.compare(nsearch, report_when_a_test_fails)) ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;

	// Zsort
	nsearch.prepare_zsort();
	nsearch.apply_zsort(0, points.points[0].data(), 3);
	bruteforce.run();
	nsearch.run();
	std::cout << "\tZsort... " << ((bruteforce.compare(nsearch)) ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;

	// Threads
	bool threads_success = true;
	for (int n_threads = 1; n_threads < 12; n_threads++) {
		nsearch.set_n_threads(n_threads);
		nsearch.run();
		threads_success = threads_success && bruteforce.compare(nsearch, report_when_a_test_fails);
	}
	std::cout << "\tDifferent n_threads SIMD... " << (threads_success ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;
	nsearch.set_n_threads(omp_get_max_threads());

	threads_success = true;
	for (int n_threads = 1; n_threads < 12; n_threads++) {
		nsearch.set_n_threads(n_threads);
		nsearch.run_scalar();
		threads_success = threads_success && bruteforce.compare(nsearch, report_when_a_test_fails);
	}
	std::cout << "\tDifferent n_threads scalar... " << (threads_success ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;
	nsearch.set_n_threads(omp_get_max_threads());

	// Cell size
	bool cell_size_success = true;
	for (float cell_size = points.search_radius; cell_size < 3.0f * points.search_radius; cell_size += 0.2f * points.search_radius) {
		nsearch.set_cell_size(cell_size);
		nsearch.run();
		cell_size_success = cell_size_success && bruteforce.compare(nsearch, report_when_a_test_fails);
	}
	std::cout << "\tDifferent cell_size SIMD... " << (cell_size_success ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;
	nsearch.set_cell_size(1.5f * points.search_radius);

	cell_size_success = true;
	for (float cell_size = points.search_radius; cell_size < 3.0f * points.search_radius; cell_size += 0.2f * points.search_radius) {
		nsearch.set_cell_size(cell_size);
		nsearch.run_scalar();
		cell_size_success = cell_size_success && bruteforce.compare(nsearch, report_when_a_test_fails);
	}
	std::cout << "\tDifferent cell_size scalar... " << (cell_size_success ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;
	nsearch.set_cell_size(1.5f * points.search_radius);

	// Recursion cap
	bool recursion_cap_success = true;
	for (int recursion_cap = 100; recursion_cap < 2000; recursion_cap += 100) {
		nsearch.set_recursion_cap(recursion_cap);
		nsearch.run();
		recursion_cap_success = recursion_cap_success && bruteforce.compare(nsearch, report_when_a_test_fails);
	}
	std::cout << "\tDifferent recursion_cap SIMD... " << (recursion_cap_success ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;
	nsearch.set_recursion_cap(1000);

	recursion_cap_success = true;
	for (int recursion_cap = 100; recursion_cap < 2000; recursion_cap += 100) {
		nsearch.set_recursion_cap(recursion_cap);
		nsearch.run_scalar();
		recursion_cap_success = recursion_cap_success && bruteforce.compare(nsearch, report_when_a_test_fails);
	}
	std::cout << "\tDifferent recursion_cap scalar... " << (recursion_cap_success ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;
	nsearch.set_recursion_cap(1000);

	std::cout << std::endl;
}

void one_set_fixed_radius(const int n_points, bool report_when_a_test_fails)
{
	std::cout << "One point set. Fixed search radius." << std::endl;

	// Create particles
	const float particle_radius = (float)(2.0/std::pow((double)n_points, 1.0/3.0));
	Points points = generate_point_grid_as_SPH({ -1, -1, -1 }, { 1, 1, 1 }, particle_radius);

	// BruteforceNSearch
	BruteforceNSearch bruteforce;
	const int set_0 = bruteforce.add_point_set(points.points[0].data(), points.search_radius, points.n);
	bruteforce.set_active_search(set_0, set_0, true);

	// Fixed radius
	tns::TreeNSearch nsearch;
	nsearch.set_search_radius(points.search_radius);
	nsearch.add_point_set(points.points[0].data(), points.n);
	nsearch.set_active_search(set_0, set_0, true);

	// Compare
	_compare_tns_with_bruteforce(points, nsearch, bruteforce, report_when_a_test_fails);
}

void two_dynamic_sets_variable_radius(const int n_points, bool report_when_a_test_fails)
{
	std::cout << "Two point sets. Variable search radius." << std::endl;

	// Create particles
	const float particle_radius = (float)(2.0 / std::pow((double)n_points, 1.0 / 3.0));
	Points points_0 = generate_point_grid_as_SPH({ -1, -1, -1 }, { 1, 1, 1 }, particle_radius);
	Points points_1 = generate_point_grid_as_SPH({ -1, -1, -1 }, { 1, 1, 1 }, 1.31f*particle_radius);
	std::vector<float> radii_0(points_0.points.size(), points_0.search_radius);
	std::vector<float> radii_1(points_1.points.size(), points_1.search_radius);

	// BruteforceNSearch
	BruteforceNSearch bruteforce;
	const int set_0 = bruteforce.add_point_set(points_0.points[0].data(), points_0.search_radius, points_0.n);
	const int set_1 = bruteforce.add_point_set(points_1.points[0].data(), points_1.search_radius, points_1.n);

	bruteforce.set_active_search(set_0, set_0, true);
	bruteforce.set_active_search(set_0, set_1, true);
	bruteforce.set_active_search(set_1, set_0, true);

	// Fixed radius
	tns::TreeNSearch nsearch;
	nsearch.add_point_set(points_0.points[0].data(), radii_0.data(), points_0.n);
	nsearch.add_point_set(points_1.points[0].data(), radii_1.data(), points_1.n);

	nsearch.set_active_search(set_0, set_0, true);
	nsearch.set_active_search(set_0, set_1, true);
	nsearch.set_active_search(set_1, set_0, true);

	// Compare
	_compare_tns_with_bruteforce(points_0, nsearch, bruteforce, report_when_a_test_fails);
}

void mixed_float_double_point_sets(const int n_points, bool report_when_a_test_fails)
{
	std::cout << "Two point sets. One of type float another double. Variable search radius." << std::endl;

	// Create particles
	const float particle_radius = (float)(2.0 / std::pow((double)n_points, 1.0 / 3.0));
	Points points_0 = generate_point_grid_as_SPH({ -1, -1, -1 }, { 1, 1, 1 }, particle_radius);
	Points points_1 = generate_point_grid_as_SPH({ -1, -1, -1 }, { 1, 1, 1 }, 1.33f * particle_radius);
	std::vector<float> radii_0(points_0.points.size(), points_0.search_radius);
	std::vector<float> radii_1(points_1.points.size(), points_1.search_radius);

	// To double
	std::vector<std::array<double, 3>> points_1_double(points_0.n);
	std::vector<double> radii_1_double(points_0.n);
	for (int i = 0; i < points_1.n; i++) {
		points_1_double[i] = { (double)points_1.points[i][0], (double)points_1.points[i][1], (double)points_1.points[i][2] };
		radii_1_double[i] = (double)radii_1[i];
	}

	// BruteforceNSearch
	BruteforceNSearch bruteforce;
	const int setd_0 = bruteforce.add_point_set(points_0.points[0].data(), points_0.search_radius, points_0.n);
	const int setd_1 = bruteforce.add_point_set(points_1.points[0].data(), points_1.search_radius, points_1.n);

	bruteforce.set_active_search(setd_0, setd_0, true);
	bruteforce.set_active_search(setd_0, setd_1, true);
	bruteforce.set_active_search(setd_1, setd_0, true);

	// Fixed radius
	tns::TreeNSearch nsearch;
	nsearch.add_point_set(points_0.points[0].data(), radii_0.data(), points_0.n);  // float
	nsearch.add_point_set(points_1_double[0].data(), radii_1_double.data(), points_1.n);  // double

	nsearch.set_active_search(setd_0, setd_0, true);
	nsearch.set_active_search(setd_0, setd_1, true);
	nsearch.set_active_search(setd_1, setd_0, true);

	// Compare
	_compare_tns_with_bruteforce(points_0, nsearch, bruteforce, report_when_a_test_fails);
}

void resize_variable_radius(const int n_points, bool report_when_a_test_fails)
{
	std::cout << "Two dynamic point sets. Multiple iterations with resizes in between. Variable search radius." << std::endl;

	// Create particles
	const float particle_radius = (float)(2.0 / std::pow((double)n_points, 1.0 / 3.0));
	Points points_0 = generate_point_grid_as_SPH({ -1, -1, -1 }, { 1, 1, 1 }, particle_radius);
	Points points_1 = generate_point_grid_as_SPH({ -1, -1, -1 }, { 1, 1, 1 }, 1.31f * particle_radius);
	std::vector<float> radii_0(points_0.points.size(), points_0.search_radius);
	std::vector<float> radii_1(points_1.points.size(), points_1.search_radius);

	// BruteforceNSearch
	BruteforceNSearch bruteforce;
	const int set_0 = bruteforce.add_point_set(points_0.points[0].data(), radii_0.data(), points_0.n / 2);
	const int set_1 = bruteforce.add_point_set(points_1.points[0].data(), radii_1.data(), points_1.n/2);

	bruteforce.set_active_search(set_0, set_0, true);
	bruteforce.set_active_search(set_0, set_1, true);
	bruteforce.set_active_search(set_1, set_0, true);

	// Fixed radius
	tns::TreeNSearch nsearch;
	nsearch.add_point_set(points_0.points[0].data(), radii_0.data(), points_0.n/2);
	nsearch.add_point_set(points_1.points[0].data(), radii_1.data(), points_1.n/2);

	nsearch.set_active_search(set_0, set_0, true);
	nsearch.set_active_search(set_0, set_1, true);
	nsearch.set_active_search(set_1, set_0, true);

	// Compare
	bruteforce.run();
	nsearch.run();
	std::cout << "\tOriginal neighborhood search... " << ((bruteforce.compare(nsearch)) ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;

	bruteforce.resize_point_set(set_0, points_0.points[0].data(), radii_0.data(), points_0.n);
	bruteforce.resize_point_set(set_1, points_1.points[0].data(), radii_1.data(), points_1.n);
	nsearch.resize_point_set(set_0, points_0.points[0].data(), radii_0.data(), points_0.n);
	nsearch.resize_point_set(set_1, points_1.points[0].data(), radii_1.data(), points_1.n);
	bruteforce.run();
	nsearch.run();
	std::cout << "\tResize x2... " << ((bruteforce.compare(nsearch)) ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;

	bruteforce.resize_point_set(set_0, points_0.points[0].data(), radii_0.data(), points_0.n/3);
	bruteforce.resize_point_set(set_1, points_1.points[0].data(), radii_1.data(), points_1.n/3);
	nsearch.resize_point_set(set_0, points_0.points[0].data(), radii_0.data(), points_0.n/3);
	nsearch.resize_point_set(set_1, points_1.points[0].data(), radii_1.data(), points_1.n/3);
	bruteforce.run();
	nsearch.run();
	std::cout << "\tResize x0.33... " << ((bruteforce.compare(nsearch)) ? "passed!" : "xxxxxxx FAILED! xxxxxxx") << std::endl;
}

void benchmark_one_dynamic_set(const int n_points)
{
	std::cout << "Benchmark: one dynamic point set. Fixed search radius." << std::endl;

	// Create particles
	const float particle_radius = (float)(2.0 / std::pow((double)n_points, 1.0 / 3.0));
	Points points = generate_point_grid_as_SPH({ -1, -1, -1 }, { 1, 1, 1 }, particle_radius);
	std::cout << "\tNumber of particles: " << points.n << std::endl;

	// Fixed radius
	tns::TreeNSearch nsearch;
	nsearch.set_search_radius(points.search_radius);
	const int set_0 = nsearch.add_point_set(points.points[0].data(), points.n);
	nsearch.set_active_search(set_0, set_0, true);

	// Zsort
	nsearch.prepare_zsort();
	nsearch.apply_zsort(0, points.points[0].data(), 3);

	// Benchmark
	const int benchmark_iterations = 1000;
	double t0, t1;

	//// Scalar
	nsearch.run_scalar();
	t0 = omp_get_wtime();
	for (int it = 0; it < benchmark_iterations; it++) {
		nsearch.run_scalar();
	}
	t1 = omp_get_wtime();
	std::cout << "\tRuntime parallel Scalar: " << 1000.0*(t1 - t0)/(double)benchmark_iterations << " ms." << std::endl;

	//// SIMD
	nsearch.run_scalar();
	t0 = omp_get_wtime();
	for (int it = 0; it < benchmark_iterations; it++) {
		nsearch.run();
	}
	t1 = omp_get_wtime();
	std::cout << "\tRuntime parallel SIMD: " << 1000.0*(t1 - t0)/(double)benchmark_iterations << " ms." << std::endl;
}
