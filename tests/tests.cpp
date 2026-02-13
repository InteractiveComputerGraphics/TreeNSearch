#include "tests.h"
#include <functional>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>

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

/*
    Stress test that iterates over different numbers of threads, sets, and particle counts.
    It generates all possible combinations of particle counts for the given sets and verifies
    correctness against the brute-force implementation.
    This is useful for catching edge cases related to empty sets, small sets, and thread boundaries.
*/
void combinatorial_stress_test()
{
    std::cout << "Combinatorial Stress Test..." << std::endl;

	constexpr bool COMPARE_WITH_BRUTEFORCE = false; // False to speed up the stress test looking for segfaults.
    std::vector<int> thread_counts = {1, 2, 4, 7, 8, 9, 12, 24};
    std::vector<int> set_counts = {1, 2, 3};

    // Fixed seed for reproducibility
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_coord(0.0f, 10.0f);
    
    for (int n_threads : thread_counts) {
        for (int n_sets : set_counts) {
            
            std::vector<int> particle_counts = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 100, 1000, 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009};

            // Define interesting particle counts based on n_threads
            if (n_threads > 1) {
                particle_counts.push_back(n_threads - 1);
                particle_counts.push_back(n_threads);
                particle_counts.push_back(n_threads + 1);
                particle_counts.push_back(2 * n_threads);
            }
            // Remove duplicates and sort
            std::sort(particle_counts.begin(), particle_counts.end());
            particle_counts.erase(std::unique(particle_counts.begin(), particle_counts.end()), particle_counts.end());

            // Generate all combinations of particle counts for the sets
            std::vector<std::vector<int>> test_cases_counts;
            std::vector<int> current_counts(n_sets);
            
            std::function<void(int)> generate_combinations = [&](int set_idx) {
                if (set_idx == n_sets) {
                    test_cases_counts.push_back(current_counts);
                    return;
                }
                for (int count : particle_counts) {
                    current_counts[set_idx] = count;
                    generate_combinations(set_idx + 1);
                }
            };
            generate_combinations(0);
            
            for (const auto& counts : test_cases_counts) {
                std::stringstream ss;
                ss << "Threads: " << n_threads << ", Sets: " << n_sets << ", Counts: [";
                for(size_t i=0; i<counts.size(); ++i) ss << counts[i] << (i<counts.size()-1 ? ", " : "");
                ss << "]";
                std::cout << "\tTesting: " << ss.str() << std::endl;
                
                tns::TreeNSearch nsearch;
                std::unique_ptr<BruteforceNSearch> bruteforce = nullptr;
                if (COMPARE_WITH_BRUTEFORCE) {
                    bruteforce = std::make_unique<BruteforceNSearch>();
                }
                
                nsearch.set_n_threads(n_threads);
                if (COMPARE_WITH_BRUTEFORCE) {
                    bruteforce->set_n_threads(n_threads);
                }
                
                std::vector<std::vector<float>> points_storage(n_sets);
                std::vector<std::vector<float>> radii_storage(n_sets);
                
                for(int s=0; s<n_sets; ++s) {
                    int n = counts[s];
                    points_storage[s].resize(3 * n);
                    radii_storage[s].resize(n);
                    for(int i=0; i<n; ++i) {
                        points_storage[s][3*i+0] = dist_coord(gen);
                        points_storage[s][3*i+1] = dist_coord(gen);
                        points_storage[s][3*i+2] = dist_coord(gen);
                        radii_storage[s][i] = 0.5f + 0.5f * dist_coord(gen) / 10.0f; // 0.5 to 1.0
                    }
                    
                    if (n > 0) {
                        nsearch.add_point_set(points_storage[s].data(), radii_storage[s].data(), n);
                        if (COMPARE_WITH_BRUTEFORCE) {
                            bruteforce->add_point_set(points_storage[s].data(), radii_storage[s].data(), n);
                        }
                    } else {
                        nsearch.add_point_set((float*)nullptr, (float*)nullptr, 0);
                        if (COMPARE_WITH_BRUTEFORCE) {
                            bruteforce->add_point_set((float*)nullptr, (float*)nullptr, 0);
                        }
                    }
                }
                
                // Activate all searches
                nsearch.set_all_searches(true);
                if (COMPARE_WITH_BRUTEFORCE) {
                    bruteforce->set_all_searches(true);
                }
                
                // Run TNS
                nsearch.run();
                
                // Run Bruteforce
                if (COMPARE_WITH_BRUTEFORCE) {
                    bruteforce->run();
                    
                    // Compare
                    if (!bruteforce->compare(nsearch, true)) {
                        std::cout << "FAILED: " << ss.str() << std::endl;
                        exit(-1);
                    }
                }
                
                // ZSort
                nsearch.prepare_zsort();
                for(int s=0; s<n_sets; ++s) {
                    if (counts[s] > 0) {
                        nsearch.apply_zsort(s, points_storage[s].data(), 3);
                        nsearch.apply_zsort(s, radii_storage[s].data(), 1);

                        // Update Bruteforce radii copy
                        if (COMPARE_WITH_BRUTEFORCE) {
                            bruteforce->resize_point_set(s, points_storage[s].data(), radii_storage[s].data(), counts[s]);
                        }
                    }
                }
                
                // Re-run TNS
                nsearch.run();
                
                // Re-run Bruteforce (data has changed in place)
                if (COMPARE_WITH_BRUTEFORCE) {
                    bruteforce->run();
                    
                    // Compare
                    if (!bruteforce->compare(nsearch, true)) {
                        std::cout << "FAILED (after ZSort): " << ss.str() << std::endl;
                        exit(-1);
                    }
                }
            }
        }
    }
    std::cout << "Combinatorial Stress Test Passed!" << std::endl;
}

/*
    Stress test that simulates a dynamic environment where particles are added, removed, or resized
    randomly over many iterations. This verifies the stability of the library under dynamic changes
    and ensures that internal structures are correctly updated or rebuilt.
*/
void dynamic_emitter_stress_test()
{
    std::cout << "Dynamic Emitter Stress Test..." << std::endl;
    
    int n_sets = 2;
    int n_threads = 8;
    int iterations = 10000;
    
    tns::TreeNSearch nsearch;
    BruteforceNSearch bruteforce;
    
    nsearch.set_n_threads(n_threads);
    bruteforce.set_n_threads(n_threads);
    
    std::vector<std::vector<float>> points_storage(n_sets);
    std::vector<std::vector<float>> radii_storage(n_sets);
    
    // Init empty
    for(int s=0; s<n_sets; ++s) {
        nsearch.add_point_set((float*)nullptr, (float*)nullptr, 0);
        bruteforce.add_point_set((float*)nullptr, (float*)nullptr, 0);
    }
    nsearch.set_all_searches(true);
    bruteforce.set_all_searches(true);
    
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist_coord(0.0f, 10.0f);
    std::uniform_int_distribution<int> dist_set(0, n_sets - 1);
    std::uniform_int_distribution<int> dist_action(0, 2); // 0: add, 1: remove, 2: replace
    std::uniform_int_distribution<int> dist_amount(1, 20);
    
    for(int iter=0; iter<iterations; ++iter) {
        int s = dist_set(gen);
        int action = dist_action(gen);
        int amount = dist_amount(gen);
        
        int current_n = (int)points_storage[s].size() / 3;
        int new_n = current_n;
        
        if (action == 0) { // Add
            new_n += amount;
        } else if (action == 1) { // Remove
            new_n = std::max(0, new_n - amount);
        } else { // Replace (resize to random)
             new_n = amount; // Just set to amount
        }
        
        // Resize storage
        points_storage[s].resize(3 * new_n);
        radii_storage[s].resize(new_n);
        
        // Fill new points if added
        // Actually, just refill everything to be safe/random
        for(int i=0; i<new_n; ++i) {
             points_storage[s][3*i+0] = dist_coord(gen);
             points_storage[s][3*i+1] = dist_coord(gen);
             points_storage[s][3*i+2] = dist_coord(gen);
             radii_storage[s][i] = 0.5f;
        }
        
        std::cout << "\tIter " << iter << ": Set " << s << " -> " << new_n << " points." << std::endl;
        
        // Update TNS
        if (new_n > 0) {
            nsearch.resize_point_set(s, points_storage[s].data(), radii_storage[s].data(), new_n);
            bruteforce.resize_point_set(s, points_storage[s].data(), radii_storage[s].data(), new_n);
        } else {
            nsearch.resize_point_set(s, (float*)nullptr, (float*)nullptr, 0);
            bruteforce.resize_point_set(s, (float*)nullptr, (float*)nullptr, 0);
        }
        
        nsearch.run();
        bruteforce.run();
        
        if (!bruteforce.compare(nsearch, true)) {
             std::cout << "FAILED Dynamic Emitter at iter " << iter << std::endl;
             exit(-1);
        }
    }
    std::cout << "Dynamic Emitter Stress Test Passed!" << std::endl;
}

