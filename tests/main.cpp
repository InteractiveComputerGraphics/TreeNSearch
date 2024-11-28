#include "tests.h"


void tests(int n_points, bool crash_at_first_discrepancy)
{
	std::cout << "\n\nTests with " << n_points << " particles." << std::endl;
	std::cout << "======================================" << std::endl;
	one_set_fixed_radius(n_points, crash_at_first_discrepancy);
	two_dynamic_sets_variable_radius(n_points, crash_at_first_discrepancy);
	mixed_float_double_point_sets(n_points, crash_at_first_discrepancy);
	resize_variable_radius(n_points, crash_at_first_discrepancy);
}

int main()
{
	// Note: Do not run TreeNSearch for 0 particles. 
	// It will give a warning when run, but it will crash when queried 
	// (e.g. unexisting particle neighbors or zsort an empty set).

	bool crash_at_first_discrepancy = false;

	// Tests
	tests(/* n_particles = */ 1, crash_at_first_discrepancy);
	tests(/* n_particles = */ 100, crash_at_first_discrepancy);
	tests(/* n_particles = */ 10000, crash_at_first_discrepancy);

	// Benchmark
	const int n_benchmark_points = 9000;
	benchmark_one_dynamic_set(n_benchmark_points);
}

