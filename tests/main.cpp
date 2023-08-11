#include "tests.h"


int main()
{
	const int n_points = 1;
	bool report_when_a_test_fails = true;

	// Tests
	one_set_fixed_radius(n_points, report_when_a_test_fails);
	two_dynamic_sets_variable_radius(n_points, report_when_a_test_fails);
	mixed_float_double_point_sets(n_points, report_when_a_test_fails);
	resize_variable_radius(n_points, report_when_a_test_fails);

	// Benchmarks
	const int n_benchmark_points = 9000;
	benchmark_one_dynamic_set(n_benchmark_points);
}

