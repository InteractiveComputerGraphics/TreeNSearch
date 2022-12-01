#pragma once
#include <vector>
#include <array>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <omp.h>

#include <TreeNSearch>

#include "BruteforceNSearch.h"


void one_set_fixed_radius(const int n_points, bool report_when_a_test_fails);
void two_dynamic_sets_variable_radius(const int n_points, bool report_when_a_test_fails);
void mixed_float_double_point_sets(const int n_points, bool report_when_a_test_fails);
void resize_variable_radius(const int n_points, bool report_when_a_test_fails);

void benchmark_one_dynamic_set(const int n_points);
