#include <stdio.h>

#include "add_matrix.h"
#include "add_reduce.h"
#include "bank_conflict.h"
#include "gemm.h"

int main() {

	// test_add_matrix();
	// test_add_reduce_main();

	// valid_bank_conflict();

	test_gemm_main();
	return 0;
}