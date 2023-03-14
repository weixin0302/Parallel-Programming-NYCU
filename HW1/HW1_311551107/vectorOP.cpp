#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x;
  __pp_vec_int y, count;
  __pp_vec_float result;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float nine = _pp_vset_float(9.999999f);
  int cnbits;
  __pp_mask maskAll, maskIsequal0, maskgreater0, maskgreater999, maskrest;
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones(N - i);
    _pp_vload_float(x, values + i, maskAll);
    _pp_vload_int(y, exponents + i, maskAll);
    _pp_veq_int(maskIsequal0, y, zero, maskAll);
    _pp_vset_float(result, 1.f, maskIsequal0);

    maskgreater0 = _pp_init_ones(0);
    maskgreater0 = _pp_mask_not(maskIsequal0);
    _pp_vload_float(result, values + i, maskgreater0);
    _pp_vsub_int(count, y, one, maskgreater0);

    maskrest = _pp_init_ones(0);
    _pp_vgt_int(maskrest, count, zero, maskgreater0);
    cnbits = _pp_cntbits(maskrest);
    while (cnbits > 0)
    {
      _pp_vmult_float(result, result, x, maskrest);
      _pp_vsub_int(count, count, one, maskrest);
      _pp_vgt_int(maskrest, count, zero, maskrest);
      cnbits = _pp_cntbits(maskrest);
    }

    maskgreater999 = _pp_init_ones(0);
    _pp_vgt_float(maskgreater999, result, nine, maskgreater0);
    _pp_vset_float(result, 9.999999f, maskgreater999);
    _pp_vstore_float(output + i, result, maskAll);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __pp_vec_float x;
  __pp_vec_float result = _pp_vset_float(0.f);
  __pp_mask maskAll;
  float ans = 0.f;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();
    _pp_vload_float(x, values + i, maskAll);
    _pp_vadd_float(result, result, x, maskAll);
  }
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    ans += result.value[i];
  }

  return ans;
}