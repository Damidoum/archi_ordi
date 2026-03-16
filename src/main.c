#include <immintrin.h>
#include <math.h>
#include <stdio.h>

// exercice 1
double dist(float *u, float *v, int n) {
  double u_d;
  double v_d;
  double sum = 0.;
  for (int i = 0; i < n; i++) {
    u_d = (double)u[i];
    v_d = (double)v[i];
    sum += sqrt(((u_d * u_d) + (v_d * v_d)) / (1 + (u_d * v_d) * (u_d * v_d)));
  }
  return sum;
}

/* float dist_avx_float(float *u, float *v, int n) {
  __m128 vec_sum = _mm_setzero_ps();
  __m128 one = _mm_set1_ps(1.0);

  for (int i = 0; i < n - 3; i += 4) {
    __m128 vec_u = _mm_load_ps(&u[i]);
    __m128 vec_v = _mm_load_ps(&v[i]);

    __m128 vec_sqrt = _mm_sqrt_ps(_mm_div_ps(
        _mm_add_ps(_mm_mul_ps(vec_u, vec_u), _mm_mul_ps(vec_v, vec_v)),
        _mm_add_ps(one, _mm_mul_ps(_mm_mul_ps(vec_u, vec_v),
                                   _mm_mul_ps(vec_u, vec_v)))));

    vec_sum = _mm_add_ps(vec_sum, vec_sqrt);
  }
  vec_sum = _mm_hadd_ps(vec_sum, vec_sum);
  vec_sum = _mm_hadd_ps(vec_sum, vec_sum);
  return ((float *)&vec_sum)[0];
}
*/

// exercice 2
double dist_avx(float *u, float *v, int n) {
  __m256d vec_sum = _mm256_setzero_pd();
  __m256d one = _mm256_set1_pd(1.0);

  for (int i = 0; i < n - 3; i += 4) {
    printf("%d\n", i);
    __m256d vec_u = _mm256_cvtps_pd(_mm_load_ps(&u[i]));
    __m256d vec_v = _mm256_cvtps_pd(_mm_load_ps(&v[i]));

    __m256d vec_sqrt = _mm256_sqrt_pd(_mm256_div_pd(
        _mm256_add_pd(_mm256_mul_pd(vec_u, vec_u), _mm256_mul_pd(vec_v, vec_v)),
        _mm256_add_pd(one, _mm256_mul_pd(_mm256_mul_pd(vec_u, vec_v),
                                         _mm256_mul_pd(vec_u, vec_v)))));

    vec_sum = _mm256_add_pd(vec_sum, vec_sqrt);
  }
  vec_sum = _mm256_hadd_pd(vec_sum, vec_sum);
  vec_sum = _mm256_add_pd(vec_sum, _mm256_permute4x64_pd(vec_sum, 0b00011011));
  return ((double *)&vec_sum)[0];
}

int main() {
#define N 8
  float u[N] __attribute__((aligned(64))) = {4, 5, 6, 10, 11, 2, 2, 3};
  float v[N] __attribute__((aligned(64))) = {1, 2, 3, 4, 5, 6, 8};

  printf("Dist: %f\n", dist(u, v, N));
  printf("Dist avx: %f\n", dist_avx_float(u, v, N));
  return 0;
}
