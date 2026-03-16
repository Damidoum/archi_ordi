#include <immintrin.h>
#include <math.h>
#include <stdio.h>

// exercice 1
double dist(float *u, float *v, int n) {
  double sum = 0.;
  for (int i = 0; i < n; i++) {
    sum += sqrt((u[i] * u[i] + (v[i] * v[i])) /
                (1 + (u[i] * v[i]) * (u[i] * v[i])));
  }
  return sum;
}

// exercice 2
double dist_avx(float *u, float *v, int n) {
  double sum = 0.;
  __m128d vec_sum = _mm_setr_ps(0.0, 0.0, 0.0, 0.0);
  __m128 one = _mm_setr_ps(1.0, 1.0, 1.0, 1.0);

  for (int i = 0; i < n - 4; i += 4) {
    __m128 vec_u = _mm_load_ps(&u[i]);
    __m128 vec_v = _mm_load_ps(&v[i]);
    __m128 vec_sqrt = _mm_sqrt_ps(
        _mm_div_ps(_mm_mul_ps(vec_u, vec_u) + _mm_mul_ps(vec_v, vec_v),
                   _mm_add_ps(one, _mm_mul_ps(vec_u, vec_v))));
    vec_sum = _mm_add_ps(vec_sum, vec_sqrt);
  }
  vec_sum = _mm_hadd_ps(vec_sum, vec_sum);
  vec_sum = _mm_hadd_ps(vec_sum, vec_sum);
  return *((float *)vec_sum);
}

int main() {
  float u[3] = {1, 2, 3};
  float v[3] = {4, 5, 6};
  int n = 3;
  printf("Dist: %f\n", dist(u, v, n));
  printf("Dist avx: %f\n", dist_avx(u, v, n));
  return 0;
}
