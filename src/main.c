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

int main() {
  float u[3] = {1, 2, 3};
  float v[3] = {4, 5, 6};
  int n = 3;
  printf("Dist: %f\n", dist(u, v, n));
  return 0;
}
