#include<math.h>

#define M_PI 3.14159267
#define M_1_PI (1.0/3.14159267)
#include <random>
std::uniform_real_distribution<double> distr(0.0,1.0);
std::default_random_engine generator;
double erand48(int X){
  return distr(generator);
}

double erand48(unsigned short *X) {
    return distr(generator);
}

double randx(double x, double y)
{
    double tmp = erand48(10);
    return x + (y-x)*tmp;
}