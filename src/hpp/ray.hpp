#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"

class ray
{
public:
    // static int ray_counter;

    ray();
    ray(const point3 &origin, const vec3 &direction);

    const point3 &origin() const;
    const vec3 &direction() const;

    point3 at(double t) const;

private:
    point3 orig;
    vec3 dir;
};

vec3 reflect(const vec3 &I, const vec3 &N);
bool refract(const vec3 &I_in, const vec3 &N_in, float eta_ratio, vec3 &refracted);

#endif // RAY_HPP
