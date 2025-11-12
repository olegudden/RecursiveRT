#include "../hpp/ray.hpp"
#include <cmath>

// int ray::ray_counter = 0;

ray::ray() = default;

ray::ray(const point3 &origin, const vec3 &direction)
    : orig(origin), dir(direction)
{
    // ray_counter++;
}

const point3 &ray::origin() const { return orig; }
const vec3 &ray::direction() const { return dir; }

point3 ray::at(double t) const { return orig + t * dir; }

vec3 reflect(const vec3 &I, const vec3 &N)
{
    return I - 2 * dot(I, N) * N;
}

bool refract(const vec3 &I_in, const vec3 &N_in, float eta_ratio, vec3 &refracted)
{
    vec3 I = unit_vector(I_in);
    vec3 N = unit_vector(N_in);

    float cos_theta = dot(-I, N);
    vec3 r_out_perp = eta_ratio * (I + cos_theta * N);
    float k = 1.0f - dot(r_out_perp, r_out_perp);
    if (k < 0.0f)
        return false;
    vec3 r_out_parallel = -std::sqrt(k) * N;
    refracted = r_out_perp + r_out_parallel;
    return true;
}
