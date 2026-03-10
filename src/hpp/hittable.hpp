#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include "ray.hpp"
#include "material.hpp"

struct hit_info
{
    point3 p;
    float t;
    vec3 normal;
    vec3 shading_normal;
    vec3 geometric_normal; // face normal (for refraction/enter-exit/offsets)
    material mat;
    material_type mt;
    bool front_face;
};

class hittable
{
public:
    virtual bool hit(const ray &r, float t_min, float t_max, hit_info &hi) const = 0;
};

#endif