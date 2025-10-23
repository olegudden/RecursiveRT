#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "material.h"

struct hit_info
{
    point3 p;
    float t;
    vec3 normal;
    material mat;
    material_type mt;
};

class hittable
{
public:
    virtual bool hit(const ray &r, float t_min, float t_max, hit_info &hi) const = 0;
};

#endif