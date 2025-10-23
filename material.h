#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include <cmath>

enum class material_type
{
    diffuse,
    mirror,
    transparent
};

class material
{
public:
    // Phong
    color ambient;
    color diffuse;
    color specular;
    float spec_exponent;

    // for ray intersections
    float reflectivity;       // 0 = matte, 1 = perfect mirror
    float transparency;       // = 1.0f - reflectivity; // 0 = opaque, 1 = fully transparent
    color transmission_color; // colored tint of transparent object
    float ior = 1.5f;         // Index of refraction

private:
};

/* inline float fresnel_schlick(float cos_theta, float ior)
{
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0; // base reflectance at normal incidence
    return r0 + (1.0f - r0) * std::pow(1.0f - cos_theta, 5.0f);
} */

inline float fresnel_schlick(float cos_theta, float n1, float n2)
{
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cos_theta, 5.0f);
}

#endif