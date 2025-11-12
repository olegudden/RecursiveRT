#ifndef LIGHT_SOURCE_HPP
#define LIGHT_SOURCE_HPP

#include "vec3.hpp"
#include "color.hpp"
#include <random>

struct area_light
{
    point3 origin; // bottom left of rectangle
    vec3 e1, e2;   // edges
    vec3 normal;
    float area;
    color emission;
};

float rand_float();
point3 sample_area_light(const area_light &light);

class light_source
{
public:
    point3 pos;
    color col;
    float intensity;

    light_source();
    light_source(point3 pos, color col, float intensity);
};

#endif // LIGHT_SOURCE_HPP
