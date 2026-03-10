#ifndef LIGHT_SOURCE_HPP
#define LIGHT_SOURCE_HPP

#include "vec3.hpp"
#include "color.hpp"
#include "ray.hpp"
#include "hittable.hpp"
#include "bvh.hpp"
#include <random>

struct area_light
{
    point3 origin; // bottom left of rectangle
    vec3 e1, e2;   // edges
    vec3 normal;
    float area;
    color emission;
    float selection_weight;
    int light_id;
    std::vector<triangle> tris;
};

float rand_float();
float random_float(float min, float max);
point3 sample_area_light(const area_light &light);
std::vector<point3> uniform_sample_light(const area_light &L, int N);
std::vector<point3> stratified_sample_light(const area_light &light, int N);
bool visible(const point3 &p, const point3 &light_point, const bvh &bvh_tree);
void compute_lightweights(std::vector<area_light> &lights);
area_light sample_multiple_AL(const std::vector<area_light> &lights);

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
