#ifndef LIGHT_SOURCE_HPP
#define LIGHT_SOURCE_HPP

#include "bvh.hpp"

#include <random>
#include <vector>

struct light_sample
{
    point3 p;
    vec3 normal;
    float pdf_area;

    // texture
    float u, v;
    material mat;
};

struct area_light
{
    point3 origin; // center (old: bottom left of rectangle)
    vec3 e1, e2;   // edges (rectangle)
    vec3 normal;
    float area;
    color emission;
    float selection_weight; // bad for parallelism
    int light_id;
    std::vector<triangle> tris;  // triangles
    std::vector<float> tris_cdf; // for triangle selection
    std::string name;
};

float rand_float();
float random_float(float min, float max);
point3 sample(vec3 &tri_normal, const area_light &light, float &out_pdf);
point3 sample_stratified(vec3 &tri_normal, const area_light &light, float &out_pdf, float r1, float r2, vec3 &out_uv, material &out_mat);
point3 sample_area_light(const area_light &light);
std::vector<point3> uniform_sample_light(const area_light &L, int N);
std::vector<point3> stratified_sample_light(const area_light &light, int N);
bool visible(const point3 &p, const point3 &light_point, const bvh &bvh_tree);
void compute_global_light_weights(std::vector<area_light> &lights);
std::vector<float> compute_local_light_weights(const point3 &hit_p, const vec3 &surface_normal, const std::vector<area_light> &lights);
const area_light &sample_multiple_AL(const std::vector<area_light> &lights, const std::vector<float> &local_probs, float &chosen_prob, int &chosen_id);

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
