#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "hittable.hpp"
#include "color.hpp"
#include "triangle.hpp"
#include "light_source.hpp"
#include "mesh_loader.hpp"
#include "material.hpp"
#include "bvh.hpp"
// #include "object.h"
#include "ray.hpp"

#include <vector>
#include <fstream>
#include <random>

struct TraceResult
{
    color full;
    color brdf_only;
};

class camera
{

public:
    bool adjustable = false;
    double aspect_ratio = 1.0;
    int image_width = 100;

    camera() = default;

    // Render the scene
    void render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree);
    void old_render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree); // area_light

    // Helper function: uniform random in [0, 1) (for antialiasing)
    double rand_double();

    vec3 random_in_unit_sphere();

    float power_heuristic(float pdf_a, float pdf_b);
    float balance_heuristic(float pdf_a, float pdf_b);

    vec3 sample_cosine_hemisphere(const vec3 &N);
    vec3 sample_uniform_hemisphere(const vec3 &N);
    vec3 phong_specular(vec3 n, vec3 w0, float s);

private:
    int image_height;
    point3 camera_center;
    point3 pixel_top_left;
    vec3 pixel_delta_h;
    vec3 pixel_delta_v;

    point3 lookfrom;
    point3 lookat;
    vec3 vup;

    void initialize();

    color ray_color_triangle(const ray &r, triangle t);
    color ray_color_phong(const ray &r, const std::vector<object> &world, std::vector<light_source> &lights, bvh &bvh_tree, int recursion_limit);
    color ray_color_area_light(const ray &r, const std::vector<object> &world, bvh &bvh_tree, const area_light &al, int recursion_limit);

    color baseline_rec_trace(const ray &r, const bvh &bvh_tree, int depth);

    color nee_rec_trace(const ray &r, const bvh &bvh_tree, int depth, const std::vector<area_light> &lights);

    color newest_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights);
    color newest_rec_trace_RR(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, color throughput);

    color divided_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights);
    color next_event_estimation(const vec3 &wo, const vec3 &N, const hit_info &hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights);
    color indirect_illumination(const vec3 &wo, const vec3 &N, const hit_info &hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights);
    color eval_bsdf(const material &m, const vec3 &p, const vec3 &n, const vec3 &wi, const vec3 &wo, float &pdf_factor, bool &is_delta);
    vec3 sample_from_bsdf(vec3 N, vec3 w0, material m);
    color recursion_mirror(vec3 I, vec3 N, hit_info hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights);
    color recursion_transparent(vec3 I, vec3 N, hit_info hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights);

    void remove_fireflies(std::vector<color> &image, int width, int height, float threshold);

    void build_orthonormal_basis(vec3 v, vec3 &t, vec3 &b);
    vec3 dielectric_F0_from_tint(const vec3 &spec_tint);
    vec3 sample_phong_lobe(const vec3 &r, const vec3 &n, float s, float &pdf_phong);
    color brdf_diffuse(const material &m);
    color brdf_specular(const material &m, const float s, const vec3 &refl_dir, const vec3 &wo);
};

#endif // CAMERA_H
