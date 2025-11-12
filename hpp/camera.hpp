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

class camera
{
public:
    bool adjustable = false;
    double aspect_ratio = 1.0;
    int image_width = 100;

    camera() = default;

    // Render the scene
    void render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree);
    void old_render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree);

    // Helper function: uniform random in [0, 1) (for antialiasing)
    double rand_double();

    vec3 sample_cosine_hemisphere(const vec3 &N);
    vec3 sample_uniform_hemisphere(const vec3 &N);

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
    color ray_color(const ray &r, const std::vector<object> &world, std::vector<light_source> &lights,
                    bvh &bvh_tree, int recursion_limit);

    color ray_color_area(const ray &r, const std::vector<object> &world, bvh &bvh_tree, const area_light &al, int recursion_limit);

    color rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights);
    void remove_fireflies(std::vector<color> &image, int width, int height, float threshold);
};

#endif // CAMERA_H
