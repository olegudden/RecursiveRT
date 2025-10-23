#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "color.h"
#include "triangle.h"
#include "light_source.h"
#include "mesh_loader.h"
#include "material.h"
#include "bvh.h"
#include "object.h"
#include "ray.h"

#include <vector>
#include <fstream>
#include <algorithm>
#include <random> // for anti aliasing

class camera
{
public:
    bool adjustable;
    double aspect_ratio = 1.0;
    int image_width = 100;
    // int primary_ray_counter = 0;

    inline double rand_double() // helper function: uniform random in [0, 1)
    {
        static thread_local std::mt19937 generator(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
        return distribution(generator);
    }

    void render(const std::vector<object> &world, std::vector<light_source> &lights, bvh &bvh_tree) // parallel anti aliasing version
    {
        initialize();
        std::vector<color> framebuffer(image_width * image_height);

        int samples_per_pixel = 2; // adjust for quality vs. speed

#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < image_height; j++)
        {
            for (int i = 0; i < image_width; i++)
            {
                color pixel_color(0, 0, 0);

                for (int s = 0; s < samples_per_pixel; s++)
                {
                    // random offset inside the pixel
                    double u = (i + rand_double()) * pixel_delta_h.x() + (j + rand_double()) * pixel_delta_v.x();
                    double v = (i + rand_double()) * pixel_delta_h.y() + (j + rand_double()) * pixel_delta_v.y();

                    auto pixel_center = pixel_top_left + (i + rand_double()) * pixel_delta_h + (j + rand_double()) * pixel_delta_v;

                    auto ray_direction = pixel_center - camera_center;
                    ray r(camera_center, ray_direction);

                    pixel_color += ray_color(r, world, lights, bvh_tree, 3);
                }

                // average samples
                pixel_color /= double(samples_per_pixel);

                framebuffer[j * image_width + i] = pixel_color;
            }
        }

        std::ofstream out("image.ppm");
        out << "P3\n"
            << image_width << ' ' << image_height << "\n255\n";
        for (const auto &c : framebuffer)
            write_color(out, c);
        out.close();
    }

    /* void render(const std::vector<object> &world, std::vector<light_source> &lights, bvh &bvh_tree) // non-parallel version
    {
        initialize();
        std::ofstream out("image.ppm", std::ios::out); // write to ppm file
        out << "P3\n"
            << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++)
        {
            for (int i = 0; i < image_width; i++)
            {
                // Calculate color from ray of every pixel
                auto pixel_center = pixel_top_left + (i * pixel_delta_h) + (j * pixel_delta_v);
                auto ray_direction = pixel_center - camera_center;
                ray r(camera_center, ray_direction);

                primary_ray_counter++;

                color pixel_color = ray_color(r, world, lights, bvh_tree, 3); // define recursion depth!

                if (!std::isfinite(pixel_color.x() || !std::isfinite(pixel_color.y() || !std::isfinite(pixel_color.z()))))
                {
                    std::cout << "error pixel\n";
                }
                // color pixel_color = ray_color_triangle(r, triangle(point3(-0.5, 0, -1), point3(0, 1, -1), point3(0.5, 0, -1)));
                write_color(out, pixel_color);
                // std_write_color(out, pixel_color);
            }
        }

        out.close(); // close file
    } */

private:
    int image_height;
    point3 camera_center;
    point3 pixel_top_left;
    vec3 pixel_delta_h;
    vec3 pixel_delta_v;

    point3 lookfrom;
    point3 lookat;
    vec3 vup;

    void initialize()
    {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        // === Set up viewport size ===
        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = aspect_ratio * viewport_height;

        if (adjustable) // Adjustable camera
        {
            // define your camera position and orientation [rb: (-1, 0, -1), (0, 0, -1.5), (0, 1, 0)]
            lookfrom = point3(0, 1, 2);  // Camera is ...
            lookat = point3(0, -2, -10); // Looking toward ...
            vup = vec3(0, 1, 0);         // "Up" is negative Z for top-down view (0, 0, -1)

            // compute orthonormal basis vectors
            vec3 w = unit_vector(lookfrom - lookat); // Camera direction (backward)
            vec3 u = unit_vector(cross(vup, w));     // Right direction
            vec3 v = cross(u, w);                    // Real "up" direction

            vec3 horizontal = viewport_width * u;
            vec3 vertical = viewport_height * v;
            vec3 lower_left_corner = lookfrom - horizontal / 2 - vertical / 2 - focal_length * w;

            // Save to camera
            camera_center = lookfrom;
            pixel_delta_h = horizontal / image_width;
            pixel_delta_v = vertical / image_height;
            pixel_top_left = lower_left_corner + pixel_delta_h / 2 + pixel_delta_v / 2;
        }
        else
        {
            camera_center = point3(0, 0, 0);
            // Calculate horizontal (right) and vertical (down) vector of viewport edges
            auto viewport_h = vec3(viewport_width, 0, 0);
            auto viewport_v = vec3(0, -viewport_height, 0);

            // Calculate horizontal and vertical delta vector from pixel to pixel
            pixel_delta_h = viewport_h / image_width;
            pixel_delta_v = viewport_v / image_height;

            // Calculate location of upper left pixel
            auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_h / 2 - viewport_v / 2;
            pixel_top_left = viewport_upper_left + pixel_delta_h / 2 + pixel_delta_v / 2;
        }
    }

    color ray_color_triangle(const ray &r, triangle t)
    {
        hit_info hi;
        if (t.hit(r, 0, 500, hi))
            return color(0, 0.5, 0) * (unit_vector(r.direction()).y() + 1.0f);
        else
        {
            float s = 0.5f * (unit_vector(r.direction()).y() + 1.0f);
            return (1.0f - s) * color(0.8, 0.9, 1.0) + s * color(0.4, 0.5, 0.9);
        }
    }

    color ray_color(const ray &r, const std::vector<object> &world, std::vector<light_source> &lights, bvh &bvh_tree, int recursion_limit)
    {
        float bounce_attenuation = 0.5f; // Reduce light contribution slightly per bounce

        if (recursion_limit <= 0)
            return color(0, 0, 0);

        hit_info final_hi;

        bool hit_anything = false;
        float closest_t = FLT_MAX;

        /* for (const object &obj : world) // BOUNDING BOX PER OBJECT
        {
            hit_info hibox;
            if (!obj.box.hit(r, 0.0f, closest_t, hibox))
                continue;

            for (triangle tri : obj.mesh)
            {
                hit_info temp_hi;
                if (tri.hit(r, 1e-5f, closest_t, temp_hi))
                {
                    closest_t = temp_hi.t;
                    final_hi = temp_hi;
                    hit_anything = true;
                }
            }
        } */

        if (!bvh_tree.hit(r, 1e-4f, FLT_MAX, final_hi)) // BVH check
        {
            // Sky background
            float s = 0.5f * (unit_vector(r.direction()).y() + 1.0f);
            return (1.0f - s) * color(0.8, 0.9, 1.0) + s * color(0.4, 0.5, 0.9);
        }

        /* for (const object &obj : world) // BRUTE-FORCE
        {
            for (triangle tri : obj.mesh)
            {
                hit_info temp_hi;
                if (tri.hit(r, 1e-5f, closest_t, temp_hi))
                {
                    hit_anything = true;
                    closest_t = temp_hi.t;
                    final_hi = temp_hi;
                }
            }
        } */

        /* if (!hit_anything) // BB & BF
        {
            float s = 0.5f * (unit_vector(r.direction()).y() + 1.0f);
            return (1.0f - s) * color(0.8, 0.9, 1.0) + s * color(0.4, 0.5, 0.9);
        } */

        color final_color(0, 0, 0);

        vec3 N = unit_vector(final_hi.normal);
        vec3 V = unit_vector(-r.direction());
        for (const light_source &light : lights)
        {
            vec3 L_un = light.pos - final_hi.p;
            // float distance_squared = dot(L_un, L_un);
            vec3 L = unit_vector(L_un);
            vec3 light_R = reflect(-L, N);
            vec3 view_dir = unit_vector(r.direction());

            float dist = std::max(L_un.length(), 1e-3f);
            float attenuation = std::clamp(5.0 / L_un.length(), 0.0, 1.0); // inverse distance falloff
            // float attenuation = 1.0f / (1.0f + L_un.length() * L_un.length()); // distance_squared (theory)

            // float spec_strength = 0.7f;
            // float ambient_strength = 0.3f;

            float diff = std::clamp(dot(N, L), 0.0f, 1.0f);
            // float spec = pow(std::clamp(dot(light_R, V), 0.0f, 1.0f), final_hi.mat.spec_exponent);
            float spec_angle = std::clamp(dot(light_R, V), 0.0f, 1.0f);
            float spec = (spec_angle > 0.0f) ? pow(spec_angle, final_hi.mat.spec_exponent) : 0.0f;
            if (!std::isfinite(spec))
                spec = 0.0f;

            color ambient = final_hi.mat.ambient; // + strength?
            color diffuse = diff * final_hi.mat.diffuse * light.col * light.intensity;
            color specular = spec * final_hi.mat.specular * light.col * light.intensity;

            color phong = ambient + attenuation * (diffuse + specular);

            color hit_color;
            // float eps = 1e-4f * std::max(1.0f, final_hi.p.length());
            float eps = 1e-4f;

            switch (final_hi.mt)
            {
            case material_type::diffuse:
            {
                bool in_shadow = false; // HARD SHADOWS
                ray shadow_ray(final_hi.p + eps * N, L);
                float light_dist = (light.pos - final_hi.p).length();

                /* for (object obj : world) // BOUNDING BOX PER OBJECT
                {
                    hit_info shadow_hit;
                    if (!obj.box.hit(shadow_ray, 0.001f, light_dist, shadow_hit))
                        continue;
                    for (triangle tri : obj.mesh)
                    {
                        if (tri.hit(shadow_ray, 1e-4f, light_dist, shadow_hit))
                        {
                            in_shadow = true;
                            break;
                        }
                    }
                    if (in_shadow)
                        break;
                } */

                hit_info shadow_hit; // HARD SHADOWS
                if (bvh_tree.hit(shadow_ray, eps, light_dist, shadow_hit))
                {
                    in_shadow = true;
                }

                hit_color = in_shadow ? ambient : phong;
                break;
            }

            case material_type::mirror:
            {
                vec3 surface_normal = dot(unit_vector(r.direction()), N) < 0.0f ? N : -N;
                vec3 reflect_dir = reflect(view_dir, surface_normal);
                ray reflected_ray(final_hi.p + eps * surface_normal, reflect_dir);

                color reflected_color = bounce_attenuation * ray_color(reflected_ray, world, lights, bvh_tree, recursion_limit - 1);

                float reflectivity = final_hi.mat.reflectivity;

                if (reflectivity >= 1.0f)
                {
                    hit_color = reflected_color;
                }
                else
                {
                    hit_color = reflectivity * reflected_color + (1.0f - reflectivity) * phong;
                }

                break;
            }

            case material_type::transparent:
            {
                /* vec3 surface_normal = dot(r.direction(), N) < 0.0f ? N : -N;
                bool front_face = dot(r.direction(), N) < 0.0f;
                float eta_ratio = front_face ? (1.0f / final_hi.mat.ior) : final_hi.mat.ior;

                // Fresnel
                float cos_theta = std::clamp(dot(-view_dir, surface_normal), 0.0f, 1.0f);
                float reflect_ratio = fresnel_schlick(cos_theta, eta_ratio); */

                vec3 surface_normal = dot(unit_vector(r.direction()), N) < 0.0f ? N : -N;
                bool entering = dot(unit_vector(r.direction()), N) < 0.0f;

                float n1 = entering ? 1.0f : final_hi.mat.ior;
                float n2 = entering ? final_hi.mat.ior : 1.0f;
                float eta_ratio = n1 / n2;

                float cos_theta = std::clamp(dot(-unit_vector(r.direction()), surface_normal), 0.0f, 1.0f);
                float reflect_ratio = fresnel_schlick(cos_theta, n1, n2);

                // Reflect
                ray reflected_ray(final_hi.p + eps * surface_normal, reflect(unit_vector(r.direction()), surface_normal));
                color reflected_color = bounce_attenuation * ray_color(reflected_ray, world, lights, bvh_tree, recursion_limit - 1);

                // Refract
                vec3 refract_dir;
                color refracted_color(0, 0, 0);
                if (refract(unit_vector(r.direction()), surface_normal, eta_ratio, refract_dir))
                {
                    ray refracted_ray(final_hi.p - eps * surface_normal, refract_dir);
                    refracted_color = bounce_attenuation * ray_color(refracted_ray, world, lights, bvh_tree, recursion_limit - 1);
                }
                else
                {
                    // total internal reflection -> treat as reflection only
                    hit_color = reflected_color;
                }

                // Mix
                // refracted_color = refracted_color * final_hi.mat.transmission_color; // adds colored tint
                color mixed = reflect_ratio * reflected_color + (1.0f - reflect_ratio) * refracted_color;
                // hit_color = 0.8f * hit_color + 0.2f * phong; // adds surface shading

                // Optional tint (soft specular)
                hit_color = mixed + specular;

                break;
            }

            default:
                hit_color = phong;
                break;
            }

            final_color += hit_color;
        }

        return final_color; // tone mapping/gamma happens in write_color()
    }

    // Firefly removal using neighbor color average
    void remove_fireflies(std::vector<color> &image, int width, int height, float threshold)
    {
        auto luminance = [](const color &c)
        {
            return 0.2126f * c.x() + 0.7152f * c.y() + 0.0722f * c.z();
        };

        std::vector<color> filtered = image;

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                int idx = y * width + x;
                float lum = luminance(image[idx]);

                // Neighborhood stats
                color sum(0, 0, 0);
                float lum_sum = 0.0f;
                int count = 0;

                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        if (dx == 0 && dy == 0)
                            continue;
                        int nidx = (y + dy) * width + (x + dx);

                        sum += image[nidx];
                        lum_sum += luminance(image[nidx]);
                        count++;
                    }
                }

                color avg_color = sum / float(count);
                float avg_lum = lum_sum / count;

                // If pixel is an outlier, replace with avg neighbor color
                if (lum > avg_lum * threshold)
                    filtered[idx] = avg_color;
            }
        }

        image.swap(filtered);
    }
};
#endif