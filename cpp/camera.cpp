#define _USE_MATH_DEFINES

#include "../hpp/camera.hpp"
#include "../hpp/light_source.hpp"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <cfloat>
#include <cmath>

// ---------------- Random helper ----------------
double camera::rand_double()
{
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

// ---------------- Sampling techniques ----------------
vec3 camera::sample_cosine_hemisphere(const vec3 &N)
{
    float eps0 = rand_float();
    float eps1 = rand_float();

    float phi = 2.0f * M_PI * eps1;
    float r = sqrt(eps0);

    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(1.0f - eps0);

    // build local tangent basis
    vec3 tangent = unit_vector(perpendicular(N));
    vec3 bitangent = cross(N, tangent);

    // convert to world space
    return unit_vector(x * tangent + y * bitangent + z * N);
}

vec3 camera::sample_uniform_hemisphere(const vec3 &N)
{
    float eps0 = rand_float();
    float eps1 = rand_float();

    float phi = 2.0f * M_PI * eps1;
    float r = sqrt(1.0f - eps0 * eps0);

    float x = r * cos(phi);
    float y = r * sin(phi);

    vec3 tangent = unit_vector(perpendicular(N));
    vec3 bitangent = cross(N, tangent);

    return unit_vector(x * tangent + y * bitangent + eps0 * N);
}

// ---------------- Initialize camera ----------------
void camera::initialize()
{
    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;

    if (adjustable)
    {
        lookfrom = point3(0, 1, 2);
        lookat = point3(0, -2, -10);
        vup = vec3(0, 1, 0);

        vec3 w = unit_vector(lookfrom - lookat);
        vec3 u = unit_vector(cross(vup, w));
        vec3 v = cross(u, w);

        vec3 horizontal = viewport_width * u;
        vec3 vertical = viewport_height * v;
        vec3 lower_left_corner = lookfrom - horizontal / 2 - vertical / 2 - focal_length * w;

        camera_center = lookfrom;
        pixel_delta_h = horizontal / image_width;
        pixel_delta_v = vertical / image_height;
        pixel_top_left = lower_left_corner + pixel_delta_h / 2 + pixel_delta_v / 2;
    }
    else
    {
        camera_center = point3(0, 0, 0);
        auto viewport_h = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        pixel_delta_h = viewport_h / image_width;
        pixel_delta_v = viewport_v / image_height;

        auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_h / 2 - viewport_v / 2;
        pixel_top_left = viewport_upper_left + pixel_delta_h / 2 + pixel_delta_v / 2;
    }
}

// ---------------- Ray-color calculations ----------------
color camera::ray_color_triangle(const ray &r, triangle t)
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

color camera::ray_color(const ray &r, const std::vector<object> &world, std::vector<light_source> &lights, bvh &bvh_tree, int recursion_limit)
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

color camera::ray_color_area(const ray &r, const std::vector<object> &world, bvh &bvh_tree, const area_light &al, int recursion_limit)
{
    if (recursion_limit <= 0)
        return color(0, 0, 0);

    hit_info hi;
    if (!bvh_tree.hit(r, 1e-4f, FLT_MAX, hi))
    {
        // non-black background helps debug reflections
        float t = 0.5f * (unit_vector(r.direction()).y() + 1.0f);
        return (1.0f - t) * color(0.8, 0.9, 1.0) + t * color(0.4, 0.5, 0.9);
    }

    // If hit an emissive surface (the area light), return emission
    if (hi.mat.is_emissive())
        return hi.mat.emission;

    vec3 N = unit_vector(hi.normal);
    vec3 V = unit_vector(-r.direction()); // view vector (from hit toward camera)
    float eps = 1e-4f;

    // Start with small ambient (or zero while debugging)
    color result = 0.4 * hi.mat.ambient;
    // color result = color(0, 0, 0);

    // ==========================
    // Direct area-light sampling (soft shadows)
    // ==========================
    const int num_samples = 20;
    color light_sum(0, 0, 0);

    float area = al.e1.length() * al.e2.length();
    vec3 light_normal = unit_vector(al.normal);

    for (int s = 0; s < num_samples; ++s)
    {
        point3 light_pos = sample_area_light(al);
        vec3 to_light = light_pos - hi.p;
        float dist = to_light.length();
        if (dist <= 1e-6f)
            continue;
        vec3 L = unit_vector(to_light);

        // don't mutate N — use local copy for orientation
        vec3 Nsurf = N;
        if (dot(Nsurf, L) < 0.0f)
            Nsurf = -Nsurf;

        // geometry checks
        float NdotL = std::max(dot(Nsurf, L), 0.0f);
        float LdotNL = std::max(dot(-L, light_normal), 0.0f);
        if (NdotL <= 0.0f || LdotNL <= 0.0f)
            continue;

        // shadow ray: test only up to the sampled light point (dist - eps)
        ray shadow_ray(hi.p + eps * Nsurf, L);
        hit_info sh;
        bool blocked = bvh_tree.hit(shadow_ray, eps, dist - eps, sh) && !sh.mat.is_emissive();
        if (blocked)
            continue;

        // geometry factor (uniform area sampling -> factor = (N·L * LdotNL) / dist^2)
        float geom = (NdotL * LdotNL) / (dist * dist);

        // Lambert (or Phong) — here we use hi.mat.diffuse as albedo
        light_sum += al.emission * hi.mat.diffuse * geom;

        // Optional simple Phong specular (keeps same look you had)
        vec3 R = unit_vector(reflect(-L, Nsurf)); // reflect incoming light about Nsurf
        float spec_angle = std::max(dot(R, V), 0.0f);
        if (spec_angle > 0.0f)
            light_sum += al.emission * hi.mat.specular * pow(spec_angle, hi.mat.spec_exponent) * (LdotNL / (dist * dist));
    }

    result += light_sum / float(num_samples);

    // ==========================
    // Reflection
    // ==========================
    if (hi.mat.reflectivity > 0.0f)
    {
        // Orient surface normal for reflection (do NOT reuse mutated N)
        vec3 surface_normal = (dot(r.direction(), N) < 0.0f) ? N : -N;

        // incident is the ray direction (pointing toward hit); compute reflected direction:
        vec3 incident = unit_vector(r.direction()); // points *into* the surface
        vec3 reflect_dir = incident - 2.0f * dot(incident, surface_normal) * surface_normal;
        reflect_dir = unit_vector(reflect_dir);

        ray reflected_ray(hi.p + eps * surface_normal, reflect_dir);
        color reflected_color = ray_color_area(reflected_ray, world, bvh_tree, al, recursion_limit - 1);

        // mix reflection with local result
        result = hi.mat.reflectivity * reflected_color + (1.0f - hi.mat.reflectivity) * result;
    }

    // ==========================
    // Refraction (transparent)
    // ==========================
    if (hi.mat.transparency > 0.0f)
    {
        // Determine correct normal orientation and eta
        bool entering = dot(r.direction(), N) < 0.0f;
        vec3 surface_normal = entering ? N : -N;
        float n1 = entering ? 1.0f : hi.mat.ior;
        float n2 = entering ? hi.mat.ior : 1.0f;
        float eta = n1 / n2;

        vec3 incident = unit_vector(r.direction());
        vec3 refract_dir;
        // Use your existing refract helper which should return bool and fill refract_dir
        if (refract(incident, surface_normal, eta, refract_dir))
        {
            refract_dir = unit_vector(refract_dir);
            ray refracted_ray(hi.p - eps * surface_normal, refract_dir);
            color refracted_color = ray_color_area(refracted_ray, world, bvh_tree, al, recursion_limit - 1);

            result = (1.0f - hi.mat.transparency) * result + hi.mat.transparency * refracted_color;
        }
        else
        {
            // total internal reflection: treat as reflection
            vec3 reflect_dir = incident - 2.0f * dot(incident, surface_normal) * surface_normal;
            reflect_dir = unit_vector(reflect_dir);
            ray reflected_ray(hi.p + eps * surface_normal, reflect_dir);
            color reflected_color = ray_color_area(reflected_ray, world, bvh_tree, al, recursion_limit - 1);
            result = reflected_color;
        }
    }

    return result;
}

color camera::rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights) // TODO sample area light
{
    /* if (depth <= 0)
        return color(0, 0, 0);

    hit_info hi;
    if (!bvh_tree.hit(r, 1e-4f, FLT_MAX, hi))
    {
        // Background
        float t = 0.5f * (unit_vector(r.direction()).y() + 1.0f);
        return (1.0f - t) * color(0.8, 0.9, 1.0) + t * color(0.4, 0.5, 0.9);
    }

    // If emissive (light), return its emission
    if (hi.mat.is_emissive())
        return hi.mat.emission;

    // Otherwise, sample new direction(s)
    vec3 N = unit_vector(hi.normal);
    color accum(0, 0, 0);
    const int num_samples = 3;

    for (int i = 0; i < num_samples; i++)
    {
        // --- Sample a direction in a cosine-weighted hemisphere ---
        vec3 new_dir = sample_cosine_hemisphere(N);

        // --- Compute PDF (for cosine-weighted hemisphere = cos(theta)/pi) ---
        float cos_theta = std::max(dot(new_dir, N), 0.0f);
        float pdf = cos_theta / M_PI;
        if (pdf <= 0)
            continue;

        // --- Recursive ray ---
        ray new_ray(hi.p + 1e-4f * N, new_dir);
        color incoming = rec_trace(new_ray, bvh_tree, depth - 1);

        // --- BRDF weight (Lambertian = albedo/pi) ---
        color brdf = hi.mat.diffuse / M_PI;

        // --- Monte Carlo estimator ---
        accum += (incoming * brdf * cos_theta) / pdf;
    }
    // Average over samples
    return accum / float(num_samples); */

    // Russian roulette
    float p_survive = 1.0f; // p_survive = 0.8f
    // float p_survive = std::clamp(max_component(hi.mat.diffuse), 0.1f, 0.9f); TODO
    if (rand_float() > p_survive)
        return color(0, 0, 0); // Terminate path
    if (depth <= 0)
        return color(0, 0, 0);

    hit_info hi;
    if (!bvh_tree.hit(r, 1e-4f, FLT_MAX, hi))
    {
        // Background (bright)
        /*return color(1,1,1);
        float t = 0.5f * (unit_vector(r.direction()).y() + 1.0f);
        return (1.0f - t) * color(0.8, 0.9, 1.0) + t * color(0.4, 0.5, 0.9); */
        return color(0, 0, 0);
    }

    if (hi.mat.is_emissive())
        // TODO check hit normal vs light normal before emitting light
        return hi.mat.emission;

    vec3 N = unit_vector(hi.normal);
    vec3 V = unit_vector(-r.direction());
    color result(0, 0, 0);

    // Small offset to avoid acne
    const float eps = 1e-4f;

    // =========================================================
    // 1. Perfect MIRROR material
    // =========================================================
    if (hi.mt == material_type::mirror)
    {
        vec3 reflect_dir = reflect(unit_vector(r.direction()), N);
        ray reflected_ray(hi.p + eps * N, reflect_dir);
        return (1.0f / p_survive) * hi.mat.specular * rec_trace(reflected_ray, bvh_tree, depth - 1, lights);
    }

    // =========================================================
    // 2. TRANSPARENT / DIELECTRIC material
    // =========================================================
    if (hi.mt == material_type::transparent)
    {
        float ior = hi.mat.ior;
        vec3 dir = unit_vector(r.direction());
        bool front_face = dot(dir, N) < 0.0f;
        vec3 normal = front_face ? N : -N;
        float n1 = front_face ? 1.0f : ior;
        float n2 = front_face ? ior : 1.0f;
        float eta = n1 / n2;

        vec3 refr_dir;
        bool can_refract = refract(dir, normal, eta, refr_dir);

        float cos_theta = std::clamp(dot(-dir, normal), 0.0f, 1.0f);
        float reflect_prob = fresnel_schlick(cos_theta, n1, n2); // use your fresnel function

        if (!can_refract)
            reflect_prob = 1.0f; // total internal reflection

        ray refracted_ray(hi.p - eps * normal, refr_dir);
        return (1.0f / p_survive) * hi.mat.specular * rec_trace(refracted_ray, bvh_tree, depth - 1, lights);

        // Russian roulette: pick reflection or refraction TODO
        /* if (rand_float() < reflect_prob)
        {
            vec3 reflect_dir = reflect(dir, normal);
            ray reflected_ray(hi.p + eps * normal, reflect_dir);
            return rec_trace(reflected_ray, bvh_tree, depth - 1);
        }
        else
        {
            ray refracted_ray(hi.p - eps * normal, refr_dir);
            return rec_trace(refracted_ray, bvh_tree, depth - 1);
        } */
    }

    // =========================================================
    // 3. Diffuse (Lambertian)
    // =========================================================
    const int num_samples = 10;
    color accum(0, 0, 0);

    for (int i = 0; i < num_samples; i++)
    {
        vec3 new_dir = sample_cosine_hemisphere(N);
        // vec3 new_dir = sample_uniform_hemisphere(N); (1)
        float cos_theta = std::max(dot(new_dir, N), 0.0f);
        float pdf = cos_theta / M_PI;
        // float pdf = 1 / M_PI; (2)

        if (pdf <= 0)
            continue;

        ray new_ray(hi.p + eps * N, new_dir);

        color incoming = (1.0f / p_survive) * rec_trace(new_ray, bvh_tree, depth - 1, lights);
        color brdf = hi.mat.diffuse / M_PI;

        // accum += (incoming * brdf * cos_theta) / pdf;
        accum += incoming * brdf;
    }

    return (M_PI / float(num_samples)) * accum;
    // return accum / float(num_samples);
}

// ---------------- Render ----------------
void camera::render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree)
{
    initialize();
    std::vector<color> framebuffer(image_width * image_height);
    int samples_per_pixel = 2;

#pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < image_height; j++)
    {
        for (int i = 0; i < image_width; i++)
        {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; s++)
            {
                auto pixel_center = pixel_top_left + (i + rand_double()) * pixel_delta_h + (j + rand_double()) * pixel_delta_v;
                ray r(camera_center, pixel_center - camera_center);
                pixel_color += rec_trace(r, bvh_tree, 2, lights); // TODO
                // pixel_color += ray_color_area(r, world, bvh_tree, al, 3);
                // pixel_color += ray_color(r, world, lights, bvh_tree, 3);
            }
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

void camera::old_render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree)
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

            // color pixel_color = ray_color(r, world, lights, bvh_tree, 3); // define recursion depth!
            color pixel_color = rec_trace(r, bvh_tree, 2, lights);

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
}

// ---------------- Firefly removal ----------------
void camera::remove_fireflies(std::vector<color> &image, int width, int height, float threshold)
{
    auto luminance = [](const color &c)
    { return 0.2126f * c.x() + 0.7152f * c.y() + 0.0722f * c.z(); };
    std::vector<color> filtered = image;

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int idx = y * width + x;
            float lum = luminance(image[idx]);
            color sum(0, 0, 0);
            float lum_sum = 0.0f;
            int count = 0;

            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0)
                        continue;
                    int nidx = (y + dy) * width + (x + dx);
                    sum += image[nidx];
                    lum_sum += luminance(image[nidx]);
                    count++;
                }

            color avg_color = sum / float(count);
            float avg_lum = lum_sum / count;
            if (lum > avg_lum * threshold)
                filtered[idx] = avg_color;
        }
    }

    image.swap(filtered);
}
