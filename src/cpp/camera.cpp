#define _USE_MATH_DEFINES

#include "../hpp/camera.hpp"
#include "../hpp/error.hpp"
#include "../hpp/texture.hpp"
#include <iostream>
#include <omp.h>

const int MAX_DEPTH = 6;
const int MIN_DEPTH = 1;

point3 test_point(0, 0, 0);

// static int lightpath_counter = 0;

void camera::build_orthonormal_basis(vec3 v, vec3 &t, vec3 &b)
{
    vec3 ze(0.0f, 0.0f, 1.0f);

    // n is assumed normalized
    if (fabs(dot(v, ze)) < 0.999f)
    {
        t = unit_vector(cross(v, vec3(0.0f, 0.0f, 1.0f)));
    }
    else
    {
        t = unit_vector(cross(v, vec3(0.0f, 1.0f, 0.0f)));
    }
    b = cross(v, t);
}

// small dielectric F0 helper — keeps dielectrics near 4% reflectance, optionally mild tint
vec3 camera::dielectric_F0_from_tint(const vec3 &spec_tint)
{
    return vec3(0.04f, 0.04f, 0.04f) * (vec3(0.9f, 0.9f, 0.9f) + 0.1f * spec_tint);
}

// Heuristics TODO: cutoff
float camera::balance_heuristic(float pdf_a, float pdf_b)
{
    return pdf_a / (pdf_a + pdf_b);
}

float camera::power_heuristic(float pdf_a, float pdf_b) // try different betas
{
    float a = pdf_a * pdf_a * pdf_a * pdf_a * pdf_a;
    float b = pdf_b * pdf_b * pdf_b * pdf_b * pdf_b;
    return (a / (a + b));
}

// BRDFs
color camera::brdf_diffuse(const material &m)
{
    return m.diffuse / M_PI;
}

color camera::tex_diffuse(hit_info &hi)
{
    color diffuse_color = hi.mat.diffuse;

    if (hi.mat.has_diffuse_tex)
    {
        // Look up the actual color from the loaded image
        diffuse_color = sample_texture(data, width, height, hi.u, hi.v);

        checker_texture my_checker(color(1, 1, 1), color(0, 0, 0), 10.0f);
        diffuse_color = my_checker.sample(hi.u, hi.v);

        /* color dark_brown(0.3f, 0.15f, 0.05f); // wood
        color light_brown(0.6f, 0.4f, 0.2f);
        color dark_grey(0.4f, 0.4f, 0.35f); // marble
        color white(1.0f, 1.0f, 1.0f);
        wood_texture procedural_wood(dark_grey, white, 40.0f);
        diffuse_color = procedural_wood.sample(hi.u, hi.v); */
    }

    // Now use 'diffuse_color' for your BRDF calculations instead of hi.mat.diffuse
    color fr = (diffuse_color / M_PI);
    return fr;
}

// 2. Update your diffuse evaluation inside your path tracer
color camera::tex_diffuse_new(const hit_info &hi, TextureManager &tex_manager)
{
    color diffuse_color = hi.mat.diffuse;

    if (hi.mat.has_diffuse_tex)
    {
        // Dynamically fetch the pixel data belonging to this specific polygon's material index
        diffuse_color = tex_manager.sample_texture_from_cache(tex_manager, hi.mat.diffuse_tex_id, hi.u, hi.v);
    }

    // Multiply the fetched color by 1/PI for diffuse Lambertian properties
    return diffuse_color / M_PI;
}

color camera::brdf_specular(const material &m, const float s, const vec3 &refl_dir, const vec3 &wo)
{
    float cosAlpha = std::max(0.0f, dot(refl_dir, wo)); //-wo?
    // return m.specular * ((s + 2.0f) / (2.0f * M_PI)) * std::pow(cosAlpha, s);
    return m.specular * ((s + 2.0f) / (2.0f * M_PI)) * std::pow(cosAlpha, s);
}

// Random helpers
double camera::rand_double()
{
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

vec3 camera::random_in_unit_sphere()
{
    while (true)
    {
        vec3 p = vec3(random_float(-1, 1),
                      random_float(-1, 1),
                      random_float(-1, 1));
        if (p.length_squared() < 1.0f)
            return p;
    }
}

// Sampling
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

vec3 camera::phong_specular(vec3 n, vec3 w0, float s) // not used
{
    vec3 r = reflect(w0, n);
    float eta1 = rand_float();
    float eta2 = rand_float();

    float phi = 2.0f * float(M_PI) * eta1;
    float cosTheta = std::pow(eta2, 1.0f / (s + 1.0f)); // importance sampling
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));

    float x = sinTheta * std::cos(phi);
    float y = sinTheta * std::sin(phi);
    float z = cosTheta;

    vec3 t, b;
    build_orthonormal_basis(n, t, b);

    vec3 wi = x * t + y * b + z * r;
    return unit_vector(wi);
}

vec3 camera::sample_phong_lobe(const vec3 &r, const vec3 &n, float s, float &pdf_phong)
{
    if (dot(r, n) <= 0.0f)
    {
        pdf_phong = 0.0f;
        return vec3(0, 0, 0);
    }

    float u1 = rand_float();
    float u2 = rand_float();

    float phi = 2.0f * M_PI * u1;
    float cosTheta = std::pow(u2, 1.0f / (s + 1.0f));
    float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);

    vec3 T, B;
    make_orthonormal_basis(r, T, B);

    vec3 wo =
        unit_vector(T * (sinTheta * cos(phi)) +
                    B * (sinTheta * sin(phi)) +
                    r * cosTheta);

    float cos_r = std::max(0.0f, dot(r, wo));
    pdf_phong = (s + 2.0f) * std::pow(std::max(0.0f, dot(r, wo)), s) / (2.0f * M_PI); //(s + 2.0f) * std::pow(std::max(dot(refl_dir, wo), 0.0f), s) / (2.0f * M_PI);

    return wo;
}

// Initialization
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
        lookat = point3(0, -2, -10); // 0, -2, -10
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

// Ray-color methods
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

color camera::ray_color_phong(const ray &r, const std::vector<object> &world, std::vector<light_source> &lights, bvh &bvh_tree, int recursion_limit)
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
    }*/
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
                        in_shadow = true; break;
                        }
                    }
                    if (in_shadow)
                    break;
                }
            */
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
            color reflected_color = bounce_attenuation * ray_color_phong(reflected_ray, world, lights, bvh_tree, recursion_limit - 1);
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
            /*
            vec3 surface_normal = dot(r.direction(), N) < 0.0f ? N : -N;
            bool front_face = dot(r.direction(), N) < 0.0f;
            float eta_ratio = front_face ? (1.0f / final_hi.mat.ior) : final_hi.mat.ior;
            // Fresnel float cos_theta = std::clamp(dot(-view_dir, surface_normal), 0.0f, 1.0f);
            float reflect_ratio = fresnel_schlick(cos_theta, eta_ratio);
            */

            vec3 surface_normal = dot(unit_vector(r.direction()), N) < 0.0f ? N : -N;
            bool entering = dot(unit_vector(r.direction()), N) < 0.0f;
            float n1 = entering ? 1.0f : final_hi.mat.ior;
            float n2 = entering ? final_hi.mat.ior : 1.0f;
            // float eta_ratio = n1 / n2;
            float eta_ratio = entering ? (1.0f / final_hi.mat.ior) : final_hi.mat.ior;
            float cos_theta = std::clamp(dot(-unit_vector(r.direction()), surface_normal), 0.0f, 1.0f);

            /* vec3 surface_normal = final_hi.normal; // physics normal
            bool entering = final_hi.front_face;

            float n1 = entering ? 1.0f : final_hi.mat.ior;
            float n2 = entering ? final_hi.mat.ior : 1.0f;

            float eta_ratio = n1 / n2;

            float cos_theta = std::clamp(dot(-unit_vector(r.direction()), surface_normal), 0.0f, 1.0f); */

            float reflect_ratio = fresnel_schlick(cos_theta, n1, n2); // Reflect
            ray reflected_ray(final_hi.p + eps * surface_normal, reflect(unit_vector(r.direction()), surface_normal));
            color reflected_color = bounce_attenuation * ray_color_phong(reflected_ray, world, lights, bvh_tree, recursion_limit - 1);
            // Refract
            vec3 refract_dir;
            color refracted_color(0, 0, 0);
            if (refract(unit_vector(r.direction()), surface_normal, eta_ratio, refract_dir))
            {
                ray refracted_ray(final_hi.p - eps * surface_normal, refract_dir);
                refracted_color = bounce_attenuation * ray_color_phong(refracted_ray, world, lights, bvh_tree, recursion_limit - 1);
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
            hit_color = mixed; // + specular;
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

color camera::ray_color_area_light(const ray &r, const std::vector<object> &world, bvh &bvh_tree, const area_light &al, int recursion_limit)
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
        color reflected_color = ray_color_area_light(reflected_ray, world, bvh_tree, al, recursion_limit - 1);

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
            color refracted_color = ray_color_area_light(refracted_ray, world, bvh_tree, al, recursion_limit - 1);

            result = (1.0f - hi.mat.transparency) * result + hi.mat.transparency * refracted_color;
        }
        else
        {
            // total internal reflection: treat as reflection
            vec3 reflect_dir = incident - 2.0f * dot(incident, surface_normal) * surface_normal;
            reflect_dir = unit_vector(reflect_dir);
            ray reflected_ray(hi.p + eps * surface_normal, reflect_dir);
            color reflected_color = ray_color_area_light(reflected_ray, world, bvh_tree, al, recursion_limit - 1);
            result = reflected_color;
        }
    }

    return result;
}

color camera::baseline_rec_trace(const ray &r, const bvh &bvh_tree, int depth)
{
    if (depth <= 0)
        return color(0, 0, 0);

    hit_info hi;
    if (!bvh_tree.hit(r, 1e-4f, FLT_MAX, hi))
        return color(0, 0, 0); // black background

    if (hi.mat.is_emissive())
        return hi.mat.emission;

    vec3 N = unit_vector(hi.normal);

    color sum(0, 0, 0);
    int samples = 1;
    for (int i = 0; i < samples; i++)
    {
        /* vec3 wo = sample_uniform_hemisphere(N); // UNIFORM
        float pdf = 1 / (2 * M_PI);
        float cos_theta = std::max(dot(wo, N), 0.0f);*/

        vec3 wo = sample_cosine_hemisphere(N); // COSINE
        float cos_theta = std::max(dot(wo, N), 0.0f);
        float pdf = cos_theta / M_PI;

        /* vec3 r_dir = unit_vector(r.direction()); // SPECULAR
        vec3 refl_dir = reflect(r_dir, N);
        float pdf = 0.0f;
        float s = phong_exponent_from_smoothness(hi.mat.smoothness);
        vec3 wo = sample_phong_lobe(refl_dir, N, s, pdf); // s= 32
        float cos_theta = std::max(dot(wo, N), 0.0f); */

        if (pdf <= 0.0f)
            return color(0, 0, 0);

        // BRDF (Lambertian)
        color fr = hi.mat.diffuse / M_PI;

        ray new_ray(hi.p + 1e-4f * N, wo);

        color Li = baseline_rec_trace(new_ray, bvh_tree, depth - 1);

        sum += fr * Li * cos_theta / pdf;
    }

    return sum / samples;
}

color camera::nee_rec_trace(const ray &r, const bvh &bvh_tree, int depth, const std::vector<area_light> &lights) // specular sampling schon einbauen?
{
    if (depth <= 0)
        return color(0, 0, 0);

    hit_info hi;
    if (!bvh_tree.hit(r, 1e-4f, FLT_MAX, hi))
        return color(0, 0, 0); // black background

    if (hi.mat.is_emissive())
        return hi.mat.emission;

    vec3 N = unit_vector(hi.normal);
    const float eps = 1e-4f;

    // Lambertian BRDF
    color fr = hi.mat.diffuse / M_PI;

    // ----------------------------------------------------
    // 1) Direct lighting via NEE (one sample)
    // ----------------------------------------------------
    color Ld(0, 0, 0);
    int light_samples = 1;

    if (!lights.empty())
    {
        const area_light &L = lights[0];

        // sample one point on the light
        // point3 lp = sample_area_light(L);

        // random
        // auto pts = uniform_sample_light(L, light_samples);
        // int total_samples = light_samples;

        // stratified jittered
        auto pts = stratified_sample_light(L, light_samples);
        int total_samples = light_samples * light_samples;

        for (const point3 &lp : pts)
        {
            vec3 Lvec = lp - hi.p;
            float dist2 = dot(Lvec, Lvec);
            vec3 wi = unit_vector(Lvec);

            float cos_s = std::max(dot(N, wi), 0.0f);
            float cos_l = std::max(dot(L.normal, -wi), 0.0f);

            if (cos_s > 0 && cos_l > 0)
            {
                if (visible(hi.p + wi * eps, lp, bvh_tree))
                {
                    float pdf_area = 1.0f / L.area;
                    float pdf_light = pdf_area * dist2 / cos_l;

                    Ld += fr * L.emission * cos_s / pdf_light;
                }
            }
        }

        Ld /= total_samples; // NOT light_samples for stratified
    }

    return Ld; // direct only
    // ----------------------------------------------------
    // 2) Indirect lighting via BRDF sampling
    // ----------------------------------------------------

    // try multiple samples
    vec3 wo = sample_cosine_hemisphere(N);
    float cos_theta = std::max(dot(wo, N), 0.0f);
    float pdf_brdf = cos_theta / M_PI;

    if (pdf_brdf <= 0.0f)
        return Ld;

    ray new_ray(hi.p + N * eps, wo);

    hit_info h2;
    if (!bvh_tree.hit(new_ray, 1e-4f, FLT_MAX, h2))
        return Ld; // background

    // ---------------------------
    // Trivial MIS: (or no mis to show problems)
    // Reject BRDF samples that hit the light
    // ---------------------------

    if (h2.mat.is_emissive()) // HERE
        return Ld;            // direct lighting already handled by NEE

    // Otherwise: indirect bounce
    color Li = nee_rec_trace(new_ray, bvh_tree, depth - 1, lights);

    color L_indirect = fr * Li * cos_theta / pdf_brdf;

    return Ld + L_indirect; // + L_indirect
}

color camera::newest_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, TextureManager &tex_manager)
{
    const float eps = 1e-4f; //-6 TODO dynamic (based on how far away the hit point is from the world origin)
    if (depth >= MAX_DEPTH)
    {
        if (depth == 0)
            timedOut++;
        return color(0, 0, 0);
    }
    // intersect
    hit_info hi;
    if (!bvh_tree.hit(r, eps, FLT_MAX, hi))
    {
        if (depth == 0)
            misses++;
        return color(0.2, 0.5, 0.8); // or return sky
    }
    // emitter
    if (hi.mat.is_emissive())
    {
        if (depth == 0)
            directHits++;
        return hi.mat.emission;
    }

    // geometric & shading normals
    vec3 N = unit_vector(hi.normal);          // geometric normal (possibly not oriented)
    vec3 dir_in = unit_vector(r.direction()); // incoming ray (points toward surface)

    bool front_face = dot(dir_in, N) < 0.0f;
    vec3 n = front_face ? N : -N; // oriented shading normal (use this everywhere) ?

    // ----------------------------------------------------
    // DIAGNOSTIC CODE: Normal Visualizer
    // ----------------------------------------------------

    // Choose which normal you want to inspect:
    // Test 1:
    /* vec3 visual_n = hi.normal; // Flat Geometric Normal
    // Test 2: vec3 visual_n = hi.shading_normal; // Smoothed Vertex Normal

    // Map the coordinates from [-1.0, 1.0] to a safe color range [0.0, 1.0]
    float red = (visual_n.x() + 1.0f) * 0.5f;
    float g = (visual_n.y() + 1.0f) * 0.5f;
    float b = (visual_n.z() + 1.0f) * 0.5f;

    // Return this color immediately to the screen, skipping all lighting math
    return color(red, g, b); */

    vec3 wi = -dir_in;
    vec3 refl_dir = reflect(dir_in, n);

    const float p_survive = 0.9f; // RR

    // glossy specular lobe
    // float s = phong_exponent_from_smoothness(hi.mat.smoothness);
    float s = hi.mat.spec_exponent;

    // local probabilites for light selection
    std::vector<float> local_probs(lights.size());
    local_probs = compute_local_light_weights(hi.p, n, lights);

    // -------------------------
    // 1) Delta/specular materials (handled first -> no NEE) // TODO RR for Delta materials
    // -------------------------

    // === PERFECT GLASS / DIELECTRIC ===
    if (hi.mat.transparency > 1e-6f)
    {
        float n1 = front_face ? 1.0f : hi.mat.ior;
        float n2 = front_face ? hi.mat.ior : 1.0f;
        float eta_ratio = n1 / n2;
        // float eta_ratio = front_face ? (1.0f / hi.mat.ior) : hi.mat.ior;
        float cos_theta = dot(-dir_in, n);
        if (cos_theta <= 0.0f)
            return color(0, 0, 0);
        cos_theta = std::min(cos_theta, 1.0f);

        // Fresnel reflectance
        float R = fresnel_schlick(cos_theta, n1, n2);

        vec3 refr_dir, refl_dir;

        // Reflection direction
        refl_dir = reflect(dir_in, n);
        // refl_dir = unit_vector(refl_dir);

        bool can_refract = refract(dir_in, n, eta_ratio, refr_dir);

        // Reflection ray
        ray refl_ray(hi.p + refl_dir * eps, refl_dir);
        color col_refl = newest_rec_trace(refl_ray, bvh_tree, depth + 1, lights, tex_manager);

        if (!can_refract)
            return col_refl; // Total internal reflection

        // Refraction ray
        ray refr_ray(hi.p + refr_dir * eps, refr_dir);

        color col_refr = newest_rec_trace(refr_ray, bvh_tree, depth + 1, lights, tex_manager); // TODO colored transparency (attenuation *)

        // Mix reflection & refraction
        return R * col_refl + (1.0f - R) * col_refr;
    }

    // --------------------------------------------------
    // Delta specular (perfect mirror)
    // --------------------------------------------------
    /* if (hi.mat.is_delta_specular)
    {

        color fr = hi.mat.specular; // mirror color (F0)

        ray new_ray(hi.p + refl_dir * eps, refl_dir);

        // delta: no MIS, no NEE, no cosine
        return newest_rec_trace(new_ray, bvh_tree, depth + 1, lights) * fr;
    } */

    int light_samples = 1;
    int brdf_samples = 1;

    // float SPEC_PROB = hi.mat.specular_prob;
    float SPEC_PROB = luminance(hi.mat.specular);
    float DIFF_PROB = 1.0f - SPEC_PROB;

    TraceResult tr;

    // -------------------------
    // 2) NEE: direct lighting
    // -------------------------
    color sum_light(0, 0, 0);
    if (!lights.empty())
    {
        // const area_light &L = lights[0];

        // importance
        int chosen_id = 0;
        float selected_weight = 1.0f;
        const area_light &L = sample_multiple_AL(lights, local_probs, selected_weight, chosen_id); // HERE

        // random
        // const area_light &L = lights[rand() % lights.size()];
        // float selected_weight = 1.0f / lights.size();

        /* std::vector<light_sample> pts; // random
        for (int s = 0; s < light_samples; s++)
        {
            light_sample sample_data;
            // sample() updates sample_data.normal and sample_data.pdf_area internally
            sample_data.p = sample(sample_data.normal, L, sample_data.pdf_area);
            pts.push_back(sample_data);
        } */

        // stratified
        int grid_dim = static_cast<int>(std::sqrt(light_samples));
        int total_actual_samples = grid_dim * grid_dim;

        std::vector<light_sample> pts;
        pts.reserve(total_actual_samples);

        // one sample
        /* light_sample ls;
        ls.p = sample(ls.normal, L, ls.pdf_area);
        pts.push_back(ls); */

        // stratified
        for (int y = 0; y < grid_dim; y++)
        {
            for (int x = 0; x < grid_dim; x++)
            {
                // Jittered stratification: maps x and y evenly into [0.0, 1.0]
                float r1 = (static_cast<float>(x) + rand_float()) / static_cast<float>(grid_dim);
                float r2 = (static_cast<float>(y) + rand_float()) / static_cast<float>(grid_dim);

                light_sample sample_data;
                // Pass r1 and r2 into your stratified triangle sampler
                sample_data.p = sample_stratified(sample_data.normal, L, sample_data.pdf_area, r1, r2);
                pts.push_back(sample_data);
            }
        }
        // auto pts = stratified_sample_light(L, light_samples);

        for (const light_sample &ls : pts)
        {
            vec3 Lvec = ls.p - hi.p;
            float dist_squared = Lvec.length_squared();
            float dist = std::sqrt(dist_squared);
            vec3 wo = Lvec / dist;

            float cos_s = dot(n, wo);
            float cos_l = dot(ls.normal, -wo);

            if (cos_s <= 0 || cos_l <= 0)
                continue;

            cos_s = std::min(cos_s, 1.0f);
            cos_l = std::min(cos_l, 1.0f);

            // visibility
            if (!visible(hi.p + wo * eps, ls.p, bvh_tree))
            {
                continue;
            }

            // light PDF in solid angle domain
            float pdf_light = (dist_squared / cos_l) * ls.pdf_area;
            float pdf_light_new = (pdf_light * selected_weight);

            // --- FULL BRDF evaluation (same as NEE)
            // color fr = brdf_diffuse(hi.mat) + brdf_specular(hi.mat, s, refl_dir, wo);
            color fr = tex_diffuse_new(hi, tex_manager) + brdf_specular(hi.mat, s, refl_dir, wo); // tex_diffuse_new(hi, tex_manager) +

            float phong_pdf = (s + 2.0f) * std::pow(std::max(dot(refl_dir, wo), 0.0f), s) / (2.0f * M_PI); // (s + 2.0f) * std::pow(std::max(dot(refl_dir, wo), 0.0f), s) / (2.0f * M_PI)

            float pdf_brdf = SPEC_PROB * phong_pdf + DIFF_PROB * (cos_s / M_PI); // NO MIS

            float w = power_heuristic(pdf_light_new, pdf_brdf);         // or balance
            sum_light += (w * fr * cos_s * L.emission / pdf_light_new); // w*
        }
        if (light_samples > 1)
            sum_light /= float(light_samples);
    }

    // RR 1
    float rr_weight = 1.0f;
    /* if (depth > MIN_DEPTH)
    {
        if (rand_float() > p_survive)
            return color(0, 0, 0);
        rr_weight = 1.0f / p_survive;
    }
    else
    {
        rr_weight = 1.0f;
    } */

    // -------------------------
    // 3) BRDF sampling (matches NEE mixture)
    // -------------------------

    color sum_brdf_full(0, 0, 0); // contributes to tr.full
    color sum_brdf_only(0, 0, 0); // contributes to tr.brdf_only

    for (int i = 0; i < brdf_samples; i++)
    {
        // NO MIS for mixed material
        bool choose_spec = rand_float() < SPEC_PROB;
        vec3 wo;
        float pdf_diff = 0.0f;
        float pdf_spec = 0.0f;

        if (choose_spec)
        {
            // sample specular, get pdf_spec directly
            wo = sample_phong_lobe(refl_dir, n, s, pdf_spec);
        }
        else
        {
            // sample diffuse
            wo = sample_cosine_hemisphere(n);
            float cos_n = dot(wo, n);
            if (cos_n <= 0.0f)
                continue;
            pdf_diff = cos_n / M_PI;
        }

        float cos_theta = std::max(0.0f, dot(wo, n));
        if (cos_theta <= 0.0f)
            continue;

        // Always compute BOTH pdfs for this direction:
        if (pdf_diff == 0.0f)
        {
            float cos_n = dot(wo, n);
            if (cos_n > 0.0f)
                pdf_diff = cos_n / M_PI;
        }
        if (pdf_spec == 0.0f)
        {
            float cos_r = std::max(0.0f, dot(refl_dir, wo));
            pdf_spec = (s + 2.0f) * std::pow(cos_r, s) / (2.0f * M_PI); // +1?
        }

        float pdf_brdf = SPEC_PROB * pdf_spec + DIFF_PROB * pdf_diff;
        if (pdf_brdf <= 0.0f)
            continue;

        // color fr = brdf_diffuse(hi.mat) + brdf_specular(hi.mat, s, refl_dir, wo);
        color fr = tex_diffuse_new(hi, tex_manager) + brdf_specular(hi.mat, s, refl_dir, wo); //  tex_diffuse_new(hi, tex_manager) +

        ray new_ray(hi.p + wo * eps, wo); // wo
        hit_info nh;
        if (!bvh_tree.hit(new_ray, eps, FLT_MAX, nh))
        {
            if (depth == 0)
                misses++;
            continue;
        }

        // DIAGNOSTIC: Inside your BRDF loop, right after if (bvh_tree.hit(new_ray, eps, FLT_MAX, nh))
        /* float dist = (nh.p - hi.p).length();

        // Map distance to a visible gradient (e.g., bright white when close, dark when far)
        float visual_dist = std::clamp(1.0f - (dist / 10.0f), 0.0f, 1.0f);

        return color(visual_dist, visual_dist, visual_dist); */

        if (nh.mat.is_emissive())
        {
            if (depth == 0)
                indirectHits++;
            // continue; // trivial MIS

            // existing direct-light MIS part (keep as is, but using pdf_brdf)
            // const area_light &L = lights[0];
            const area_light &L = lights[nh.mat.light_id];

            float dist = (nh.p - hi.p).length();
            float cos_l = std::max(0.0f, dot(L.normal, -wo));
            if (cos_l <= 0.0f)
                continue;

            float pdf_light = (dist * dist / cos_l) / L.area;
            float pdf_light_new = pdf_light * local_probs[nh.mat.light_id] / static_cast<float>(light_samples); // importance
            // float pdf_light_new = pdf_light * (1.0f / lights.size()); // random

            float w = power_heuristic(pdf_brdf, pdf_light_new);

            sum_brdf_full += w * nh.mat.emission * fr * cos_theta / pdf_brdf; // w *
        }
        else
        {
            color child = newest_rec_trace(new_ray, bvh_tree, depth + 1, lights, tex_manager);
            color contrib = fr * cos_theta / pdf_brdf;
            sum_brdf_full += rr_weight * child * contrib; // RR 2 rr_weight *

            if (depth == 0)
            {
                if (child.length() <= 0)
                {
                    timedOut++; // This path used up all its bounces and found nothing
                }
                else
                {
                    indirectHits++; // This path successfully hit a light eventually
                }
            }
        }

        // TODO MIS for mixed material
    }

    if (brdf_samples > 1)
        sum_brdf_full /= float(brdf_samples);

    // color result = (sum_light + sum_brdf_full);
    // return result;

    tr.full = (sum_light + sum_brdf_full);

    // tr.brdf_only = sum_brdf_only;
    return tr.full; // tr.full
}

color camera::newest_rec_trace_RR(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, color throughput)
{
    const float eps = 1e-6f;

    if (depth >= MAX_DEPTH)
        return color(0, 0, 0);

    // --- intersect
    hit_info hi;
    if (!bvh_tree.hit(r, eps, FLT_MAX, hi))
        return color(0, 0, 0);

    // --- emitter
    if (hi.mat.is_emissive())
        return throughput * hi.mat.emission;

    // geometric & shading normals
    vec3 N = unit_vector(hi.normal);
    vec3 dir_in = unit_vector(r.direction());

    bool front_face = dot(dir_in, N) < 0.0f;
    vec3 n = front_face ? N : -N;

    vec3 wi = -dir_in;
    vec3 refl_dir = reflect(dir_in, n);

    // ---- glossy specular lobe
    float s = phong_exponent_from_smoothness(hi.mat.smoothness);

    // -------------------------
    // 1) Delta/specular materials (no NEE, no RR)
    // -------------------------

    // PERFECT GLASS / DIELECTRIC
    if (hi.mat.transparency > 1e-6f)
    {
        float n1 = front_face ? 1.0f : hi.mat.ior;
        float n2 = front_face ? hi.mat.ior : 1.0f;
        float eta_ratio = n1 / n2;

        float cos_theta = dot(-dir_in, n);
        if (cos_theta <= 0.0f)
            return color(0, 0, 0);
        cos_theta = std::min(cos_theta, 1.0f);

        float R = fresnel_schlick(cos_theta, n1, n2);

        vec3 refr_dir, refl_dir_local;
        refl_dir_local = reflect(dir_in, n);

        bool can_refract = refract(dir_in, n, eta_ratio, refr_dir);

        // reflection
        ray refl_ray(hi.p + refl_dir_local * eps, refl_dir_local);
        color refl_col = newest_rec_trace_RR(refl_ray, bvh_tree, depth + 1, lights,
                                             throughput * R);

        if (!can_refract)
            return refl_col; // total internal reflection

        // refraction
        ray refr_ray(hi.p + refr_dir * eps, refr_dir);
        color refr_col = newest_rec_trace_RR(refr_ray, bvh_tree, depth + 1, lights,
                                             throughput * (1.0f - R));

        return refl_col + refr_col;
    }

    // PERFECT MIRROR
    if (hi.mat.is_delta_specular)
    {
        color fr = hi.mat.specular; // mirror color (F0)
        ray new_ray(hi.p + refl_dir * eps, refl_dir);

        // delta: no MIS, no NEE, usually no RR
        return newest_rec_trace_RR(new_ray, bvh_tree, depth + 1, lights,
                                   throughput * fr);
    }

    int light_samples = 1;
    int brdf_samples = 1; // keep it 1 for throughput-based RR!
    float SPEC_PROB = hi.mat.specular_prob;
    float DIFF_PROB = 1.0f - SPEC_PROB;

    // -------------------------
    // 2) NEE: direct lighting (scaled by throughput)
    // -------------------------

    color sum_light(0, 0, 0);
    if (!lights.empty())
    {
        const area_light &L = lights[0];
        // const area_light &L = sample_multiple_AL(lights);

        const float area = std::max(L.area, 1e-8f);
        auto pts = stratified_sample_light(L, light_samples);

        for (const point3 &lp : pts)
        {
            vec3 Lvec = lp - hi.p;
            float dist = Lvec.length();
            vec3 wo = Lvec / dist;

            float cos_s = dot(n, wo);
            float cos_l = dot(L.normal, -wo);

            if (cos_s <= 0 || cos_l <= 0)
                continue;

            cos_s = std::min(cos_s, 1.0f);
            cos_l = std::min(cos_l, 1.0f);

            if (!visible(hi.p + wo * eps, lp, bvh_tree))
                continue;

            float pdf_light = (dist * dist / cos_l) / area;

            float phong_pdf = (s + 1.0f) *
                              std::pow(std::max(dot(refl_dir, wo), 0.0f), s) /
                              (2.0f * M_PI);

            color fr = brdf_diffuse(hi.mat) +
                       brdf_specular(hi.mat, s, refl_dir, wo);

            float pdf_brdf = SPEC_PROB * phong_pdf +
                             DIFF_PROB * (cos_s / M_PI);

            float w = power_heuristic(pdf_light, pdf_brdf);

            sum_light += fr * w * cos_s * L.emission / pdf_light;
        }

        if (light_samples > 1)
            sum_light /= float(light_samples * light_samples);
    }

    // -------------------------
    // 3) BRDF sampling (path continuation)
    // -------------------------

    color sum_brdf(0, 0, 0);

    for (int i = 0; i < brdf_samples; i++)
    {
        bool choose_spec = rand_float() < SPEC_PROB;
        vec3 wo;
        float pdf_diff = 0.0f;
        float pdf_spec = 0.0f;

        if (choose_spec)
        {
            wo = sample_phong_lobe(refl_dir, n, s, pdf_spec);
        }
        else
        {
            wo = sample_cosine_hemisphere(n);
            float cos_n = dot(wo, n);
            if (cos_n <= 0.0f)
                continue;
            pdf_diff = cos_n / M_PI;
        }

        float cos_theta = std::max(0.0f, dot(wo, n));
        if (cos_theta <= 0.0f)
            continue;

        if (pdf_diff == 0.0f)
        {
            float cos_n = dot(wo, n);
            if (cos_n > 0.0f)
                pdf_diff = cos_n / M_PI;
        }
        if (pdf_spec == 0.0f)
        {
            float cos_r = std::max(0.0f, dot(refl_dir, wo));
            pdf_spec = (s + 1.0f) * std::pow(cos_r, s) / (2.0f * M_PI);
        }

        float pdf_brdf = SPEC_PROB * pdf_spec + DIFF_PROB * pdf_diff;
        if (pdf_brdf <= 0.0f)
            continue;

        color fr = brdf_diffuse(hi.mat) +
                   brdf_specular(hi.mat, s, refl_dir, wo);

        // --- update throughput for this bounce
        color new_throughput = throughput * (fr * cos_theta / pdf_brdf);

        // --- Russian roulette (throughput-based)
        if (depth >= MIN_DEPTH)
        {
            float lum = luminance(new_throughput);
            float p = std::clamp(lum, 0.05f, 0.95f); // tweakable

            if (rand_float() > p)
                continue; // path terminated

            new_throughput /= p;
        }

        ray new_ray(hi.p + wo * eps, wo);
        hit_info nh;
        if (!bvh_tree.hit(new_ray, eps, FLT_MAX, nh))
            continue;

        if (nh.mat.is_emissive())
        {
            const area_light &L = lights[0]; // or nh.mat.light_id

            float dist = (nh.p - hi.p).length();
            float cos_l = std::max(0.0f, dot(L.normal, -wo));
            if (cos_l <= 0.0f)
                continue;

            float pdf_light = (dist * dist / cos_l) / L.area;
            float w = power_heuristic(pdf_brdf, pdf_light);

            sum_brdf += new_throughput *
                        (w * nh.mat.emission);
        }
        else
        {
            // recurse with updated throughput
            color child = newest_rec_trace_RR(new_ray, bvh_tree,
                                              depth + 1, lights,
                                              new_throughput);
            sum_brdf += child;
        }
    }

    if (brdf_samples > 1)
        sum_brdf /= float(brdf_samples);

    // total radiance along this ray
    return throughput * sum_light + sum_brdf;
}

color camera::guided_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, VoxelGrid &grid, bool use_path_guiding)
{
    const float eps = 1e-6f;
    if (depth >= MAX_DEPTH)
    {
        if (depth == 0)
            timedOut++;
        return color(0, 0, 0);
    }

    hit_info hi;
    if (!bvh_tree.hit(r, eps, FLT_MAX, hi))
    {
        if (depth == 0)
            misses++;
        return color(0.3, 0.7, 1);
    }

    if (hi.mat.is_emissive())
    {
        if (depth == 0)
            directHits++;
        return hi.mat.emission;
    }

    vec3 N = unit_vector(hi.normal);
    vec3 dir_in = unit_vector(r.direction());

    bool front_face = dot(dir_in, N) < 0.0f;
    vec3 n = front_face ? N : -N;

    vec3 wi = -dir_in;
    vec3 refl_dir = reflect(dir_in, n);

    float s = phong_exponent_from_smoothness(hi.mat.smoothness);

    // === PERFECT GLASS / DIELECTRIC ===
    if (hi.mat.transparency > 1e-6f)
    {
        float n1 = front_face ? 1.0f : hi.mat.ior;
        float n2 = front_face ? hi.mat.ior : 1.0f;
        float eta_ratio = n1 / n2;
        float cos_theta = dot(-dir_in, n);
        if (cos_theta <= 0.0f)
            return color(0, 0, 0);
        cos_theta = std::min(cos_theta, 1.0f);

        float R = fresnel_schlick(cos_theta, n1, n2);

        vec3 refr_dir, refl_dir;
        refl_dir = reflect(dir_in, n);

        bool can_refract = refract(dir_in, n, eta_ratio, refr_dir);

        ray refl_ray(hi.p + refl_dir * eps, refl_dir);
        color col_refl = guided_rec_trace(refl_ray, bvh_tree, depth + 1, lights, grid, false);

        if (!can_refract)
            return col_refl;

        ray refr_ray(hi.p + refr_dir * eps, refr_dir);
        color col_refr = guided_rec_trace(refr_ray, bvh_tree, depth + 1, lights, grid, false);

        return R * col_refl + (1.0f - R) * col_refr;
    }

    // === PERFECT MIRROR ===
    if (hi.mat.is_delta_specular)
    {
        color fr = hi.mat.specular;
        ray new_ray(hi.p + refl_dir * eps, refl_dir);
        return guided_rec_trace(new_ray, bvh_tree, depth + 1, lights, grid, false) * fr;
    }

    int light_samples = 1;
    int brdf_samples = 1;

    float SPEC_PROB = hi.mat.specular_prob;
    float DIFF_PROB = 1.0f - SPEC_PROB;

    // -------------------------
    // 2) NEE: direct lighting
    // -------------------------
    color sum_light(0, 0, 0);
    if (!lights.empty())
    {
        const area_light &L = lights[0];
        // const area_light &L = sample_multiple_AL(lights);

        const float area = std::max(L.area, 1e-8f);
        auto pts = stratified_sample_light(L, light_samples);

        for (const point3 &lp : pts)
        {
            vec3 Lvec = lp - hi.p;
            float dist = Lvec.length();
            vec3 wo = Lvec / dist;

            float cos_s = dot(n, wo);
            float cos_l = dot(L.normal, -wo);

            if (cos_s <= 0 || cos_l <= 0)
                continue;

            cos_s = std::min(cos_s, 1.0f);
            cos_l = std::min(cos_l, 1.0f);

            if (!visible(hi.p + wo * eps, lp, bvh_tree))
            {
                continue;
            }

            float pdf_light = (dist * dist / cos_l) / area;
            color fr = tex_diffuse(hi);

            float phong_pdf = (s + 2.0f) * std::pow(std::max(dot(refl_dir, wo), 0.0f), s) / (2.0f * M_PI);
            float pdf_brdf = SPEC_PROB * phong_pdf + DIFF_PROB * (cos_s / M_PI);

            float w = power_heuristic(pdf_light, pdf_brdf);
            sum_light += w * fr * cos_s * L.emission / pdf_light;
        }

        if (light_samples > 1)
            sum_light /= float(light_samples * light_samples);
    }

    // -------------------------
    // 3) BRDF / Guiding sampling
    // -------------------------
    color brdf_sum(0, 0, 0);

    for (int i = 0; i < brdf_samples; i++)
    {
        vec3 wo;
        float pdf_brdf = 0.0f;

        if (use_path_guiding)
        {
            float alpha = 0.7f;
            float pdf_guided = 0.0f;
            float pdf_diff = 0.0f;

            if (rand_float() < alpha)
            {
                wo = grid.sample_guided_direction(hi.p, n, pdf_guided);
            }
            else
            {
                wo = sample_cosine_hemisphere(n);
                pdf_guided = grid.get_guided_pdf_interpolated(hi.p, n, wo);
            }
            float cos_theta = std::max(0.0f, dot(n, wo));
            pdf_diff = cos_theta / M_PI;
            pdf_brdf = alpha * pdf_guided + (1.0f - alpha) * pdf_diff;
        }
        else
        {
            wo = sample_cosine_hemisphere(n);
            float cos_theta = std::max(0.0f, dot(n, wo));
            pdf_brdf = cos_theta / M_PI;
        }

        if (pdf_brdf <= 0.0f)
            continue;

        float cos_theta = std::max(0.0f, dot(wo, n));
        if (cos_theta <= 0.0f)
            continue;

        color fr = tex_diffuse(hi);

        ray new_ray(hi.p + wo * eps, wo);
        hit_info nh;
        if (!bvh_tree.hit(new_ray, eps, FLT_MAX, nh))
        {
            if (depth == 0)
                misses++;
            continue;
        }

        if (nh.mat.is_emissive())
        {
            if (depth == 0)
                indirectHits++;

            const area_light &L = lights[0];

            float dist = (nh.p - hi.p).length();
            float cos_l = std::max(0.0f, dot(L.normal, -wo));
            if (cos_l <= 0.0f)
                continue;

            float pdf_light = (dist * dist / cos_l) / L.area;
            float w = power_heuristic(pdf_brdf, pdf_light);

            brdf_sum += nh.mat.emission * fr * cos_theta / pdf_brdf;
        }
        else
        {
            color child = guided_rec_trace(new_ray, bvh_tree, depth + 1, lights, grid, use_path_guiding);
            color contrib = fr * cos_theta / pdf_brdf;
            brdf_sum += child * contrib;

            if (depth == 0)
            {
                if (child.length() <= 0)
                {
                    timedOut++; // This path used up all its bounces and found nothing
                }
                else
                {
                    indirectHits++; // This path successfully hit a light eventually
                }
            }
        }

        // --- FIXED RECORD RADIANCE STEP (FIX 2) ---
        // Save direct lighting (sum_light) + indirect lighting (brdf_sum)
        color total_incoming_light = sum_light + brdf_sum;
        if (total_incoming_light.length() > 0.0f)
        {
            grid.record_radiance(hi.p, wo, total_incoming_light);

            if (test_point.length() == 0)
            {
                test_point = hi.p;
                std::cout << "Locked test point: " << hi.p << std::endl;
            }
        }
    }

    if (brdf_samples > 1)
        brdf_sum /= float(brdf_samples);

    return (sum_light + brdf_sum);
}

// Split ray-color
color camera::divided_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights)
{
    if (depth <= 0)
        return color(0, 0, 0);

    // --- intersect
    hit_info hi;
    if (!bvh_tree.hit(r, 1e-10f, FLT_MAX, hi))
        return color(0, 0, 0);

    // --- emitter (if we hit an emitter, return emission directly)
    color Le(0, 0, 0);
    if (hi.mat.is_emissive())
    {
        Le = hi.mat.emission;
        return Le;
    }

    // small offset
    const float eps = 1e-4f;

    // vectors
    vec3 N = unit_vector(hi.normal);
    vec3 dir_in = unit_vector(r.direction());
    vec3 wi = -dir_in;
    bool front_face = dot(dir_in, N) < 0.0f;
    vec3 n = front_face ? N : -N;

    /* if (hi.mat.flag == 3) // mirror
    {
        return recursion_mirror(r_dir_inv, N, hi, bvh_tree, depth, lights);
    }
    if (hi.mat.flag == 4) // transparent
    {
        return recursion_transparent(dir_in, N, hi, bvh_tree, depth, lights);
    } */

    return Le + next_event_estimation(wi, n, hi, bvh_tree, depth - 1, lights) + indirect_illumination(wi, n, hi, bvh_tree, depth - 1, lights);
}

color camera::next_event_estimation(const vec3 &wi, const vec3 &N, const hit_info &hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights)
{
    color L(0, 0, 0);

    // sample area light
    float light_area = std::max(lights[0].area, 1e-8f);
    vec3 sample_pos = sample_area_light(lights[0]);
    const vec3 n_light = unit_vector(lights[0].normal);

    // vector to light point
    vec3 Lvec = (sample_pos - hit.p);
    float dist = Lvec.length();
    vec3 wo = Lvec / dist;

    float cos_on_s = std::min(1.0f, dot(N, wo));               // cos between face normal and ray from light source
    float cos_on_l = std::min(1.0f, dot(n_light, wo * -1.0f)); // cos between light normal and ray from light source

    if (cos_on_s <= 0 || cos_on_l <= 0)
        return L;

    // ray r_from_light(sampled_point + 1e-04f * n_light, wi);

    // shadow test
    if (!visible(hit.p + wo * 1e-04f, sample_pos, bvh_tree)) // check!
        return L;

    float pdf_light = (dist * dist) / (cos_on_l * light_area);

    float pdf_bsdf = 0.0f;
    bool is_delta = false;

    color fr = eval_bsdf(hit.mat, hit.p, N, wi, wo, pdf_bsdf, is_delta);

    if (pdf_bsdf <= 0.0f)
        return L;

    float mis_weight = (pdf_light) / (pdf_bsdf + (pdf_light));
    return fr * mis_weight * cos_on_s * lights[0].emission / pdf_light;
}

color camera::indirect_illumination(const vec3 &wi, const vec3 &N, const hit_info &hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights)
{
    bool is_delta = false;
    float pdf_bsdf = 0.0f;

    vec3 wo = sample_from_bsdf(wi, N, hit.mat);

    float cos_wo = std::max(0.0f, dot(wo, N));

    color fr = eval_bsdf(hit.mat, hit.p, N, wi, wo, pdf_bsdf, is_delta);

    if (pdf_bsdf <= 0.0f)
        return color(0, 0, 0);

    vec3 sample_origin = hit.p + 1e-04f * wo;
    ray sample_ray(sample_origin, wo);

    hit_info next_hit;
    bool bang = bvh_tree.hit(sample_ray, 1e-04f, FLT_MAX, next_hit);

    if (!next_hit.mat.is_emissive())
    {
        color Li = divided_rec_trace(sample_ray, bvh_tree, depth - 1, lights);
        return Li * (fr * cos_wo / pdf_bsdf);
    }

    else
    {
        color Li = lights[0].emission;
        float area = lights[0].area;
        float pdf_light = 1.0f / area;

        vec3 light_pos = next_hit.p;
        vec3 n_light = lights[0].normal; // sample point

        float distance2 = (light_pos - sample_origin).length_squared(); // ^2?

        float cos_light = std::max(0.0f, dot(n_light, wo * -1.0f));
        if (cos_light <= 0.0f)
            return color(0, 0, 0);

        pdf_light = pdf_light * distance2 / cos_light;

        float mis_weight = pdf_bsdf / (pdf_bsdf + pdf_light);

        return Li * fr * cos_wo * mis_weight / pdf_bsdf;
    }
}

color camera::eval_bsdf(const material &m, const vec3 &p, const vec3 &n, const vec3 &wi, const vec3 &wo, float &pdf_factor, bool &is_delta)
{
    color shading;

    float cosI = std::max(0.0f, dot(n, wi));

    switch (m.flag)
    {
    case 1:
    {
        color diff = m.diffuse / M_PI;
        shading = diff;
        pdf_factor = cosI / M_PI;
        is_delta = false;
        break;
    }

    case 2:
    {
        float s = m.spec_exponent;

        // Diffuse BRDF
        color diff = m.diffuse / M_PI;

        // Phong specular BRDF
        vec3 r = unit_vector(reflect(wo, n)); // or wo.reflect(n) depending on your convention
        float roi = std::max(0.0f, dot(r, wi));

        color spec = m.specular * ((s + 2.0f) / (2.0f * M_PI)) * std::pow(roi, s);
        // Full BRDF is sum of lobes
        shading = (diff + spec) * 0.5;

        // PDFs for each lobe
        float diff_factor = cosI / M_PI;
        float spec_factor = (s + 1.0f) * std::pow(roi, s) / (2.0f * M_PI);
        // Sampling weights (example: based on energy)
        float Ed = 0.5;
        float Es = 0.5;
        float sum = Ed + Es;
        float wd = (sum > 0.0f) ? Ed / sum : 0.5f;
        float ws = (sum > 0.0f) ? Es / sum : 0.5f;

        pdf_factor = wd * diff_factor + ws * spec_factor;
        is_delta = false;
        break;
    }

    case 3:
        // handled elsewhere via delta reflection/refraction
        pdf_factor = 0.0f;
        is_delta = true; // here it *is* delta
        break;

    default:
        /* shading = color(1, 1, 1);
        pdf_factor = 0.0f;
        is_delta = false;
        break; */

        // TODO
        vec3 refl_dir = reflect(wi * -1.0f, n);
        float s = m.spec_exponent;
        float SPEC_PROB = m.specular_prob;
        float DIFF_PROB = 1 - SPEC_PROB;
        color fr = brdf_diffuse(m) + brdf_specular(m, s, refl_dir, wo);
        float phong_pdf = (s + 2.0f) * std::pow(std::max(dot(refl_dir, wo), 0.0f), s) / (2.0f * M_PI);
        pdf_factor = SPEC_PROB * phong_pdf + DIFF_PROB * (dot(n, wo) / M_PI);

        shading = fr;
        break;
    }

    return shading;
}

vec3 camera::sample_from_bsdf(vec3 wi, vec3 N, material m)
{
    vec3 wo;
    // return sample_cosine_hemisphere(N);
    switch (m.flag)
    {
    case 1:
    {
        wo = sample_cosine_hemisphere(N);
    }
    break;
    case 2:
    {
        float k = rand_float();
        if (k < 0.5)
            wo = phong_specular(N, wi, m.spec_exponent);
        else
            wo = sample_cosine_hemisphere(N);
    }
    break;
    default:
        wo = sample_uniform_hemisphere(N);
        break;
    }
    return wo;
}

color camera::recursion_mirror(vec3 I, vec3 N, hit_info hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights)
{
    vec3 reflection_dir = reflect(I, N);
    ray reflection_ray(hit.p + 1e-04f * reflection_dir, reflection_dir);
    return divided_rec_trace(reflection_ray, bvh_tree, depth - 1, lights);
}

color camera::recursion_transparent(vec3 I, vec3 N, hit_info hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights)
{
    bool front_face = dot(I, N) < 0.0f;
    vec3 n = front_face ? N : -N; // oriented shading normal (use this everywhere)

    float n1 = front_face ? 1.0f : hit.mat.ior;
    float n2 = front_face ? hit.mat.ior : 1.0f;
    float eta_ratio = n1 / n2;
    // float eta_ratio = front_face ? (1.0f / hi.mat.ior) : hi.mat.ior;
    float cos_theta = dot(-I, n);
    if (cos_theta <= 0.0f)
        return color(0, 0, 0);
    cos_theta = std::min(cos_theta, 1.0f);

    // Fresnel reflectance
    float R = fresnel_schlick(cos_theta, n1, n2);

    vec3 refr_dir, refl_dir;

    // Reflection direction
    refl_dir = reflect(I, n);
    refl_dir = unit_vector(refl_dir);

    bool can_refract = refract(I, n, eta_ratio, refr_dir);
    if (can_refract)
    {
        refr_dir = unit_vector(refr_dir);
    }

    // Reflection ray
    ray refl_ray(hit.p + refl_dir * 1e-04f, refl_dir);
    color col_refl = divided_rec_trace(refl_ray, bvh_tree, depth - 1, lights);

    if (!can_refract)
        return col_refl; // Total internal reflection

    // Refraction ray
    ray refr_ray(hit.p + refr_dir * 1e-04f, refr_dir);
    color col_refr = divided_rec_trace(refr_ray, bvh_tree, depth - 1, lights);

    // Mix reflection & refraction
    return R * col_refl + (1.0f - R) * col_refr;
}

// Rendering
void camera::render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree, VoxelGrid &guiding_grid, TextureManager &tex_manager)
{
    initialize();
    std::vector<color> framebuffer(image_width * image_height);

    // ErrorAnalyzer analyzer(image_width, image_height);
    int spp = 1;
    int N = int(std::sqrt(spp));
    bool guiding = false;

#pragma omp parallel for schedule(dynamic) // TODO threads
    for (int j = 0; j < image_height; j++)
    {
        for (int i = 0; i < image_width; i++)
        {
            color pixel_color(0, 0, 0);

            /* for (int py = 0; py < N; py++)
            {
                for (int px = 0; px < N; px++)
                {
                    float u = i + rand_double();
                    float v = j + rand_double(); // random sampling */
            /* for (int py = 0; py < N; py++)
            {
                for (int px = 0; px < N; px++)
                {
                    float u = i + (px + 0.5f) / N;
                    float v = j + (py + 0.5f) / N; // uniform sampling */

            for (int py = 0; py < N; py++)
            {
                for (int px = 0; px < N; px++)
                {
                    float u = i + (px + rand_double()) / N; // stratified jittered pixel sampling
                    float v = j + (py + rand_double()) / N;

                    point3 pixel_center = pixel_top_left + u * pixel_delta_h + v * pixel_delta_v;

                    ray r(camera_center, pixel_center - camera_center);
                    // color sample = nee_rec_trace(r, bvh_tree, 5, lights); // ,lights
                    color sample = newest_rec_trace(r, bvh_tree, 0, lights, tex_manager); // define recusion depth!, lights, throughput: color(1, 1, 1)
                    // color sample = guided_rec_trace(r, bvh_tree, 0, lights, guiding_grid, guiding);

                    // color sample = divided_rec_trace(r, bvh_tree, 3, lights); TODO
                    pixel_color += sample;

                    // contribution analysis
                    /* if (sample.length() > 0)
                    {
                        contributions++;
                    }
                    else
                    {
                        no_contribution++;
                    } */

                    // Error
                    // #pragma omp critical
                    // analyzer.add_pixel_sample(i, j, sample);
                }
            }
            pixel_color /= float(N * N);
            framebuffer[j * image_width + i] = pixel_color;
        }

        // progress
        if (j % 25 == 0)
        {
            float percentage = float(j) / float(image_height);
            std::cout << percentage << " done" << std::endl;
        }
    }

    std::ofstream out("image.ppm");
    out << "P3\n"
        << image_width << ' ' << image_height << "\n255\n";
    for (const auto &c : framebuffer)
        write_color(out, c);
    out.close();

    // Error
    /* std::vector<color> image = analyzer.load_ppm("image.ppm");
    std::vector<color> groundtruth = analyzer.load_ppm("groundtruthDiffuse2500.ppm");
    float mse = analyzer.mse(image, groundtruth);
    float psnr = analyzer.psnr(image, groundtruth);
    float ssim = analyzer.ssim(image, groundtruth);
    std::cout << "MSE = " << mse << ", PSNR = " << psnr << ", SSIM = " << ssim << "\n";
    float noise = analyzer.compute_global_noise();
    std::cout << "Image noise = " << noise << "\n"; */
}

void camera::old_render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree) // TODO area_light light_source
{
    initialize();
    std::ofstream out("image.ppm", std::ios::out); // write to ppm file
    out << "P3\n"
        << image_width << ' ' << image_height << "\n255\n";

    int spp = 1;
    int N = int(std::sqrt(spp));

    for (int j = 0; j < image_height; j++)
    {
        for (int i = 0; i < image_width; i++)
        {
            color pixel_color(0, 0, 0);
            for (int py = 0; py < N; py++)
            {
                for (int px = 0; px < N; px++)
                {

                    float u = i + (px + rand_double()) / N; // uniform vs stratified jittered pixel sampling
                    float v = j + (py + rand_double()) / N;
                    // Calculate color from ray of every pixel
                    auto pixel_center = pixel_top_left + (u * pixel_delta_h) + (v * pixel_delta_v);
                    auto ray_direction = pixel_center - camera_center;
                    ray r(camera_center, ray_direction);

                    // color pixel_color = ray_color_phong(r, world, lights, bvh_tree, 3); // define recursion depth!
                    // color pixel_color = baseline_rec_trace(r, bvh_tree, 5);
                    pixel_color += nee_rec_trace(r, bvh_tree, 1, lights);

                    if (!std::isfinite(pixel_color.x() || !std::isfinite(pixel_color.y() || !std::isfinite(pixel_color.z()))))
                    {
                        std::cout << "error pixel\n";
                    }
                }
            }
            pixel_color /= float(N * N);

            // color pixel_color = ray_color_triangle(r, triangle(point3(-0.5, 0, -1), point3(0, 1, -1), point3(0.5, 0, -1)));
            write_color(out, pixel_color);
            // std_write_color(out, pixel_color);
        }
    }

    out.close(); // close file
}

void camera::new_render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree, VoxelGrid &guiding_grid)
{
    initialize();

    // =========================================================================
    // MULTI-PASS PROGRESSIVE TRAINING LOOP
    // =========================================================================
    int num_training_iterations = 5; // sequential learning updates
    int samples_per_iteration = 4;   // Small sample size per loop iteration
    int N_train = static_cast<int>(std::sqrt(samples_per_iteration));

    for (int iter = 0; iter < num_training_iterations; iter++)
    {
        std::cout << "--> Starting Training Iteration " << (iter + 1)
                  << "/" << num_training_iterations << "..." << std::endl;

        // Reset our global debug marker for each pass
        test_point = point3(0, 0, 0);

        for (int j = 0; j < image_height; j++)
        {
            for (int i = 0; i < image_width; i++)
            {
                for (int py = 0; py < N_train; py++)
                {
                    for (int px = 0; px < N_train; px++)
                    {
                        float u = i + (px + rand_double()) / N_train;
                        float v = j + (py + rand_double()) / N_train;

                        point3 pixel_center = pixel_top_left + u * pixel_delta_h + v * pixel_delta_v;
                        ray r(camera_center, pixel_center - camera_center);

                        // CRITICAL:
                        // Iteration 0 has no prior knowledge -> use_path_guiding = false (blind exploration)
                        // Iterations 1, 2, 3 -> use_path_guiding = true!
                        // It uses the prior iteration's guide to discover even deeper light paths.
                        bool guiding_enabled = (iter > 0);

                        guided_rec_trace(r, bvh_tree, 0, lights, guiding_grid, guiding_enabled);
                    }
                }
            }
        }

        // Bake and freeze the data collected in this iteration.
        // This copies current 'spherical_bins' to 'guiding_pdf' and zeroes out 'spherical_bins'.
        std::cout << "    Baking distributions for iteration " << (iter + 1) << "..." << std::endl;
        guiding_grid.finalize_and_build_distributions();

        // Export a progressive diagnostic heatmap frame to see it converge!
        if (test_point.length() > 0.0f)
        {
            std::string heatmap_name = "voxel_heatmap_iter_" + std::to_string(iter + 1) + ".ppm";
            guiding_grid.export_voxel_heatmap_ppm2(test_point, heatmap_name);
        }

        std::cout << "guiding pdf (testpoint (0,0)): " << guiding_grid.get_voxel(test_point).guiding_pdf[0][0] << std::endl;
        std::cout << "direct: " << directHits << "indirect: " << indirectHits << "misses: " << misses << "timeout: " << timedOut << std::endl;
    }

    // =========================================================================
    // PASS 2: FINAL PRODUCTION PASS (Use the beautifully refined guides)
    // =========================================================================
    int production_spp = 16;
    int N_prod = static_cast<int>(std::sqrt(production_spp));
    std::vector<color> framebuffer(image_width * image_height);
    std::cout << "--> Starting Production Pass with Fully Trained Guides (" << production_spp << " SPP)..." << std::endl;

    // #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < image_height; j++)
    {
        for (int i = 0; i < image_width; i++)
        {
            color pixel_color(0, 0, 0);

            for (int py = 0; py < N_prod; py++)
            {
                for (int px = 0; px < N_prod; px++)
                {
                    float u = i + (px + rand_double()) / N_prod;
                    float v = j + (py + rand_double()) / N_prod;

                    point3 pixel_center = pixel_top_left + u * pixel_delta_h + v * pixel_delta_v;
                    ray r(camera_center, pixel_center - camera_center);

                    pixel_color += guided_rec_trace(r, bvh_tree, 0, lights, guiding_grid, true);
                }
            }

            framebuffer[j * image_width + i] = pixel_color / static_cast<float>(production_spp);
        }
    }

    std::cout << "--> Rendering complete!" << std::endl;
    std::ofstream out("image.ppm");
    out << "P3\n"
        << image_width << ' ' << image_height << "\n255\n";
    for (const auto &c : framebuffer)
        write_color(out, c);
    out.close();
}