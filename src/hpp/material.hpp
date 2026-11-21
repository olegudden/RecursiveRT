#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "color.hpp"

#include <string>

class material
{
public:
    // Phong
    color ambient;
    color diffuse;
    color specular;
    float spec_exponent = 16.0f;

    // for ray intersections
    float transparency; // 0 = opaque, 1 = fully transparent
    float reflectivity; // 0 = matte, 1 = perfect mirror
    // float diffuse_weight = 1.0f - reflectivity - transparency;

    color absorption; // colored tint of transparent object
    float ior = 1.5f; // Index of refraction

    float smoothness = 0.0f;
    float roughness = 1.0f - smoothness;
    float specular_prob = 0.0f;
    bool is_delta_specular = false;

    float metallic = 0.0f;
    float clearcoat_roughness;
    float clearcoat_thickness;
    float sheen = 0.0f;
    float subsurface_scale = 0.0f;
    float subsurface_weight = 0.0f;

    int flag;

    // for (area) lights
    color emission = color(0, 0, 0);
    int light_id;

    // for textures
    std::string diffuse_texname;
    bool has_diffuse_tex;
    int diffuse_tex_id = -1;

    // emissive
    std::string emissive_texname;
    bool has_emissive_tex;
    int emissive_tex_id = -1;

    bool is_emissive() const
    {
        return (emission[0] > 0 || emission[1] > 0 || emission[2] > 0);
    }

private:
};

inline float phong_exponent_from_roughness(float roughness)
{
    roughness = std::max(roughness, 0.02f);
    float s = (2.0f / (roughness * roughness)) - 2.0f;
    return std::min(s, 500.0f); // or 1000
}

inline float phong_exponent_from_smoothness(float smoothness)
{ // Clamp to avoid zero exponent
    smoothness = std::clamp(smoothness, 0.0f, 1.0f);
    return std::pow(1000.0f, smoothness); // s ∈ [1, 1000]
}

/* inline float fresnel(float cos_theta, float ior)
{
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0; // base reflectance at normal incidence
    return r0 + (1.0f - r0) * std::pow(1.0f - cos_theta, 5.0f);
} */
inline float fresnel(const vec3 &wi, const vec3 &N, float ior)
{
    float cosi = std::clamp(dot(-wi, N), -1.0f, 1.0f);
    float etai = 1.0f;
    float etat = ior;
    if (cosi > 0.0f)
        std::swap(etai, etat);
    float r0 = (etai - etat) / (etai + etat);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - fabs(cosi), 5.0f);
}

inline float schlick_fresnel(float cos_theta, float ior)
{
    float r0 = (ior - 1.0f) / (ior + 1.0f);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
}

inline float fresnel_schlick(float cos_theta, float n1, float n2)
{
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cos_theta, 5.0f);
}

// Fresnel Schlick using a vector F0 (RGB color)
inline vec3 fresnel_schlick_vec(float cosTheta, const vec3 &F0)
{
    // Clamp for safety
    cosTheta = std::clamp(cosTheta, 0.0f, 1.0f);

    // Schlick approximation
    return F0 + (vec3(1.0f, 1.0f, 1.0f) - F0) * pow(1.0f - cosTheta, 5.0f);
}

/* inline material make_material(material_preset preset, color base)
{
    material m{};

    m.ambient = base;
    m.diffuse = base;
    m.specular = color(1, 1, 1);
    m.spec_exponent = 64.0f;

    m.transparency = 0.0f;
    m.ior = 1.0f;

    m.smoothness = 0.0f;
    m.absorption = color(0, 0, 0);

    m.emission = color(0, 0, 0);
    m.light_id = -1;

    switch (preset)
    {
    case material_preset::diffuse:
        m.smoothness = 0.0f;
        m.transparency = 0.0f;
        m.ior = 1.0f;
        m.specular = color(0, 0, 0);

        m.specular_prob = 0.0f;
        m.is_delta_specular = false;
        break;

    case material_preset::metal:
        m.smoothness = 0.0f;
        m.transparency = 0.0f;
        m.ior = 1.0f;
        m.specular = base; // metals get colored specular
        break;

    case material_preset::mirror:
        m.diffuse = color(0, 0, 0);
        m.specular = color(1, 1, 1);
        m.smoothness = 1.0f;
        m.specular_prob = 1.0f;
        m.is_delta_specular = true;
        break;

    case material_preset::plastic:
        m.roughness = 0.3f;
        m.metallic = 0.0f;
        m.transparency = 0.0f;
        m.ior = 1.5f;
        break;

    case material_preset::plastic1:
        // Glossy white plastic
        m.diffuse = color(0.9f, 0.9f, 0.9f);
        m.specular = color(0.1f, 0.1f, 0.1f);
        m.smoothness = 1;
        m.specular_prob = 0.02f;
        m.is_delta_specular = false;
        break;

    case material_preset::plastic2:
        // Rough reflective metal
        m.diffuse = color(0.8, 0.2, 0.5);  // 0.8, 0.2, 0.5 pink
        m.specular = color(0.5, 0.5, 0.5); // reflective but not mirror
        m.smoothness = 0.9f;               // rough 0.1
        m.specular_prob = 0.7f;
        m.is_delta_specular = false;
        break;

    case material_preset::plastic3:
        m.diffuse = color(0.3, 0.3, 0.3);
        m.specular = color(0.7, 0.7, 0.7);
        m.smoothness = 0.7f;
        m.specular_prob = 0.4f;
        m.is_delta_specular = false;
        break;

    case material_preset::plastic4:
        m.diffuse = color(0.0, 0.0, 0.0);
        m.specular = color(1, 1, 1);
        m.smoothness = 0.9f;
        m.specular_prob = 0.9f;
        m.is_delta_specular = false;
        break;

    case material_preset::glass:
        m.smoothness = 1.0f;
        m.transparency = 1.0f;
        m.ior = 1.5f;
        break;

    case material_preset::frosted_glass:
        m.smoothness = 0.5f;
        m.transparency = 1.0f;
        m.ior = 1.5f;
        break;

    case material_preset::ice:
        m.smoothness = 0.9f;
        m.transparency = 1.0f;
        m.ior = 1.31f;
        m.absorption = color(0.9f, 0.95f, 1.0f); // slight blue tint
        break;

    case material_preset::light:
        m.roughness = 0.0f;
        m.transparency = 0.0f;
        m.ior = 1.0f;
        m.emission = color(50, 50, 50);
        break;
    }
    return m;
}*/

/* inline material mat_from_type(material_type mt, color base)
{
    material m{};

    // Universal defaults
    m.diffuse = base;
    m.specular = color(1.0f, 1.0f, 1.0f);
    m.absorption = color(1.0f, 1.0f, 1.0f);
    m.emission = color(0.0f, 0.0f, 0.0f);

    m.spec_exponent = 8.0f; // for Phong glossy
    m.smoothness = 0.5f;    // for GGX (if you use it later)
    m.transparency = 0.0f;
    m.ior = 1.0f;

    switch (mt)
    {
    case material_type::diffuse:
        m.diffuse = base;
        m.specular = color(0.0f, 0.0f, 0.0f); // no specular
        m.smoothness = 0.0f;
        m.transparency = 0.0f;
        m.ior = 1.0f;
        m.flag = 1;
        break;

    case material_type::specular: // glossy plastic
        m.diffuse = base;
        m.specular = color(0.6, 0.6, 0.6); // white specular lobe
        m.smoothness = 0.7f;               // glossy but not mirror
        m.spec_exponent = 32.0f;           // Phong exponent
        m.transparency = 0.0f;
        m.ior = 1.5f; // typical plastic IOR
        m.flag = 2;
        break;

    case material_type::mirror: // perfect specular
        m.diffuse = color(0.0f, 0.0f, 0.0f);
        m.specular = color(1.0f, 1.0f, 1.0f);
        m.smoothness = 1.0f;        // indicates delta distribution
        m.spec_exponent = INFINITY; // or any sentinel you use
        m.transparency = 0.0f;
        m.ior = 1.0f;
        m.flag = 3;
        break;

    case material_type::transparent: // glass / dielectric
        m.diffuse = color(0.0f, 0.0f, 0.0f);
        m.specular = color(1.0f, 1.0f, 1.0f);
        m.absorption = color(1.0f, 1.0f, 1.0f);
        m.smoothness = 1.0f; // delta again
        m.spec_exponent = INFINITY;
        m.transparency = 1.0f; // enables refraction
        m.ior = 1.5f;          // glass IOR
        m.flag = 4;
        break;

    default:
        break;
    }

    return m;
} */

#endif