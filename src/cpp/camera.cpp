#define _USE_MATH_DEFINES

#include "../hpp/camera.hpp"
#include "../hpp/error.hpp"
#include "../hpp/texture.hpp"
#include "../hpp/mesh_loader.hpp"
#include <iostream>
#include <omp.h>
#include <sstream>
#include <chrono>

const int MAX_DEPTH = 2;
const int MIN_DEPTH = 1;

// Allocate the actual storage space for the vector here:
std::vector<VisualRayPath> g_heatmap_full_paths;
point3 test_point(0, 0, 0);
vec3 test_point_normal(0, 0, 0);

// static int lightpath_counter = 0;

void camera::get_heatmap_color(const color &radiance, float &out_r, float &out_g, float &out_b)
{
    if (radiance.length() <= 1e-12f)
    {
        out_r = 0.2f;
        out_g = 0.2f;
        out_b = 0.2f; // Base gray for unhit segments
        return;
    }
    // Individual channel Reinhard tone mapping
    out_r = radiance.x() / (radiance.x() + 1.0f);
    out_g = radiance.y() / (radiance.y() + 1.0f);
    out_b = radiance.z() / (radiance.z() + 1.0f);
}

void camera::export_radiance_forward_paths(
    const std::string &base_scene_filename,
    const std::string &base_mtl_filename,
    const std::string &output_obj_filename,
    const std::vector<VisualRayPath> &paths,
    const VoxelGrid &guiding_grid)
{
    std::ifstream in_scene(base_scene_filename);
    std::ofstream out_obj(output_obj_filename);

    // Generate the path for the unique, standalone path material file
    std::string unique_mtl_filename = output_obj_filename;
    size_t ext_pos = unique_mtl_filename.find_last_of(".");
    if (ext_pos != std::string::npos)
    {
        unique_mtl_filename = unique_mtl_filename.substr(0, ext_pos);
    }
    unique_mtl_filename += "_materials.mtl";

    std::ofstream out_mtl(unique_mtl_filename); // Fresh write mode, guarantees no lockups

    if (!out_obj.is_open())
    {
        std::cerr << "Error: Cannot open OBJ output file: " << output_obj_filename << std::endl;
        return;
    }
    if (!out_mtl.is_open())
    {
        std::cerr << "Error: Cannot create custom MTL output file: " << unique_mtl_filename << std::endl;
        return;
    }

    int vertex_counter = 0;

    // --- LINK BOTH MATERIAL LIBRARIES ---
    // First, link the original Cornell box materials (e.g., red, green walls)
    std::string base_mtl_short = base_mtl_filename.substr(base_mtl_filename.find_last_of("/\\") + 1);
    out_obj << "mtllib " << base_mtl_short << "\n";

    // Second, link our custom path diagnostic colors
    std::string path_mtl_short = unique_mtl_filename.substr(unique_mtl_filename.find_last_of("/\\") + 1);
    out_obj << "mtllib " << path_mtl_short << "\n\n";

    // 1. Copy the original scene file EXACTLY as it is, just tracking the vertex count
    if (in_scene.is_open())
    {
        std::string line;
        while (std::getline(in_scene, line))
        {
            std::stringstream ss(line);
            std::string prefix;
            ss >> prefix;

            // Skip the mtllib line from the source file since we wrote both explicitly at the top
            if (prefix == "mtllib")
                continue;

            if (prefix == "v")
                vertex_counter++;
            out_obj << line << "\n";
        }
        in_scene.close();
    }

    int next_global_vertex_idx = vertex_counter + 1;
    out_obj << "\ng diagnostic_tubes\n";
    const int segments = 6;

    // 2. Parse paths and export dynamic geometry + distinct segment materials
    for (size_t p_idx = 0; p_idx < paths.size(); ++p_idx)
    {
        const auto &path = paths[p_idx];
        if (path.vertices.size() < 2)
            continue;

        for (size_t s = 0; s < path.vertices.size() - 1; ++s)
        {
            const auto &p0 = path.vertices[s];
            const auto &p1 = path.vertices[s + 1];

            // Target color assignment based on forward intersection vertex radiance
            color segment_radiance(0, 0, 0);
            if (s + 1 < path.vertex_radiance.size())
            {
                segment_radiance = path.vertex_radiance[s + 1];
            }

            // Map colors to a clean chromatic ratio to preserve genuine hue data
            float r = 0.0f, g = 0.0f, b = 0.0f;
            float max_channel = std::max({segment_radiance.x(), segment_radiance.y(), segment_radiance.z()});

            if (max_channel > 1e-5f)
            {
                float r_ratio = segment_radiance.x() / max_channel;
                float g_ratio = segment_radiance.y() / max_channel;
                float b_ratio = segment_radiance.z() / max_channel;

                float intensity = max_channel / (max_channel + 1.0f);
                r = r_ratio * intensity;
                g = g_ratio * intensity;
                b = b_ratio * intensity;
            }

            std::string mat_name = "Mat_Path" + std::to_string(p_idx) + "_Seg" + std::to_string(s);

            // Write the custom path material properties cleanly to the separate file
            out_mtl << "newmtl " << mat_name << "\n";
            out_mtl << "  Kd " << r << " " << g << " " << b << "\n";
            if (max_channel > 15.0f)
            {
                out_mtl << "  Ke " << r * 0.5f << " " << g * 0.5f << " " << b * 0.5f << "\n";
            }
            out_mtl << "\n";

            out_obj << "usemtl " << mat_name << "\n";

            // --- RADIANCE TAPERED THICKNESS ---
            float e0 = 0.0f;
            float e1 = 0.0f;
            if (s < path.vertex_radiance.size())
                e0 = path.vertex_radiance[s].length();
            if (s + 1 < path.vertex_radiance.size())
                e1 = path.vertex_radiance[s + 1].length();

            if (s == 0 && e0 <= 1e-6f)
            {
                e0 = e1 * 0.15f;
            }

            float r0 = 0.0012f + std::min(0.02f, 0.0008f * e0);
            float r1 = 0.0012f + std::min(0.02f, 0.0008f * e1);

            vec3 dir = unit_vector(p1 - p0);
            vec3 up = (std::abs(dir.x()) > 0.9f) ? vec3(0, 1, 0) : vec3(1, 0, 0);
            vec3 u = unit_vector(cross(dir, up));
            vec3 v = cross(u, dir);

            int r0_base = next_global_vertex_idx;
            for (int i = 0; i < segments; ++i)
            {
                float theta = (2.0f * M_PI * i) / segments;
                point3 pt = p0 + r0 * (std::cos(theta) * u + std::sin(theta) * v);
                out_obj << "v " << pt.x() << " " << pt.y() << " " << pt.z() << "\n";
                next_global_vertex_idx++;
            }

            int r1_base = next_global_vertex_idx;
            for (int i = 0; i < segments; ++i)
            {
                float theta = (2.0f * M_PI * i) / segments;
                point3 pt = p1 + r1 * (std::cos(theta) * u + std::sin(theta) * v);
                out_obj << "v " << pt.x() << " " << pt.y() << " " << pt.z() << "\n";
                next_global_vertex_idx++;
            }

            for (int i = 0; i < segments; ++i)
            {
                int next_i = (i + 1) % segments;
                int v0 = r0_base + i;
                int v1 = r0_base + next_i;
                int v2 = r1_base + next_i;
                int v3 = r1_base + i;
                out_obj << "f " << v0 << " " << v3 << " " << v2 << " " << v1 << "\n";
            }
        }
    }

    // TODO: --- INTEGRATE FULL VOXEL GRID EXPORT ---
    // --- RELATIVE INDEXING GRID EXPORT ---
    out_obj << "\ng full_voxel_grid\n";
    out_obj << "usemtl Mat_VoxelGrid_Full\n";

    // Calculate the dimensions of a single voxel based on total bounds and resolution
    float step_x = (guiding_grid.bounds_max.x() - guiding_grid.bounds_min.x()) / (float)guiding_grid.res_x;
    float step_y = (guiding_grid.bounds_max.y() - guiding_grid.bounds_min.y()) / (float)guiding_grid.res_y;
    float step_z = (guiding_grid.bounds_max.z() - guiding_grid.bounds_min.z()) / (float)guiding_grid.res_z;

    // DYNAMIC LOOPS: These use the member variables from your grid object
    for (int x = 0; x < guiding_grid.res_x; x++)
    {
        for (int y = 0; y < guiding_grid.res_y; y++)
        {
            for (int z = 0; z < guiding_grid.res_z; z++)
            {

                // Calculate the corner positions for this specific voxel
                point3 v_min = guiding_grid.bounds_min + vec3(x * step_x, y * step_y, z * step_z);
                point3 v_max = v_min + vec3(step_x, step_y, step_z);

                // Write 8 vertices for this voxel
                out_obj << "v " << v_min.x() << " " << v_min.y() << " " << v_min.z() << "\n"
                        << "v " << v_max.x() << " " << v_min.y() << " " << v_min.z() << "\n"
                        << "v " << v_max.x() << " " << v_max.y() << " " << v_min.z() << "\n"
                        << "v " << v_min.x() << " " << v_max.y() << " " << v_min.z() << "\n"
                        << "v " << v_min.x() << " " << v_min.y() << " " << v_max.z() << "\n"
                        << "v " << v_max.x() << " " << v_min.y() << " " << v_max.z() << "\n"
                        << "v " << v_max.x() << " " << v_max.y() << " " << v_max.z() << "\n"
                        << "v " << v_min.x() << " " << v_max.y() << " " << v_max.z() << "\n";

                // Relative indexing: draws the wireframe box using the 8 vertices just written
                out_obj << "l -8 -7 -6 -5 -8\n" // Bottom ring
                        << "l -4 -3 -2 -1 -4\n" // Top ring
                        << "l -8 -4\n"          // Pillar 1
                        << "l -7 -3\n"          // Pillar 2
                        << "l -6 -2\n"          // Pillar 3
                        << "l -5 -1\n";         // Pillar 4
            }
        }
    }
    out_obj.close();
    out_mtl.close();
}

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

    float denom = a + b;
    if (denom <= 1e-6f)
        return 0.0f; // Blocks 0 / 0 NaN generation completely

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
color camera::tex_base_color(const hit_info &hi, TextureManager &tex_manager)
{
    color diffuse_color = hi.mat.diffuse;

    if (hi.mat.has_diffuse_tex)
    {
        // Dynamically fetch the pixel data belonging to this specific polygon's material index
        diffuse_color = tex_manager.sample_texture_from_cache(tex_manager, hi.mat.diffuse_tex_id, hi.u, hi.v);
    }

    return diffuse_color;
}

color camera::evaluate_disney_diffuse(
    const color &base_color,
    float roughness,
    float subsurface_weight,
    float NdotL,
    float NdotV,
    float NdotH,
    float LdotH)
{
    // --- 1. Standard Burley Retro-Reflection Term ---
    float FD90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
    float f_L = 1.0f + (FD90 - 1.0f) * std::pow(1.0f - NdotL, 5.0f);
    float f_V = 1.0f + (FD90 - 1.0f) * std::pow(1.0f - NdotV, 5.0f);
    color f_base_diffuse = base_color * (f_L * f_V / M_PI);

    // If real SSS is active, the surface diffuse layer is attenuated
    // because that energy has penetrated INTO the volume instead.
    return (1.0f - subsurface_weight) * f_base_diffuse;
}

color camera::brdf_specular(const material &m, const float s, const vec3 &refl_dir, const vec3 &wo)
{
    float cosAlpha = std::max(0.0f, dot(refl_dir, wo)); //-wo?
    // return m.specular * ((s + 2.0f) / (2.0f * M_PI)) * std::pow(cosAlpha, s);
    return m.specular * ((s + 2.0f) / (2.0f * M_PI)) * std::pow(cosAlpha, s);
}

float camera::ndf_ggx(float NdotH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float denom_term = NdotH * NdotH * (alpha2 - 1.0f) + 1.0f;
    float denominator = M_PI * denom_term * denom_term;

    if (denominator < 1e-7f)
        return 0.0f;

    return alpha2 / denominator;
}

vec3 camera::sample_ggx_lobe(const vec3 &n, float roughness, float r1, float r2, float &pdf)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float phi = 2.0f * M_PI * r1;

    float cos_theta = std::sqrt((1.0f - r2) / (1.0f + (alpha2 - 1.0f) * r2));
    float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));

    vec3 t, b;
    make_orthonormal_basis(n, t, b);

    vec3 world_H = sin_theta * std::cos(phi) * t +
                   sin_theta * std::sin(phi) * b +
                   cos_theta * n;

    float denom = (cos_theta * cos_theta * (alpha2 - 1.0f) + 1.0f);
    pdf = (alpha2 * cos_theta) / (M_PI * denom * denom);

    return world_H;
}

float camera::evaluate_clearcoat(float NdotH, float clearcoat_roughness, float NdotL, float NdotV, float VdotH)
{
    float alpha = clearcoat_roughness;
    float alpha2 = alpha * alpha;

    float denom = (alpha2 - 1.0f) * NdotH * NdotH + 1.0f;
    if (denom <= 0.0f)
        return 0.0f;

    float D_cc = (alpha2 - 1.0f) / (M_PI * std::log(alpha2) * denom);
    if (alpha2 == 1.0f)
        D_cc = 1.0f / M_PI;

    float G_cc = (1.0f / (NdotL * 0.75f + 0.25f)) * (1.0f / (NdotV * 0.75f + 0.25f));

    // RETURN ONLY THE CORE MICROFACET PIECE (No F_cc here!)
    return 0.25f * D_cc * G_cc;
}

color camera::evaluate_sheen(const color &base_color, float VdotH)
{
    // Automatically use a 50% blend, or simplify C_sheen to color(1.0f, 1.0f, 1.0f) for purely white fuzz
    float luminance_val = luminance(base_color);
    color tint = (luminance_val > 0.0f) ? (base_color / luminance_val) : color(1.0f, 1.0f, 1.0f);
    color C_sheen = 0.5f * color(1.0f, 1.0f, 1.0f) + 0.5f * tint;

    float fresnel_sheen = std::pow(std::clamp(1.0f - VdotH, 0.0f, 1.0f), 5.0f);
    return C_sheen * fresnel_sheen;
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

    // Use the EXACT same basis function that your GGX sampler uses!
    vec3 tangent, bitangent;
    make_orthonormal_basis(N, tangent, bitangent);

    // Transform directly to world space using the uniform basis vectors
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
    const float eps = 1e-4f;

    // 1. Max Depth Limit Reached
    if (depth >= MAX_DEPTH)
    {
        return color(0, 0, 0);
    }

    // 2. Primary / Secondary Ray Intersection
    hit_info hi;
    if (!bvh_tree.hit(r, eps, FLT_MAX, hi))
    {
        if (depth == 0)
            misses++;
        return color(0, 0, 0); // Sky / Background
    }

    // 3. Emitter Intersection
    if (hi.mat.is_emissive())
    {
        if (depth == 0)
            directHits++;
        return hi.mat.emission;
    }

    // Local flags to audit primary camera rays (depth == 0) exactly once
    bool has_brdf_hit = false;

    // Shading Frame Normals
    vec3 N = unit_vector(hi.normal);
    vec3 ray_dir = unit_vector(r.direction());

    bool front_face = dot(ray_dir, N) < 0.0f;
    vec3 n = front_face ? N : -N;

    // Local probabilities for light selection
    std::vector<float> local_probs = compute_local_light_weights(hi.p, n, lights);

    // V = Vector pointing towards the camera (View)
    vec3 V = -ray_dir;

    // Fetch Base Attributes
    color base_color = tex_base_color(hi, tex_manager);
    float metallic = hi.mat.metallic;
    float roughness = std::max(hi.mat.roughness, 0.01f);

    float sheen = hi.mat.sheen;
    float clearcoat_thickness = hi.mat.clearcoat_thickness;
    float cc_roughness = std::max(hi.mat.clearcoat_roughness, 0.01f);

    if (metallic == 0.0f && roughness == 0.01f && hi.mat.spec_exponent > 0.0f)
    {
        roughness = std::max(std::sqrt(2.0f / (hi.mat.spec_exponent + 2.0f)), 0.01f);
    }

    vec3 mirror_dir = reflect(ray_dir, n);

    // --- Handling Delta Materials ---
    if (hi.mat.transparency > 1e-6f)
    {
        float n1 = front_face ? 1.0f : hi.mat.ior;
        float n2 = front_face ? hi.mat.ior : 1.0f;
        float eta_ratio = n1 / n2;
        float cos_theta = std::clamp(dot(V, n), 0.0f, 1.0f);

        float R = fresnel_schlick(cos_theta, n1, n2);
        vec3 refr_dir;
        bool can_refract = refract(ray_dir, n, eta_ratio, refr_dir);

        ray refl_ray(hi.p + mirror_dir * eps, mirror_dir);
        color col_refl = newest_rec_trace(refl_ray, bvh_tree, depth + 1, lights, tex_manager);

        if (!can_refract)
        {
            if (depth == 0)
            {
                if (col_refl.length_squared() > 1e-6f)
                    brdfHits++;
                else
                    timedOut++;
            }
            return col_refl;
        }

        ray refr_ray(hi.p + refr_dir * eps, refr_dir);
        color col_refr = newest_rec_trace(refr_ray, bvh_tree, depth + 1, lights, tex_manager);
        color final_trans = R * col_refl + (1.0f - R) * col_refr;

        if (depth == 0)
        {
            if (final_trans.length_squared() > 1e-6f)
                brdfHits++;
            else
                timedOut++;
        }

        return final_trans;
    }

    if (hi.mat.is_delta_specular || (metallic > 0.99f && roughness <= 0.01f))
    {
        color fr = (metallic > 0.99f) ? base_color : hi.mat.specular;
        ray new_ray(hi.p + mirror_dir * eps, mirror_dir);
        color child = newest_rec_trace(new_ray, bvh_tree, depth + 1, lights, tex_manager);

        if (depth == 0)
        {
            if (child.length_squared() > 1e-6f)
                brdfHits++;
            else
                timedOut++;
        }

        return child * fr;
    }

    // =========================================================================
    // LOBE BALANCING
    // =========================================================================
    float ior = (hi.mat.ior > 0.0f) ? hi.mat.ior : 1.5f;
    float root_f0 = (ior - 1.0f) / (ior + 1.0f);
    float calculated_f0 = root_f0 * root_f0;

    color dielectric_F0 = color(calculated_f0, calculated_f0, calculated_f0);
    color F0 = dielectric_F0 * (1.0f - metallic) + base_color * metallic;

    float diffuse_energy = luminance(base_color) * (1.0f - metallic);
    float specular_energy = luminance(F0);
    float total_energy = diffuse_energy + specular_energy;

    float SPEC_PROB = 0.5f;
    if (total_energy > 0.0f)
    {
        SPEC_PROB = specular_energy / total_energy;
    }
    float TOTAL_DIFF_PROB = 1.0f - SPEC_PROB;

    float active_weight = (depth == 0) ? std::clamp(hi.mat.subsurface_weight, 0.0f, 1.0f) : 0.0f;

    float SUBSURFACE_PROB = TOTAL_DIFF_PROB * active_weight;
    float DIFF_PROB = TOTAL_DIFF_PROB - SUBSURFACE_PROB;

    float xi_lobe = rand_float();

    // =========================================================================
    // PATH A: SUBSURFACE SCATTERING LOBE (BSSRDF)
    // =========================================================================
    if ((depth == 0) && SUBSURFACE_PROB > 1e-4f && xi_lobe >= SPEC_PROB && xi_lobe < (SPEC_PROB + SUBSURFACE_PROB))
    {
        color total_sss_illumination(0, 0, 0);
        int sss_samples = 10;

        color fake_lambert_albedo = base_color / M_PI;

        float scene_density_scale = 50.0f;
        float sigma = hi.mat.subsurface_scale * scene_density_scale;

        for (int s = 0; s < sss_samples; s++)
        {
            float r_dist = -std::log(std::max(1e-7f, 1.0f - rand_float())) / sigma;

            float phi = 2.0f * M_PI * rand_float();
            vec3 tangent = unit_vector(cross(n, (std::abs(n.y()) > 0.99f) ? vec3(1, 0, 0) : vec3(0, 1, 0)));
            vec3 bitangent = cross(n, tangent);

            vec3 displacement = (tangent * std::cos(phi) + bitangent * std::sin(phi)) * r_dist;

            float height_offset = r_dist * 0.5f + 1e-3f;
            point3 displaced_origin = hi.p + displacement + n * height_offset;

            vec3 inward_dir = -n;
            ray probe_ray(displaced_origin, inward_dir);

            hit_info exit_hi;

            if (!bvh_tree.hit(probe_ray, 1e-4f, FLT_MAX, exit_hi))
            {
                continue;
            }

            if (exit_hi.mesh_id != hi.mesh_id)
            {
                continue;
            }

            vec3 exit_n = unit_vector(exit_hi.normal);
            if (dot(inward_dir, exit_n) > 0.0f)
            {
                exit_n = -exit_n;
            }

            color absorption = (color(1.0f, 1.0f, 1.0f) - base_color) * sigma;
            color bssrdf_weight = color(
                                      std::exp(-absorption.x() * r_dist),
                                      std::exp(-absorption.y() * r_dist),
                                      std::exp(-absorption.z() * r_dist)) /
                                  SUBSURFACE_PROB;

            // Direct Illumination at Exit Point
            color direct_lighting(0, 0, 0);
            if (!lights.empty())
            {
                int chosen_id = 0;
                float selected_weight = 1.0f;
                const area_light &L = sample_multiple_AL(lights, local_probs, selected_weight, chosen_id);

                float pdf_area;
                vec3 light_normal;
                point3 light_p = sample_stratified(light_normal, L, pdf_area, rand_float(), rand_float());

                vec3 Lvec = light_p - exit_hi.p;
                float dist_sq = Lvec.length_squared();
                vec3 light_dir = unit_vector(Lvec);

                float exit_NdotL = std::clamp(dot(exit_n, light_dir), 0.0f, 1.0f);

                if (exit_NdotL > 0.0f && visible(exit_hi.p, light_p, bvh_tree))
                {
                    float cos_l = std::clamp(dot(light_normal, -light_dir), 0.0f, 1.0f);
                    float pdf_light = (dist_sq / (cos_l * L.area)) * selected_weight;

                    if (pdf_light > 1e-5f)
                    {
                        direct_lighting = (fake_lambert_albedo * exit_NdotL * L.emission) / pdf_light;
                    }
                }
            }

            // Indirect Path Extension
            color indirect_lighting(0, 0, 0);
            vec3 outward_dir = sample_cosine_hemisphere(exit_n);
            float exit_NdotL = std::clamp(dot(outward_dir, exit_n), 0.0f, 1.0f);

            if (exit_NdotL > 0.0f)
            {
                point3 safe_origin = exit_hi.p + exit_n * 1e-4f + outward_dir * 1e-4f;
                ray bounce_ray(safe_origin, outward_dir);
                color incoming = newest_rec_trace(bounce_ray, bvh_tree, depth + 1, lights, tex_manager);

                indirect_lighting = incoming * base_color;
            }

            total_sss_illumination += (direct_lighting + indirect_lighting) * bssrdf_weight;
        }

        color sss_res = total_sss_illumination / static_cast<float>(sss_samples);
        if (depth == 0)
        {
            if (sss_res.length_squared() > 1e-6f)
                brdfHits++;
            else
                timedOut++;
        }
        return sss_res;
    }

    int light_samples = 1;
    int brdf_samples = 1;

    // --------------------------------------------------
    // 1) Next Event Estimation (Direct Light Sampling)
    // --------------------------------------------------
    color sum_light(0, 0, 0);
    if (!lights.empty())
    {
        int chosen_id = 0;
        float selected_weight = 1.0f;
        const area_light &L = sample_multiple_AL(lights, local_probs, selected_weight, chosen_id);

        int grid_dim = static_cast<int>(std::sqrt(light_samples));
        std::vector<light_sample> pts;
        pts.reserve(grid_dim * grid_dim);

        for (int y = 0; y < grid_dim; y++)
        {
            for (int x = 0; x < grid_dim; x++)
            {
                float r1 = (static_cast<float>(x) + rand_float()) / static_cast<float>(grid_dim);
                float r2 = (static_cast<float>(y) + rand_float()) / static_cast<float>(grid_dim);
                light_sample sample_data;
                sample_data.p = sample_stratified(sample_data.normal, L, sample_data.pdf_area, r1, r2);
                pts.push_back(sample_data);
            }
        }

        for (const light_sample &ls : pts)
        {
            vec3 Lvec = ls.p - hi.p;
            float dist_squared = Lvec.length_squared();
            float dist = std::sqrt(dist_squared);
            vec3 light_dir = Lvec / dist;

            vec3 light_normal = ls.normal;
            float NdotL = std::clamp(dot(n, light_dir), 0.0f, 1.0f);
            float NdotV = std::clamp(dot(n, V), 0.0f, 1.0f);
            float cos_l = std::clamp(dot(light_normal, -light_dir), 0.0f, 1.0f);

            if (cos_l <= 1e-5f)
                continue;

            if (NdotL <= 1e-5f || NdotV <= 1e-5f || cos_l <= 1e-5f)
                continue;

            if (!visible(hi.p + light_dir * eps, ls.p, bvh_tree))
                continue;

            float pdf_light = dist_squared / (cos_l * L.area);
            float pdf_light_new = (pdf_light * selected_weight);
            if (pdf_light_new <= 1e-5f)
                continue;

            vec3 H = unit_vector(V + light_dir);
            float NdotH = std::clamp(dot(n, H), 0.0f, 1.0f);
            float LdotH = std::clamp(dot(light_dir, H), 0.0f, 1.0f);
            float VdotH = std::clamp(dot(V, H), 0.0f, 1.0f);

            // Specular BRDF Evaluation
            float D = ndf_ggx(NdotH, roughness);

            float alpha2 = (roughness * roughness) * (roughness * roughness);
            float denom_V = NdotV + std::sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV);
            float denom_L = NdotL + std::sqrt(alpha2 + (1.0f - alpha2) * NdotL * NdotL);

            float fresnel_base = std::clamp(1.0f - VdotH, 0.0f, 1.0f);
            color F = F0 + (color(1.0f, 1.0f, 1.0f) - F0) * std::pow(fresnel_base, 5.0f);

            color f_spec = (D * F) / (2.0f * denom_V * denom_L);
            color f_diff = evaluate_disney_diffuse(base_color, roughness, hi.mat.subsurface_weight, NdotL, NdotV, NdotH, LdotH);

            color fr = f_diff * (1.0f - metallic) + f_spec;

            if (sheen > 0.0f)
            {
                fr += (sheen * evaluate_sheen(base_color, VdotH));
            }

            if (clearcoat_thickness > 0.0f)
            {
                float microfacet_cc = evaluate_clearcoat(NdotH, cc_roughness, NdotL, NdotV, VdotH);
                float F_cc = 0.04f + (1.0f - 0.04f) * std::pow(fresnel_base, 5.0f);
                float f_cc = microfacet_cc * F_cc;

                fr = fr * (1.0f - clearcoat_thickness * F_cc) + (color(1.0f, 1.0f, 1.0f) * (clearcoat_thickness * f_cc));
            }

            float pdf_spec = 0.0f;
            if (VdotH > 1e-4f)
            {
                pdf_spec = (D * NdotH) / (4.0f * VdotH);
            }

            float pdf_diffuse_pure = NdotL / M_PI;
            float pdf_brdf = SPEC_PROB * pdf_spec + DIFF_PROB * pdf_diffuse_pure;
            if (pdf_brdf <= 1e-5f)
                continue;

            if (!std::isfinite(fr.x()) || !std::isfinite(pdf_brdf))
                continue;

            float w = power_heuristic(pdf_light_new, pdf_brdf);
            sum_light += (w * fr * NdotL * L.emission / pdf_light_new);
        }

        if (grid_dim * grid_dim > 1)
            sum_light /= float(grid_dim * grid_dim);
    }

    // --------------------------------------------------
    // 2) BRDF Sampling Loop (Indirect Path Extension)
    // --------------------------------------------------
    color sum_brdf_full(0, 0, 0);

    for (int i = 0; i < brdf_samples; i++)
    {
        float relative_spec_prob = SPEC_PROB / (SPEC_PROB + DIFF_PROB);
        bool choose_spec = rand_float() < relative_spec_prob;
        vec3 light_dir;
        float pdf_spec = 0.0f;

        if (choose_spec)
        {
            float pdf_H_out = 0.0f;
            vec3 H = sample_ggx_lobe(n, roughness, rand_float(), rand_float(), pdf_H_out);
            light_dir = reflect(-V, H);

            float VdotH = std::clamp(dot(V, H), 0.0f, 1.0f);
            if (VdotH > 1e-4f)
            {
                pdf_spec = pdf_H_out / (4.0f * VdotH);
            }
        }
        else
        {
            light_dir = sample_cosine_hemisphere(n);
            vec3 H = unit_vector(V + light_dir);

            float NdotH = std::clamp(dot(n, H), 0.0f, 1.0f);
            float VdotH = dot(V, H);

            if (VdotH <= 1e-5f)
            {
                pdf_spec = 0.0f;
            }
            else
            {
                float alpha2 = (roughness * roughness) * (roughness * roughness);
                float denom = (NdotH * NdotH * (alpha2 - 1.0f) + 1.0f);
                float pdf_H_eval = (alpha2 * NdotH) / (M_PI * denom * denom);

                pdf_spec = pdf_H_eval / (4.0f * VdotH);
            }
        }

        float NdotL = std::clamp(dot(light_dir, n), 0.0f, 1.0f);
        float NdotV = std::clamp(dot(V, n), 0.0f, 1.0f);
        if (NdotL <= 1e-5f || NdotV <= 1e-5f)
            continue;

        float pdf_diff = NdotL / M_PI;
        float pdf_brdf = SPEC_PROB * pdf_spec + DIFF_PROB * pdf_diff;

        if (pdf_brdf <= 1e-5f)
            continue;

        if (!std::isfinite(pdf_brdf) || pdf_brdf > 1e6f)
        {
            pdf_brdf = 1e6f;
        }

        vec3 V_plus_L = V + light_dir;
        vec3 H = (V_plus_L.length_squared() > 1e-6f) ? unit_vector(V_plus_L) : n;

        float NdotH = std::clamp(dot(n, H), 0.0f, 1.0f);
        float LdotH = std::clamp(dot(light_dir, H), 0.0f, 1.0f);
        float VdotH = std::clamp(dot(V, H), 0.0f, 1.0f);

        float D = ndf_ggx(NdotH, roughness);
        float alpha2 = (roughness * roughness) * (roughness * roughness);
        float denom_V = NdotV + std::sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV);
        float denom_L = NdotL + std::sqrt(alpha2 + (1.0f - alpha2) * NdotL * NdotL);
        float fresnel_base = std::clamp(1.0f - VdotH, 0.0f, 1.0f);

        color F = F0 + (color(1.0f, 1.0f, 1.0f) - F0) * std::pow(fresnel_base, 5.0f);
        color f_spec = (D * F) / (2.0f * denom_V * denom_L);
        color f_diff = evaluate_disney_diffuse(base_color, roughness, hi.mat.subsurface_weight, NdotL, NdotV, NdotH, LdotH);
        color fr = f_diff * (1.0f - metallic) + f_spec;

        if (sheen > 0.0f)
        {
            fr += (sheen * evaluate_sheen(base_color, VdotH));
        }

        if (clearcoat_thickness > 0.0f)
        {
            float microfacet_cc = evaluate_clearcoat(NdotH, cc_roughness, NdotL, NdotV, VdotH);
            float F_cc = 0.04f + (1.0f - 0.04f) * std::pow(fresnel_base, 5.0f);
            float f_cc = microfacet_cc * F_cc;

            fr = fr * (1.0f - clearcoat_thickness * F_cc) + (color(1.0f, 1.0f, 1.0f) * (clearcoat_thickness * f_cc));
        }

        ray new_ray(hi.p + light_dir * eps, light_dir);

        hit_info nh;
        if (!bvh_tree.hit(new_ray, eps, FLT_MAX, nh))
        {
            continue;
        }

        if (nh.mat.is_emissive())
        {
            const area_light &L = lights[nh.mat.light_id];

            float dist = (nh.p - hi.p).length();

            vec3 light_normal = L.normal;
            float cos_l = std::clamp(dot(light_normal, -light_dir), 0.0f, 1.0f);
            if (cos_l <= 1e-5f)
                continue;

            float pdf_light = (dist * dist) / (cos_l * L.area);
            float pdf_light_new = pdf_light * local_probs[nh.mat.light_id];
            float w = power_heuristic(pdf_brdf, pdf_light_new);

            color direct_contrib = w * nh.mat.emission * fr * NdotL / pdf_brdf;
            sum_brdf_full += direct_contrib;

            if (direct_contrib.length_squared() > 1e-6f)
            {
                has_brdf_hit = true;
            }
        }
        else
        {
            color child = newest_rec_trace(new_ray, bvh_tree, depth + 1, lights, tex_manager);
            color contrib = fr * NdotL / pdf_brdf;

            color total_bounce = child * contrib;
            sum_brdf_full += total_bounce;

            if (total_bounce.length_squared() > 1e-6f)
            {
                has_brdf_hit = true;
            }
        }
    }

    if (brdf_samples > 1)
        sum_brdf_full /= float(brdf_samples);

    // =========================================================================
    // AUDIT LOGIC FOR PRIMARY CAMERA RAYS (depth == 0)
    // =========================================================================
    if (depth == 0)
    {
        bool has_nee_hit = (sum_light.length_squared() > 1e-6f);

        if (has_brdf_hit)
        {
            // Path gathered illumination through BRDF indirect sample rays
            brdfHits++;
        }
        else if (has_nee_hit)
        {
            // Path gathered illumination ONLY via Next Event Estimation (direct light sampling)
            neeHits++;
        }
        else
        {
            // Ray hit scene surface, but both NEE and BRDF bounce failed to find light
            timedOut++;
        }
    }

    return sum_light + sum_brdf_full;
}

bool camera::is_nan_color(const color &c)
{
    return std::isnan(c.x()) || std::isnan(c.y()) || std::isnan(c.z());
}

void camera::debug_nan_state(const std::string &location, float NdotL, float NdotV, float NdotH, float VdotH, float LdotH, float roughness, float D, float G, float pdf_brdf)
{
    std::cout << "\n--- [NAN DETECTED AT: " << location << "] ---" << std::endl;
    std::cout << "Roughness: " << roughness << std::endl;
    std::cout << "NdotL: " << NdotL << " | NdotV: " << NdotV << std::endl;
    std::cout << "NdotH: " << NdotH << " | VdotH: " << VdotH << " | LdotH: " << LdotH << std::endl;
    std::cout << "NDF (D): " << D << " | Geometry (G): " << G << std::endl;
    std::cout << "Combined PDF: " << pdf_brdf << std::endl;
}

color camera::guided_rec_trace(
    const ray &r,
    const bvh &bvh_tree,
    int depth,
    std::vector<area_light> &lights,
    TextureManager &tex_manager,
    VoxelGrid &grid,
    bool use_path_guiding,
    std::vector<point3> path_history)
{
    const float eps = 1e-4f;
    if (depth >= MAX_DEPTH)
    {
        // if (depth == 0)
        //   timedOut++;

        // --- ADD THIS TO CAPTURE PATHS THAT ENDED WITHOUT HITTING A LIGHT ---
        /* bool hits_target_voxel = false;
        for (const auto &p : path_history)
        {
            if (grid.get_voxel_index(p) == grid.get_voxel_index(test_point))
            {
                hits_target_voxel = true;
                break;
            }
        }
        if (hits_target_voxel)
        {
            VisualRayPath full_path;
            full_path.vertices = path_history;
#pragma omp critical
            {
                if (g_heatmap_full_paths.size() < 1000)
                {
                    g_heatmap_full_paths.push_back(full_path);
                }
            }
        } */
        // ---------------------------------------------------------------------
        return color(0, 0, 0);
    }

    // 1. Scene Intersection Check
    hit_info hi;
    if (!bvh_tree.hit(r, eps, FLT_MAX, hi))
    {
        if (depth == 0)
            misses++;
        return color(0, 0, 0); // color(0.2, 0.5, 0.8); // Sky color
    }

    // 2. Emissive Check (Direct Hit)
    if (hi.mat.is_emissive())
    {
        if (depth == 0)
            directHits++;

        return hi.mat.emission;
    }

    // 3. Normal Setup and Geometry Alignment
    vec3 N = unit_vector(hi.normal);
    vec3 dir_in = unit_vector(r.direction());
    bool front_face = dot(dir_in, N) < 0.0f;
    vec3 n = front_face ? N : -N;

    vec3 wi = -dir_in;
    vec3 refl_dir = reflect(dir_in, n);
    float s = hi.mat.spec_exponent;

    // Precompute spatial light probabilities for multi-light selection
    std::vector<float> local_probs = compute_local_light_weights(hi.p, n, lights);

    // ====================================================
    // SPECULAR INFRASTRUCTURE (GLASS & DIELECTRICS)
    // ====================================================

    // === PERFECT GLASS / DIELECTRIC ===
    /* if (hi.mat.transparency > 1e-6f)
    {
        float n1 = front_face ? 1.0f : hi.mat.ior;
        float n2 = front_face ? hi.mat.ior : 1.0f;
        float eta_ratio = n1 / n2;
        float cos_theta = std::min(dot(-dir_in, n), 1.0f);
        if (cos_theta <= 0.0f)
            return color(0, 0, 0);

        float R = fresnel_schlick(cos_theta, n1, n2);
        vec3 refr_dir, spec_refl_dir;
        spec_refl_dir = reflect(dir_in, n);

        bool can_refract = refract(dir_in, n, eta_ratio, refr_dir);

        ray refl_ray(hi.p + spec_refl_dir * eps, spec_refl_dir);
        // Path guiding is disabled (false) down specular channels to avoid variance spikes
        color col_refl = guided_rec_trace(refl_ray, bvh_tree, depth + 1, lights, tex_manager, grid, false);

        if (!can_refract)
            return col_refl; // Total Internal Reflection

        ray refr_ray(hi.p + refr_dir * eps, refr_dir);
        color col_refr = guided_rec_trace(refr_ray, bvh_tree, depth + 1, lights, tex_manager, grid, false);

        return R * col_refl + (1.0f - R) * col_refr;
    } */

    // === PERFECT MIRROR DELTA LAYER ===
    /* if (hi.mat.is_delta_specular)
     {
         color fr = hi.mat.specular;
         ray new_ray(hi.p + refl_dir * eps, refl_dir);
         return guided_rec_trace(new_ray, bvh_tree, depth + 1, lights, tex_manager, grid, false) * fr;
     } */

    // ====================================================
    // MATERIAL PROBABILITY & DYNAMIC FRESNEL SETUP
    // ====================================================
    int light_samples = 1;
    int brdf_samples = 1;

    float SPEC_PROB = luminance(hi.mat.specular);
    float DIFF_PROB = 1.0f - SPEC_PROB;

    /* bool is_shiny_dielectric = (luminance(hi.mat.diffuse) > 0.0f) && (SPEC_PROB > 0.0f) && (!hi.mat.is_delta_specular);
    if (is_shiny_dielectric)
    {
        float cos_view = dot(-dir_in, n);
        float R0 = std::max(0.04f, SPEC_PROB);
        float fresnel_factor = R0 + (1.0f - R0) * std::pow(1.0f - cos_view, 5.0f);

        SPEC_PROB = fresnel_factor;
        DIFF_PROB = 1.0f - SPEC_PROB;
    } */

    // ====================================================
    // UPDATE: Append current surface hit point to history
    // ====================================================
    std::vector<point3> current_path_history = path_history;
    current_path_history.push_back(hi.p);

    float alpha = 1.0f; // exploration vs exploitation
    // ====================================================
    // 2) NEXT EVENT ESTIMATION (DIRECT LIGHTING LOOP)
    // ====================================================
    color sum_light(0, 0, 0);
    if (!lights.empty())
    {
        int chosen_id = 0;
        float selected_weight = 1.0f;
        const area_light &L = sample_multiple_AL(lights, local_probs, selected_weight, chosen_id);

        int grid_dim = static_cast<int>(std::sqrt(light_samples));
        int total_actual_samples = grid_dim * grid_dim;
        std::vector<light_sample> pts;
        pts.reserve(total_actual_samples);

        for (int y = 0; y < grid_dim; y++)
        {
            for (int x = 0; x < grid_dim; x++)
            {
                float r1 = (static_cast<float>(x) + rand_float()) / static_cast<float>(grid_dim);
                float r2 = (static_cast<float>(y) + rand_float()) / static_cast<float>(grid_dim);

                light_sample sample_data;
                sample_data.p = sample_stratified(sample_data.normal, L, sample_data.pdf_area, r1, r2);
                pts.push_back(sample_data);
            }
        }

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

            if (!visible(hi.p + wo * eps, ls.p, bvh_tree))
                continue;

            float pdf_light = (dist_squared / cos_l) * ls.pdf_area;
            float pdf_light_new = pdf_light * selected_weight;

            // 1. Properly normalize the diffuse component by PI
            color f_diffuse = tex_base_color(hi, tex_manager) / M_PI;
            // 2. Evaluate the specular BRDF component
            color f_specular = brdf_specular(hi.mat, s, refl_dir, wo);
            // 3. Balance the components based on your lobe configuration
            color fr = f_diffuse * DIFF_PROB + f_specular * SPEC_PROB;
            // color fr = tex_base_color(hi, tex_manager) / M_PI + brdf_specular(hi.mat, s, refl_dir, wo); // TODO

            float phong_pdf = (s + 2.0f) * std::pow(std::max(dot(refl_dir, wo), 0.0f), s) / (2.0f * M_PI);
            float pdf_brdf_pure = SPEC_PROB * phong_pdf + DIFF_PROB * (cos_s / M_PI);

            float pdf_final_eval = pdf_brdf_pure;
            if (use_path_guiding)
            {
                float pdf_guided = grid.get_guided_pdf(hi.p, n, wo); // interpolation? (1)
                pdf_final_eval = alpha * pdf_guided + (1.0f - alpha) * pdf_brdf_pure;
            }

            float w = power_heuristic(pdf_light_new, pdf_final_eval);
            color sample_nee_contrib = (w * fr * cos_s * L.emission / pdf_light_new);
            sum_light += sample_nee_contrib;

            // --- NEE PATH DIAGNOSTIC LOGGING (DISABLED) ---
            /* if (depth == 0 && sample_nee_contrib.length() > 0.0f && test_point.length() > 1e-4f)
            {
                std::vector<point3> full_path_verts = current_path_history;
                full_path_verts.push_back(ls.p);

                bool hits_target_voxel = false;
                for (const auto &p : full_path_verts)
                {
                    if (grid.get_voxel_index(p) == grid.get_voxel_index(test_point))
                    {
                        hits_target_voxel = true;
                        break;
                    }
                }

                if (hits_target_voxel)
                {
                    VisualRayPath full_path;
                    full_path.vertices = full_path_verts;
                    full_path.vertex_radiance.resize(full_path_verts.size(), color(0, 0, 0));
                    full_path.vertex_radiance[full_path_verts.size() - 1] = L.emission;
                    full_path.vertex_radiance[full_path_verts.size() - 2] = sample_nee_contrib;

                    #pragma omp critical
                    {
                        if (g_heatmap_full_paths.size() < 1000)
                            g_heatmap_full_paths.push_back(full_path);
                    }
                }
            }
            */
        }
        if (light_samples > 1)
            sum_light /= float(light_samples);
    }

    // ====================================================
    // 3) BRDF & PATH GUIDED SAMPLING LOOP
    // ====================================================
    color sum_brdf_full(0, 0, 0);
    float rr_weight = 1.0f;

    bool sample_was_guided = false;
    for (int i = 0; i < brdf_samples; i++)
    {
        vec3 wo;
        float pdf_brdf = 0.0f;

        if (use_path_guiding)
        {
            float pdf_guided = 0.0f;

            if (rand_float() < alpha)
            {
                // Ensure sample_guided_direction internally uses the same spatial interpolation!
                wo = grid.sample_guided_direction(hi.p, n, pdf_guided); // interpolation? (2)
                sample_was_guided = true;
                // std::cout << "sample guided PDF: " << pdf_guided << " , get guided PDF: " << grid.get_guided_pdf(hi.p, n, wo) << std::endl;
            }
            else
            {
                if (rand_float() < SPEC_PROB)
                    wo = sample_phong_lobe(refl_dir, n, s, pdf_guided);
                else
                    wo = sample_cosine_hemisphere(n);

                pdf_guided = grid.get_guided_pdf(hi.p, n, wo); // interpolation? (3)
            }

            float cos_n = std::max(0.0f, dot(wo, n));
            float cos_r = std::max(0.0f, dot(refl_dir, wo));
            float pdf_diff_pure = cos_n / M_PI;
            float pdf_spec_pure = (s + 2.0f) * std::pow(cos_r, s) / (2.0f * M_PI);
            float pdf_material_total = SPEC_PROB * pdf_spec_pure + DIFF_PROB * pdf_diff_pure;

            pdf_brdf = alpha * pdf_guided + (1.0f - alpha) * pdf_material_total;
        }
        else
        {
            if (rand_float() < SPEC_PROB)
                wo = sample_phong_lobe(refl_dir, n, s, pdf_brdf);
            else
                wo = sample_cosine_hemisphere(n);

            float cos_n = std::max(0.0f, dot(wo, n));
            float cos_r = std::max(0.0f, dot(refl_dir, wo));
            float pdf_diff_pure = cos_n / M_PI;
            float pdf_spec_pure = (s + 2.0f) * std::pow(cos_r, s) / (2.0f * M_PI);

            pdf_brdf = SPEC_PROB * pdf_spec_pure + DIFF_PROB * pdf_diff_pure;
        }

        float cos_theta = std::max(0.0f, dot(wo, n));
        if (pdf_brdf <= 0.0f || cos_theta <= 0.0f)
            continue;

        // 1. Properly normalize the diffuse component by PI
        color f_diffuse = tex_base_color(hi, tex_manager) / M_PI;
        // 2. Evaluate the specular BRDF component
        color f_specular = brdf_specular(hi.mat, s, refl_dir, wo);
        // 3. Balance the components based on your lobe configuration
        color fr = f_diffuse * DIFF_PROB + f_specular * SPEC_PROB;
        // color fr = tex_base_color(hi, tex_manager) + brdf_specular(hi.mat, s, refl_dir, wo);

        ray new_ray(hi.p + wo * eps, wo);
        hit_info nh;
        if (!bvh_tree.hit(new_ray, eps, FLT_MAX, nh))
        {
            // if (depth == 0)
            //   misses++;
            continue;
        }

        // --- DIAGNOSTIC TEST POINT LOCK-ON ---
        if (depth == 0 && test_point.length() <= 1e-4f)
        {
            if (!nh.mat.is_emissive())
            {
                test_point = nh.p;
                test_point_normal = nh.normal;
                std::cout << "--> Diagnostic Lock: test_point pinned at " << test_point << std::endl;
            }
        }

        // Processing Branch A: Secondary ray hits an emissive surface directly
        color child(0, 0, 0);
        if (nh.mat.is_emissive())
        {
            /* if (depth == 0)
            {
                indirectHits++;
                if (sample_was_guided)
                    guidedHits++;
            } */

            const area_light &L = lights[nh.mat.light_id];
            float dist = (nh.p - hi.p).length();
            float cos_l = std::max(0.0f, dot(L.normal, -wo));
            if (cos_l <= 0.0f)
                continue;

            float pdf_light = (dist * dist / cos_l) / L.area;
            float pdf_light_new = pdf_light * local_probs[nh.mat.light_id];

            float w = power_heuristic(pdf_brdf, pdf_light_new);
            color sample_brdf_contrib = w * nh.mat.emission * fr * cos_theta / pdf_brdf; // w*
            sum_brdf_full += sample_brdf_contrib;

            // --- BRANCH A PATH DIAGNOSTIC LOGGING ---
            // Guarded with depth == 0 to keep tracing layers clean and eliminate coordinate clones
            if (depth == 0 && sample_brdf_contrib.length() > 0.0f && test_point.length() > 1e-4f)
            {
                std::vector<point3> full_path_verts = current_path_history;
                full_path_verts.push_back(nh.p);

                bool hits_target_voxel = false;
                for (const auto &p : full_path_verts)
                {
                    if (grid.get_voxel_index(p) == grid.get_voxel_index(test_point))
                    {
                        hits_target_voxel = true;
                        break;
                    }
                }

                if (hits_target_voxel)
                {
                    VisualRayPath full_path;
                    full_path.vertices = full_path_verts;
                    full_path.vertex_radiance.resize(full_path_verts.size(), color(0, 0, 0));

                    // Light geometry node gets raw emission value
                    full_path.vertex_radiance[full_path_verts.size() - 1] = nh.mat.emission;
                    // Current vertex position (hi.p) gets the evaluated BRDF throughput
                    full_path.vertex_radiance[full_path_verts.size() - 2] = sample_brdf_contrib;

#pragma omp critical
                    {
                        if (g_heatmap_full_paths.size() < 1000)
                        {
                            g_heatmap_full_paths.push_back(full_path);
                        }
                    }
                }
            }
        }
        // Processing Branch B: Secondary ray bounces deeper into scene graph
        else
        {
            child = guided_rec_trace(new_ray, bvh_tree, depth + 1, lights, tex_manager, grid, use_path_guiding, current_path_history);
            color contrib = fr * cos_theta / pdf_brdf;
            color sample_indirect_contrib = rr_weight * child * contrib;
            sum_brdf_full += sample_indirect_contrib;

            /* if (depth == 0)
            {
                if (child.length() <= 0)
                    timedOut++;
                else
                {
                    indirectHits++;
                    if (sample_was_guided)
                        guidedHits++;
                }
            } */

            // --- BRANCH B PATH DIAGNOSTIC LOGGING ---
            // Guarded with depth == 0 to keep tracing layers clean and eliminate coordinate clones
            if (depth == 0 && sample_indirect_contrib.length() > 0.0f && test_point.length() > 1e-4f)
            {
                std::vector<point3> full_path_verts = current_path_history;
                full_path_verts.push_back(nh.p);

                bool hits_target_voxel = false;
                for (const auto &p : full_path_verts)
                {
                    if (grid.get_voxel_index(p) == grid.get_voxel_index(test_point))
                    {
                        hits_target_voxel = true;
                        break;
                    }
                }

                if (hits_target_voxel)
                {
                    VisualRayPath full_path;
                    full_path.vertices = full_path_verts;
                    full_path.vertex_radiance.resize(full_path_verts.size(), color(0, 0, 0));

                    // At the secondary bounce position (nh.p), store the incoming light value
                    full_path.vertex_radiance[full_path_verts.size() - 1] = child;
                    // At our current position (hi.p), store the modulated output moving toward camera
                    full_path.vertex_radiance[full_path_verts.size() - 2] = sample_indirect_contrib;

#pragma omp critical
                    {
                        if (g_heatmap_full_paths.size() < 1000)
                        {
                            g_heatmap_full_paths.push_back(full_path);
                        }
                    }
                }
            }
        }

        // --- CHOOSE WHAT TO RECORD ---
        color incoming_radiance(0, 0, 0);
        if (nh.mat.is_emissive())
        {
            // Record the raw emission of the light hit along direction wo
            incoming_radiance = nh.mat.emission;
        }
        else
        {
            // Record the raw incoming indirect radiance returned from the child path
            incoming_radiance = child;
        }

        if (grid.is_training && incoming_radiance.length() > 0.0f)
        {
            grid.record_radiance(hi.p, wo, incoming_radiance); // TODO multithreading race conditions?
        }
    }

    if (brdf_samples > 1)
        sum_brdf_full /= float(brdf_samples);

    color final_pixel_color = (sum_light + sum_brdf_full);

    // =========================================================================
    // CENTRALIZED ROOT ACCOUNTING (Strictly captures 1 outcome per pixel)
    // =========================================================================
    if (depth == 0)
    {
        // 1. Did the ray completely fail to find any light (from NEE or Bounces)?
        if (final_pixel_color.length() <= 1e-5f)
        {
            timedOut++;
        }
        else
        {
            if (sum_brdf_full.length() > 0.0f || sum_light.length() > 0.0f)
                indirectHits++;

            if (sum_brdf_full.length() == 0.0f)
                neeHits++;
            else
                brdfHits++;
            // Optional: If you want to track your guided sub-metrics separately:
            if (sample_was_guided)
                guidedHits++;
        }
    }

    return final_pixel_color;
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

color camera::oldest_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights)
{
    const float eps = 1e-6f; //-6
    if (depth >= MAX_DEPTH)
        return color(0, 0, 0);

    // intersect
    hit_info hi;
    if (!bvh_tree.hit(r, eps, FLT_MAX, hi))
        return color(0, 0, 0); // or return sky
    // emitter
    if (hi.mat.is_emissive())
        return hi.mat.emission;

    // geometric & shading normals
    vec3 N = unit_vector(hi.normal);          // geometric normal (possibly not oriented)
    vec3 dir_in = unit_vector(r.direction()); // incoming ray (points toward surface)

    bool front_face = dot(dir_in, N) < 0.0f;
    vec3 n = front_face ? N : -N; // oriented shading normal (use this everywhere) ?

    vec3 wi = -dir_in;
    vec3 refl_dir = reflect(dir_in, n);

    const float p_survive = 0.7f; // RR

    // glossy specular lobe
    float s = phong_exponent_from_smoothness(hi.mat.smoothness);
    // float s = hi.mat.spec_exponent;

    // local probabilities for light selection
    std::vector<float> local_probs = compute_local_light_weights(hi.p, n, lights);

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
        /* if (can_refract)
        {
            refr_dir = unit_vector(refr_dir);
        } */

        // Reflection ray
        ray refl_ray(hi.p + refl_dir * eps, refl_dir);
        color col_refl = oldest_rec_trace(refl_ray, bvh_tree, depth + 1, lights);

        if (!can_refract)
            return col_refl; // Total internal reflection

        // Refraction ray
        ray refr_ray(hi.p + refr_dir * eps, refr_dir);

        color col_refr = oldest_rec_trace(refr_ray, bvh_tree, depth + 1, lights); // TODO colored transparency (attenuation *)

        // Mix reflection & refraction
        return R * col_refl + (1.0f - R) * col_refr;
    }

    // --------------------------------------------------
    // Delta specular (perfect mirror)
    // --------------------------------------------------
    if (hi.mat.is_delta_specular)
    {

        color fr = hi.mat.specular; // mirror color (F0)

        ray new_ray(hi.p + refl_dir * eps, refl_dir);

        // delta: no MIS, no NEE, no cosine
        return oldest_rec_trace(new_ray, bvh_tree, depth + 1, lights) * fr;
    }

    int light_samples = 1;
    int brdf_samples = 1;

    float SPEC_PROB = hi.mat.specular_prob;
    float DIFF_PROB = 1.0f - SPEC_PROB;

    TraceResult tr;
    // -------------------------
    // 2) NEE: direct lighting
    // -------------------------

    color sum_light(0, 0, 0);
    if (!lights.empty())
    {
        const area_light &L = lights[0];
        // int chosen_id = 0;
        // float selected_weight = 1.0f;
        // const area_light &L = sample_multiple_AL(lights, local_probs, selected_weight, chosen_id); // HERE

        const float area = std::max(L.area, 1e-8f);
        auto pts = stratified_sample_light(L, light_samples);

        for (const point3 &lp : pts)
        {
            vec3 Lvec = lp - hi.p;
            float dist = Lvec.length();
            vec3 wo = Lvec / dist; // unit_vector()

            float cos_s = dot(n, wo); //-wo
            float cos_l = dot(L.normal, -wo);

            if (cos_s <= 0 || cos_l <= 0)
                continue;

            cos_s = std::min(cos_s, 1.0f);
            cos_l = std::min(cos_l, 1.0f);

            // visibility
            if (!visible(hi.p + wo * eps, lp, bvh_tree))
                continue;

            // light PDF in solid angle domain
            float pdf_light = (dist * dist / cos_l) / area;

            // pdf_light *= L.selection_weight; // HERE

            // --- FULL BRDF evaluation (same as NEE)
            color fr = brdf_diffuse(hi.mat) + brdf_specular(hi.mat, s, refl_dir, wo);
            float phong_pdf = (s + 2.0f) * std::pow(std::max(dot(refl_dir, wo), 0.0f), s) / (2.0f * M_PI); // (s + 2.0f) * std::pow(std::max(dot(refl_dir, wo), 0.0f), s) / (2.0f * M_PI)

            float pdf_brdf = SPEC_PROB * phong_pdf + DIFF_PROB * (cos_s / M_PI); // NO MIS

            // TODO MIS

            float w = power_heuristic(pdf_light, pdf_brdf);       // or balance
            sum_light += w * fr * cos_s * L.emission / pdf_light; // w *
            std::cout << "w: " << w << " fr: " << fr << " cos: " << cos_s << " emission: " << L.emission << " pdf: " << pdf_light << std::endl;
        }

        if (light_samples > 1)
            sum_light /= float(light_samples * light_samples);
    }

    // RR 1
    float rr_weight;
    if (depth > MIN_DEPTH)
    {
        if (rand_float() > p_survive)
            return color(0, 0, 0);
        rr_weight = 1.0f / p_survive;
    }
    else
    {
        rr_weight = 1.0f;
    }

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

        color fr = brdf_diffuse(hi.mat) + brdf_specular(hi.mat, s, refl_dir, wo);

        ray new_ray(hi.p + wo * eps, wo);
        hit_info nh;
        if (!bvh_tree.hit(new_ray, eps, FLT_MAX, nh))
            continue;

        if (nh.mat.is_emissive())
        {
            // continue; // trivial MIS

            // existing direct-light MIS part (keep as is, but using pdf_brdf)
            const area_light &L = lights[0];
            // const area_light &L = lights[nh.mat.light_id];

            float dist = (nh.p - hi.p).length();
            float cos_l = std::max(0.0f, dot(L.normal, -wo));
            if (cos_l <= 0.0f)
                continue;

            float pdf_light = (dist * dist / cos_l) / L.area;
            float w = power_heuristic(pdf_brdf, pdf_light);

            sum_brdf_full += w * nh.mat.emission * fr * cos_theta / pdf_brdf; // w *
        }
        else
        {
            color child = oldest_rec_trace(new_ray, bvh_tree, depth + 1, lights);
            color contrib = fr * cos_theta / pdf_brdf;
            sum_brdf_full += rr_weight * child * contrib; // RR 2 rr_weight *
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

// Rendering
void camera::render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree, VoxelGrid &guiding_grid, TextureManager &tex_manager)
{
    initialize();
    std::vector<color> framebuffer(image_width * image_height);

    // ErrorAnalyzer analyzer(image_width, image_height);
    int spp = 9;
    int N = int(std::sqrt(spp));
    bool guiding = false;

    // #pragma omp parallel for schedule(dynamic) // TODO threads
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
                    // color sample = oldest_rec_trace(r, bvh_tree, 0, lights);
                    color sample = newest_rec_trace(r, bvh_tree, 0, lights, tex_manager); // define recusion depth!, lights, throughput: color(1, 1, 1)
                    //  color sample = guided_rec_trace(r, bvh_tree, 0, lights, guiding_grid, guiding);

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

void camera::new_render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree, VoxelGrid &guiding_grid, TextureManager &tex_manager)
{
    initialize();

    auto training_start = std::chrono::system_clock::now(); // training time
    // =========================================================================
    // MULTI-PASS PROGRESSIVE TRAINING LOOP
    // =========================================================================
    int num_training_iterations = 1;
    int samples_per_iteration = 9;
    int N_train = static_cast<int>(std::sqrt(samples_per_iteration));

    for (int iter = 0; iter < num_training_iterations; iter++)
    {
        std::cout << "--> Starting Training Iteration " << (iter + 1)
                  << "/" << num_training_iterations << "..." << std::endl;

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

                        bool guiding_enabled = (iter > 0);

                        // SEED history with the camera center
                        guided_rec_trace(r, bvh_tree, 0, lights, tex_manager, guiding_grid, guiding_enabled, {camera_center});
                    }
                }
            }
        }

        /* for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                std::cout << "PDF (before) at row " << i << " and colomn " << j << " : " << guiding_grid.get_voxel(test_point).guiding_pdf[i][j] << std::endl;
            }
        } */

        std::cout << "    Baking distributions for iteration " << (iter + 1) << "..." << std::endl;
        guiding_grid.finalize_and_build_distributions();

        /* for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                std::cout << "PDF (after) at row " << i << " and colomn " << j << " : " << guiding_grid.get_voxel(test_point).guiding_pdf[i][j] << std::endl;
            }
        } */

        std::string heatmap_name = "voxel_heatmap_iter_" + std::to_string(iter + 1) + ".ppm";
        guiding_grid.export_voxel_heatmap_ppm(test_point, heatmap_name);

        std::cout << "guiding pdf (testpoint (0,0)): " << guiding_grid.get_voxel(test_point).guiding_pdf[0][0] << std::endl;
        std::cout << "direct: " << directHits << " indirect: " << indirectHits << " misses: " << misses << " timeout: " << timedOut << std::endl;
    }

    // VISUALIZATION
    std::cout << "--> Exporting final trained 3D PDF Lobe mesh..." << std::endl;
    // vec3 dummy_normal(0.0f, 1.0f, 0.0f);
    // guiding_grid.export_raw_pdf_heatmap_sphere_to_ply("final_guiding_lobe.ply", test_point, guiding_grid);

    float radius = 0.2f; // Scaled down so it sits nicely inside
    guiding_grid.g_debug_print_rays = true;
    guiding_grid.export_scene_with_heatmap_and_ray_bundle_to_ply(
        "debug_scene.ply",
        world[0],
        test_point,
        test_point_normal,
        radius,
        guiding_grid,
        25);
    guiding_grid.g_debug_print_rays = false;

    auto training_end = std::chrono::system_clock::now();
    double training_time = std::chrono::duration<double>(training_end - training_start).count();
    std::cout << "TRAINING FINISHED in " << training_time << std::endl;

// =========================================================================
// RESET DIAGNOSTICS FOR PRODUCTION PASS ONLY
// =========================================================================
#pragma omp critical
    {
        g_heatmap_full_paths.clear(); // Wipe out training paths entirely
    }
    directHits = 0;
    indirectHits = 0;
    guidedHits = 0;
    neeHits = 0;
    brdfHits = 0;
    misses = 0;
    timedOut = 0;
    test_point = point3(0, 0, 0); // Clear the pinned target so it locks onto a production path

    auto production_start = std::chrono::system_clock::now(); // production time
    // =========================================================================
    // PASS 2: FINAL PRODUCTION PASS
    // =========================================================================
    int production_spp = 9;
    guiding_grid.is_training = false; // Freeze modifications!
    int N_prod = static_cast<int>(std::sqrt(production_spp));
    std::vector<color> framebuffer(image_width * image_height);
    std::cout << "--> Starting Production Pass with Fully Trained Guides (" << production_spp << " SPP)..." << std::endl;

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

                    // SEED history with the camera center
                    pixel_color += guided_rec_trace(r, bvh_tree, 0, lights, tex_manager, guiding_grid, true, {camera_center}); // true
                }
            }

            framebuffer[j * image_width + i] = pixel_color / static_cast<float>(production_spp);
        }
    }

    auto production_end = std::chrono::system_clock::now();
    double production_time = std::chrono::duration<double>(production_end - production_start).count();

    std::cout << "RENDERING COMPLETE in " << production_time << std::endl;
    std::ofstream out("image.ppm");
    out << "P3\n"
        << image_width << ' ' << image_height << "\n255\n";
    for (const auto &c : framebuffer)
        write_color(out, c);
    out.close();

    // =========================================================================
    // PRINT PATHS TERMINAL PROFILE WITH PER-VERTEX RADIANCE
    // =========================================================================
    std::cout << "\n==================================================" << std::endl;
    std::cout << "--> Printing captured paths containing the test_point voxel ("
              << g_heatmap_full_paths.size() << " total paths found)" << std::endl;
    std::cout << "==================================================" << std::endl;

    /* for (size_t path_idx = 0; path_idx < g_heatmap_full_paths.size(); ++path_idx)
    {
        const auto &path = g_heatmap_full_paths[path_idx];

         std::cout << "Path #" << path_idx << " [Length: " << path.vertices.size() << " vertices]:\n";

        for (size_t v_idx = 0; v_idx < path.vertices.size(); ++v_idx)
        {
            const auto &pt = path.vertices[v_idx];

            std::cout << "  Vertex [" << v_idx << "]: ("
                      << pt.x() << ", " << pt.y() << ", " << pt.z() << ")";

            if (v_idx < path.vertex_radiance.size())
            {
                const auto &rad = path.vertex_radiance[v_idx];
                std::cout << " | Throughput Radiance -> [R: " << rad.x()
                          << ", G: " << rad.y()
                          << ", B: " << rad.z() << "]";
            }
            else
            {
                std::cout << " | Throughput Radiance -> [R: 0, G: 0, B: 0] (Primary Lens Origin)";
            }

            if (guiding_grid.get_voxel_index(pt) == guiding_grid.get_voxel_index(test_point))
            {
                std::cout << "  <-- MATCHES TEST_POINT VOXEL";
            }
            std::cout << "\n";
        }
        std::cout << "============================================" << std::endl;
    } */

    // =========================================================================
    // EXPORT 3D STRANDS TO OBJ
    // =========================================================================
    if (!g_heatmap_full_paths.empty())
    {
        std::cout << "--> Exporting 3D radiant path strands of point " << test_point << " to OBJ format..." << std::endl;
        export_radiance_forward_paths("./objects/CornellBox-Empty-RG.obj", "CornellBox-Empty-RG.mtl", "captured_radiant_paths.obj", g_heatmap_full_paths, guiding_grid);
    }
}