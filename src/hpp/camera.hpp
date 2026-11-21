#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "light_source.hpp"
#include "path_guiding.hpp"
#include "spatial_hash_grid.hpp"
#include "texture.hpp"
#include "color.hpp"
#include "instance.hpp"

#include <vector>
#include <fstream>
#include <atomic>

struct TraceResult
{
    color full;
    color brdf_only;
};

struct VisualRayPath
{
    std::vector<point3> vertices;
    std::vector<color> vertex_radiance;
};

class camera
{

public:
    // texture
    int width = 1024;
    int height = 1024;
    int channels = 3;
    unsigned char *data = stbi_load("Crate_Texture.png", &width, &height, &channels, 3); // single texture

    bool adjustable = false;
    double aspect_ratio = 1.0;
    int image_width = 100;

    std::atomic<int> directHits{0};
    std::atomic<int> indirectHits{0};
    std::atomic<int> neeHits{0};
    std::atomic<int> brdfHits{0};
    std::atomic<int> guidedHits{0};
    std::atomic<int> misses{0};
    std::atomic<int> timedOut{0};

    std::atomic<int> transparentHits{0};
    std::atomic<int> mirrorHits{0};

    camera() = default;

    // Render the scene
    void timed_comparison_render(const std::vector<object> &world,
                                 std::vector<area_light> &lights,
                                 bvh &bvh_tree,
                                 SpatialHashGrid &guiding_grid, // VoxelGrid/SpatialHashGrid
                                 TextureManager &tex_manager,
                                 double total_time_budget_seconds,
                                 bool enable_guiding,
                                 double training_time_budget_seconds); // 0 for non-guided
    void render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree, VoxelGrid &grid, TextureManager &tex_manager);
    void old_render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree);                                                             // area_light
    void new_render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree, SpatialHashGrid &guiding_grid, TextureManager &tex_manager); // VoxelGrid/SpatialHashGrid

    void online_render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree, SpatialHashGrid &guiding_grid, TextureManager &tex_manager);
    void hybrid_render(const std::vector<object> &world, std::vector<area_light> &lights, bvh &bvh_tree, SpatialHashGrid &guiding_grid, TextureManager &tex_manager);

    // Helper function: uniform random in [0, 1) (for antialiasing)
    double rand_double();

    vec3 random_in_unit_sphere();

    float power_heuristic(float pdf_a, float pdf_b);
    float balance_heuristic(float pdf_a, float pdf_b);

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

    color baseline_rec_trace(const ray &r, const bvh &bvh_tree, int depth);

    color nee_rec_trace(const ray &r, const bvh &bvh_tree, int depth, const std::vector<area_light> &lights);

    color newest_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, TextureManager &tex_manager);

    color oldest_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights);

    void debug_nan_state(const std::string &location, float NdotL, float NdotV, float NdotH, float VdotH, float LdotH, float roughness, float D, float G, float pdf_brdf);
    bool is_nan_color(const color &c);

    color guided_rec_trace(
        const ray &r,
        const bvh &bvh_tree,
        int depth,
        std::vector<area_light> &lights,
        TextureManager &tex_manager,
        VoxelGrid &grid,
        bool use_path_guiding,
        std::vector<point3> path_history,
        float current_alpha);

    color guided_rec_trace_HASH(
        const ray &r,
        const bvh &bvh_tree,
        int depth,
        std::vector<area_light> &lights,
        TextureManager &tex_manager,
        SpatialHashGrid &grid, // <--- CHANGED TO SpatialHashGrid
        SpatialHashGrid &grid2,
        bool use_path_guiding,
        std::vector<point3> path_history,
        float current_alpha);

    color divided_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, TextureManager &tex_manager, VoxelGrid &grid, bool use_guiding);
    color next_event_estimation(const vec3 &wi, const vec3 &n, const hit_info &hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, std::vector<float> local_probs, TextureManager &tex_manager, VoxelGrid &grid, float alpha, bool use_guiding);
    color indirect_illumination(const vec3 &wi, const vec3 &n, const hit_info &hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, std::vector<float> local_probs, TextureManager &tex_manager, VoxelGrid &grid, float alpha, bool use_guiding);
    color eval_bsdf(const material &m, const vec3 &p, const vec3 &n, const vec3 &wi, const vec3 &wo, float &pdf_factor, bool &is_delta);
    vec3 sample_from_bsdf(const vec3 &wi, const vec3 &n, material m);
    color recursion_mirror(vec3 I, vec3 N, hit_info hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, TextureManager &tex_manager, VoxelGrid &grid, bool use_guiding);
    color recursion_transparent(vec3 I, vec3 N, hit_info hit, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, TextureManager &tex_manager, VoxelGrid &grid, bool use_guiding);

    void get_heatmap_color(const color &radiance, float &out_r, float &out_g, float &out_b);

    void export_radiance_forward_paths(
        const std::string &base_scene_filename,
        const std::string &base_mtl_filename,
        const std::string &output_obj_filename,
        const std::vector<VisualRayPath> &paths,
        const VoxelGrid &guiding_grid);

    void export_spatial_hash_grid(
        const std::string &base_scene_filename,
        const std::string &base_mtl_filename,
        const std::string &output_obj_filename,
        const SpatialHashGrid &hash_grid);

    vec3 dielectric_F0_from_tint(const vec3 &spec_tint);
    vec3 sample_phong_lobe(const vec3 &r, const vec3 &n, float s, float &pdf_phong);
    color brdf_diffuse(const material &m);
    color tex_diffuse(hit_info &hi);
    color tex_base_color(const hit_info &hi, TextureManager &tex_manager);
    color eval_emission(const material &mat, float u, float v, TextureManager &tex_manager);
    color evaluate_disney_diffuse(
        const color &base_color,
        float roughness,
        float subsurface, // New parameter (0.0 = pure diffuse, 1.0 = pure subsurface)
        float NdotL,
        float NdotV,
        float NdotH, // New dot product needed for subsurface grazing
        float LdotH);

    color brdf_specular(const material &m, const float s, const vec3 &refl_dir, const vec3 &wo);
    float ndf_ggx(float NdotH, float roughness);
    // float ga_smith_ggx(float NdotL, float NdotV, float roughness);
    vec3 sample_ggx_lobe(const vec3 &n, float roughness, float r1, float r2, float &pdf);

    float evaluate_clearcoat(float NdotH, float clearcoat_roughness, float NdotL, float NdotV, float VdotH);
    color evaluate_sheen(const color &base_color, float VdotH);
};

#endif // CAMERA_H
