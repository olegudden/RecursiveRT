#ifndef PATH_GUIDING_HPP
#define PATH_GUIDING_HPP
#define _USE_MATH_DEFINES

#include "vec3.hpp"
#include "light_source.hpp"
#include "directional_quadtree.hpp"
#include "sphere_mapping.hpp"
#include "spatial_datastructure.hpp"

struct Voxel
{
    bool use_quadtree = true; // enable/disable quadtree TODO for uniformGrid

    // float total_radiance = 0.0f;
    float total_weight = 0.0f;

    // Quadtree
    DirectionalQuadtree dtree;

    // Spherical Bins
    // 8 rows (theta: 0 to pi) x 16 columns (phi: 0 to 2pi) storing accumulated incoming radiance
    float spherical_bins[BIN_ROWS][BIN_COLOMNS] = {0.0f}; // Raw workspace (Wiped every iteration)

    float persistent_bins[BIN_ROWS][BIN_COLOMNS];
    // Normalized guiding probabilities.
    // Initialized to 1.0 / 128 so unvisited voxels start as a uniform distribution.
    // TODO dominating  material pdf as prior
    float guiding_pdf[BIN_ROWS][BIN_COLOMNS];

    Voxel()
    {
        reset();
    }

    void reset()
    {
        if (use_quadtree)
        {
            total_weight = 0.0f;
            dtree.reset();
        }
        else
        {
            // total_weight = 0.0f;
            for (int t = 0; t < BIN_ROWS; ++t)
            {
                for (int p = 0; p < BIN_COLOMNS; ++p)
                {
                    spherical_bins[t][p] = 0.0f;
                    guiding_pdf[t][p] = 1.0f / static_cast<float>(BIN_ROWS * BIN_COLOMNS);
                }
            }
        }
    }
};

class VoxelGrid : public SpatialDatastructure
{
public:
    std::vector<Voxel> voxels;

    point3 bounds_min, bounds_max;
    int res_x = 1, res_y = 1, res_z = 1;

    // Set to true only when generating your diagnostic PLY rays
    bool g_debug_print_rays = false;
    bool is_training = true;

    VoxelGrid() = default;

    // Initializes the grid based on the scene's bounding box boundaries
    void initialize(point3 min_bounds, point3 max_bounds, int x_res, int y_res, int z_res);

    // Find which voxel a 3D world position sits inside
    int get_voxel_index(const point3 &p) const;
    Voxel &get_voxel(const point3 &p);

    // Record incoming light into the data structure
    void record_radiance(const point3 &hit_p, const vec3 &ray_dir, const color &radiance);

    float get_bin_solid_angle(int t, int p) const;

    point3 get_voxel_center(int x, int y, int z) const;

    // Generate a guided direction based on a voxel's distribution and a surface normal
    vec3 sample_guided_direction(const point3 &hit_p, const vec3 &normal, float &out_pdf_guiding);
    vec3 sample_guided_direction_interpolated(const point3 &hit_p, const vec3 &normal, float &out_pdf_guiding);

    void finalize_and_build_distributions();
    void finalize_and_build_distributions_EMA(int iteration_m);

    // color guided_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, VoxelGrid &grid, bool use_path_guiding);

    void export_voxel_heatmap_ppm(const point3 &world_p, const std::string &filename);

    float get_guided_pdf(const point3 &world_p, const vec3 &normal, const vec3 &dir) const;
    float get_guided_pdf_interpolated(const point3 &world_p, const vec3 &normal, const vec3 &dir) const;

    void export_scene_with_heatmap_and_ray_bundle_to_ply(
        const std::string &filename,
        const object &scene_obj,
        const vec3 &position,
        const vec3 &surface_normal,
        float sphere_radius,
        VoxelGrid &grid,
        int num_samples);

    void export_deterministic_test_to_ply(
        const std::string &filename,
        const object &scene_obj,
        const vec3 &position,
        const vec3 &surface_normal,
        float sphere_radius,
        int test_t, // Theta bin to force to 100% (e.g. 3)
        int test_p, // Phi bin to force to 100% (e.g. 6)
        int num_samples);

    // void export_pdf_lobe_to_obj(const std::string &filename, const vec3 &position, const vec3 &normal, VoxelGrid &grid);

    void test_heatmap_directions();

private:
    // Helper functions to convert between 3D vectors and spherical bin coordinates
    void vector_to_bin(const vec3 &dir, int &out_theta_idx, int &out_phi_idx) const;
    vec3 bin_to_vector(int theta_idx, int phi_idx) const;
};

#endif