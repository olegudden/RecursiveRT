#ifndef PATH_GUIDING_HPP
#define PATH_GUIDING_HPP
#define _USE_MATH_DEFINES

#include "vec3.hpp"
#include "light_source.hpp"

struct Voxel
{
    // 8 rows (theta: 0 to pi) x 16 columns (phi: 0 to 2pi)
    // Storing accumulated incoming radiance
    float spherical_bins[8][16] = {0.0f}; // Raw workspace (Wiped every iteration)

    float guiding_pdf[8][16] = {0.0f}; // Clean, normalized probabilities (Kept for guiding)
};

class VoxelGrid
{
public:
    VoxelGrid() = default;

    // Initializes the grid based on the scene's bounding box boundaries
    void initialize(point3 min_bounds, point3 max_bounds, int x_res, int y_res, int z_res);

    // Find which voxel a 3D world position sits inside
    int get_voxel_index(const point3 &p) const;
    Voxel &get_voxel(const point3 &p);

    // Record incoming light into the data structure
    void record_radiance(const point3 &hit_p, const vec3 &ray_dir, const color &radiance);

    float get_bin_solid_angle(int t, int p) const;

    // Generate a guided direction based on a voxel's distribution and a surface normal
    vec3 sample_guided_direction(const point3 &hit_p, const vec3 &normal, float &out_pdf_guiding);

    void finalize_and_build_distributions();

    // color guided_rec_trace(const ray &r, const bvh &bvh_tree, int depth, std::vector<area_light> &lights, VoxelGrid &grid, bool use_path_guiding);

    void export_voxel_heatmap_ppm(const point3 &world_p, const std::string &filename);
    void export_voxel_heatmap_ppm2(const point3 &world_p, const std::string &filename);

    float get_guided_pdf(const point3 &world_p, const vec3 &normal, const vec3 &dir) const;
    float get_guided_pdf_interpolated(const point3 &world_p, const vec3 &normal, const vec3 &dir) const;

private:
    point3 bounds_min, bounds_max;
    int res_x = 0, res_y = 0, res_z = 0;
    std::vector<Voxel> voxels;

    // Helper functions to convert between 3D vectors and spherical bin coordinates
    void vector_to_bin(const vec3 &dir, int &out_theta_idx, int &out_phi_idx) const;
    vec3 bin_to_vector(int theta_idx, int phi_idx) const;
};

#endif