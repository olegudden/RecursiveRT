#include "../hpp/path_guiding.hpp"
#include "../hpp/mesh_loader.hpp"
#include <fstream>

using ActiveMapping = SphereMapping::EqualArea; // choose sphere mapping

void VoxelGrid::initialize(point3 min_bounds, point3 max_bounds, int x_res, int y_res, int z_res)
{
    bounds_min = min_bounds;
    bounds_max = max_bounds;
    res_x = x_res;
    res_y = y_res;
    res_z = z_res;
    voxels.resize(res_x * res_y * res_z);

    std::cout << "UNIFORM GRID (resolution): " << res_x << "x" << res_y << "x" << res_z << ", overall: " << res_x * res_y * res_z << std::endl;
    std::cout << "Voxel size (uniform grid): " << vec3((max_bounds.x() - min_bounds.x()) / res_x, (max_bounds.y() - min_bounds.y()) / res_y, (max_bounds.z() - min_bounds.z()) / res_z) << std::endl;
}

int VoxelGrid::get_voxel_index(const point3 &p) const
{
    // 1. Check if the point is actually inside the bounds (with a tiny epsilon)
    const float eps = 1e-5f;
    if (p.x() < bounds_min.x() - eps || p.x() > bounds_max.x() + eps ||
        p.y() < bounds_min.y() - eps || p.y() > bounds_max.y() + eps ||
        p.z() < bounds_min.z() - eps || p.z() > bounds_max.z() + eps)
    {
        return -1; // Clearly outside
    }

    // 2. Calculate normalized coordinates (0.0 to 1.0)
    float tx = (p.x() - bounds_min.x()) / (bounds_max.x() - bounds_min.x());
    float ty = (p.y() - bounds_min.y()) / (bounds_max.y() - bounds_min.y());
    float tz = (p.z() - bounds_min.z()) / (bounds_max.z() - bounds_min.z());

    // 3. Map to integer indices without clamping to the very edge
    // This allows points exactly on the max boundary to safely fall into the last voxel
    int ix = std::min(static_cast<int>(tx * res_x), res_x - 1);
    int iy = std::min(static_cast<int>(ty * res_y), res_y - 1);
    int iz = std::min(static_cast<int>(tz * res_z), res_z - 1);

    return ix + res_x * (iy + res_y * iz);
}

Voxel &VoxelGrid::get_voxel(const point3 &p)
{
    return voxels[get_voxel_index(p)];
}

point3 VoxelGrid::get_voxel_center(int x, int y, int z) const
{
    float dx = (bounds_max.x() - bounds_min.x()) / res_x;
    float dy = (bounds_max.y() - bounds_min.y()) / res_y;
    float dz = (bounds_max.z() - bounds_min.z()) / res_z;

    return point3(
        bounds_min.x() + (x + 0.5f) * dx,
        bounds_min.y() + (y + 0.5f) * dy,
        bounds_min.z() + (z + 0.5f) * dz);
}

void VoxelGrid::vector_to_bin(const vec3 &dir, int &out_theta_idx, int &out_phi_idx) const
{
    // Delegate directly to the mapping implementation
    ActiveMapping::dir_to_bin(dir, out_theta_idx, out_phi_idx);
}

vec3 VoxelGrid::bin_to_vector(int theta_idx, int phi_idx) const
{
    float theta = (theta_idx + 0.5f) * (M_PI / 8.0f);
    float phi = (phi_idx + 0.5f) * (2.0f * M_PI / 16.0f);

    // Subtract the exact shift we added in vector_to_bin
    phi -= 1.5f * M_PI;

    float dx = std::sin(theta) * std::cos(phi);
    float dy = std::cos(theta);
    float dz = std::sin(theta) * std::sin(phi);
    vec3 ray_dir(dx, dy, dz);

    // Return the negative to match the photon direction format expected by record_radiance
    return -ray_dir;
}

void VoxelGrid::record_radiance(const point3 &hit_p, const vec3 &dir, const color &radiance)
{
    Voxel &vox = get_voxel(hit_p);

    // FIX: Do NOT negate 'dir' if you are passing 'wo' (outgoing direction)!
    // Just normalize it to ensure clean vector bounds.
    vec3 outgoing_dir = unit_vector(dir);

    int theta_idx, phi_idx;
    vector_to_bin(outgoing_dir, theta_idx, phi_idx);

    float luminance = 0.2126f * radiance[0] + 0.7152f * radiance[1] + 0.0722f * radiance[2];

// Atomic addition for OpenMP safety across threads
#pragma omp atomic
    vox.spherical_bins[theta_idx][phi_idx] += luminance;
    // vox.total_radiance += luminance;
}

float VoxelGrid::get_bin_solid_angle(int t, int p) const
{
    float delta_theta = M_PI / 8.0f;
    float delta_phi = (2.0f * M_PI) / 16.0f;

    float theta_start = t * delta_theta;
    float theta_end = (t + 1) * delta_theta;

    return delta_phi * (std::cos(theta_start) - std::cos(theta_end));
}

vec3 VoxelGrid::sample_guided_direction(const point3 &hit_p, const vec3 &normal, float &out_pdf_guiding)
{
    int sample_vox_idx = get_voxel_index(hit_p);
    const Voxel &vox = voxels[sample_vox_idx];

    float local_pdf[BIN_ROWS][BIN_COLOMNS];
    float total_sum = 0.0f;

    // 1. Compute surface-aligned weights for all bins using bin centers
    for (int t = 0; t < BIN_ROWS; t++)
    {
        for (int p = 0; p < BIN_COLOMNS; p++)
        {
            vec3 bin_center_dir = ActiveMapping::sample_bin(t, p, 0.5f, 0.5f);
            if (dot(normal, unit_vector(bin_center_dir)) > 0.0f)
            {
                local_pdf[t][p] = vox.guiding_pdf[t][p];
            }
            else
            {
                local_pdf[t][p] = 0.0f;
            }
            total_sum += local_pdf[t][p];
        }
    }

    vec3 wo;

    // 2. Fallback or CDF Sampling (NO early return with hardcoded PDFs!)
    if (total_sum <= 0.0f)
    {
        // Cosine-hemisphere fallback matching get_guided_pdf's fallback
        float r1 = rand_float();
        float r2 = rand_float();
        float phi = 2.0f * M_PI * r1;
        float r = std::sqrt(r2);

        float x = r * std::cos(phi);
        float y = r * std::sin(phi);
        float z = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y));

        vec3 w = unit_vector(normal);
        vec3 a = (std::abs(w.x()) > 0.9f) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        vec3 v = unit_vector(cross(w, a));
        vec3 u = cross(w, v);

        wo = unit_vector(x * u + y * v + z * w);
    }
    else
    {
        float r = rand_float() * total_sum;
        float running_sum = 0.0f;
        int sampled_t = 0, sampled_p = 0;
        bool found = false;

        for (int t = 0; t < BIN_ROWS && !found; t++)
        {
            for (int p = 0; p < BIN_COLOMNS; p++)
            {
                running_sum += local_pdf[t][p];
                if (r <= running_sum)
                {
                    sampled_t = t;
                    sampled_p = p;
                    found = true;
                    break;
                }
            }
        }

        // 3. Jitter within bin safely
        float u_jitter = std::clamp(rand_float(), 0.01f, 0.99f);
        float v_jitter = std::clamp(rand_float(), 0.01f, 0.99f);

        wo = unit_vector(ActiveMapping::sample_bin(sampled_t, sampled_p, u_jitter, v_jitter));

        if (dot(normal, wo) <= 0.0f)
        {
            wo = unit_vector(ActiveMapping::sample_bin(sampled_t, sampled_p, 0.5f, 0.5f));
        }
    }

    // 4. GUARANTEE exact match by evaluating through get_guided_pdf
    out_pdf_guiding = get_guided_pdf(hit_p, normal, wo);

    return wo;
}

vec3 VoxelGrid::sample_guided_direction_interpolated(const point3 &hit_p, const vec3 &normal, float &out_pdf_guiding)
{
    // 1. Compute trilinear spatial coordinates
    float tx = (hit_p.x() - bounds_min.x()) / (bounds_max.x() - bounds_min.x()) * res_x - 0.5f;
    float ty = (hit_p.y() - bounds_min.y()) / (bounds_max.y() - bounds_min.y()) * res_y - 0.5f;
    float tz = (hit_p.z() - bounds_min.z()) / (bounds_max.z() - bounds_min.z()) * res_z - 0.5f;

    int ix0 = std::clamp(static_cast<int>(std::floor(tx)), 0, res_x - 1);
    int iy0 = std::clamp(static_cast<int>(std::floor(ty)), 0, res_y - 1);
    int iz0 = std::clamp(static_cast<int>(std::floor(tz)), 0, res_z - 1);

    int ix1 = std::clamp(ix0 + 1, 0, res_x - 1);
    int iy1 = std::clamp(iy0 + 1, 0, res_y - 1);
    int iz1 = std::clamp(iz0 + 1, 0, res_z - 1);

    float fx = std::clamp(tx - std::floor(tx), 0.0f, 1.0f);
    float fy = std::clamp(ty - std::floor(ty), 0.0f, 1.0f);
    float fz = std::clamp(tz - std::floor(tz), 0.0f, 1.0f);

    float w[8];
    w[0] = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
    w[1] = fx * (1.0f - fy) * (1.0f - fz);
    w[2] = (1.0f - fx) * fy * (1.0f - fz);
    w[3] = fx * fy * (1.0f - fz);
    w[4] = (1.0f - fx) * (1.0f - fy) * fz;
    w[5] = fx * (1.0f - fy) * fz;
    w[6] = (1.0f - fx) * fy * fz;
    w[7] = fx * fy * fz;

    const int corner_indices[8][3] = {
        {ix0, iy0, iz0}, {ix1, iy0, iz0}, {ix0, iy1, iz0}, {ix1, iy1, iz0}, {ix0, iy0, iz1}, {ix1, iy0, iz1}, {ix0, iy1, iz1}, {ix1, iy1, iz1}};

    // 2. Geometry-aware filtering using voxel extent
    float voxel_size = (bounds_max.x() - bounds_min.x()) / static_cast<float>(res_x);
    float plane_epsilon = -0.5f * voxel_size;

    float accum_w = 0.0f;
    for (int k = 0; k < 8; ++k)
    {
        point3 corner_p = get_voxel_center(corner_indices[k][0], corner_indices[k][1], corner_indices[k][2]);

        // Zero out corners that sit deep inside walls behind the surface
        if (dot(corner_p - hit_p, normal) < plane_epsilon)
        {
            w[k] = 0.0f;
        }

        accum_w += w[k];
    }

    // Fallback to nearest neighbor if all corners are rejected by geometry plane
    if (accum_w <= 0.0f)
    {
        int nearest_x = std::clamp(static_cast<int>(std::round(tx)), 0, res_x - 1);
        int nearest_y = std::clamp(static_cast<int>(std::round(ty)), 0, res_y - 1);
        int nearest_z = std::clamp(static_cast<int>(std::round(tz)), 0, res_z - 1);

        for (int k = 0; k < 8; ++k)
        {
            w[k] = (corner_indices[k][0] == nearest_x &&
                    corner_indices[k][1] == nearest_y &&
                    corner_indices[k][2] == nearest_z)
                       ? 1.0f
                       : 0.0f;
        }
    }
    else
    {
        for (int k = 0; k < 8; ++k)
        {
            w[k] /= accum_w;
        }
    }

    // 3. Select corner stochastically
    float r_spatial = rand_float();
    float run_w = 0.0f;
    int chosen_corner = 0;

    for (int k = 0; k < 8; ++k)
    {
        run_w += w[k];
        if (r_spatial <= run_w)
        {
            chosen_corner = k;
            break;
        }
    }

    // 4. Extract chosen voxel corner
    int cx = corner_indices[chosen_corner][0];
    int cy = corner_indices[chosen_corner][1];
    int cz = corner_indices[chosen_corner][2];
    int idx = cx + res_x * (cy + res_y * cz);
    const Voxel &vox = voxels[idx];

    // 5. Build dynamic upper-hemisphere distribution for chosen corner
    float local_pdf[BIN_ROWS][BIN_COLOMNS];
    float total_sum = 0.0f;

    for (int t = 0; t < BIN_ROWS; t++)
    {
        for (int p = 0; p < BIN_COLOMNS; p++)
        {
            vec3 bin_center = ActiveMapping::sample_bin(t, p, 0.5f, 0.5f);
            if (dot(normal, bin_center) > 0.0f)
            {
                local_pdf[t][p] = vox.guiding_pdf[t][p];
            }
            else
            {
                local_pdf[t][p] = 0.0f;
            }
            total_sum += local_pdf[t][p];
        }
    }

    // Cosine-weighted hemisphere sampling fallback if guided weights are empty
    if (total_sum <= 0.0f)
    {
        float r1 = rand_float();
        float r2 = rand_float();
        float phi = 2.0f * M_PI * r1;
        float r = std::sqrt(r2);

        float x = r * std::cos(phi);
        float y = r * std::sin(phi);
        float z = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y));

        vec3 w_axis = unit_vector(normal);
        vec3 a = (std::abs(w_axis.x()) > 0.9f) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        vec3 v_axis = unit_vector(cross(w_axis, a));
        vec3 u_axis = cross(w_axis, v_axis);

        vec3 wo = unit_vector(x * u_axis + y * v_axis + z * w_axis);

        out_pdf_guiding = get_guided_pdf_interpolated(hit_p, normal, wo);
        return wo;
    }

    // 6. CDF bin search
    float r_bin = rand_float() * total_sum;
    float running_sum = 0.0f;
    int sampled_t = 0, sampled_p = 0;
    bool found = false;

    for (int t = 0; t < BIN_ROWS && !found; t++)
    {
        for (int p = 0; p < BIN_COLOMNS; p++)
        {
            running_sum += local_pdf[t][p];
            if (r_bin <= running_sum)
            {
                sampled_t = t;
                sampled_p = p;
                found = true;
                break;
            }
        }
    }

    // 7. Continuous direction jittering
    float u_jitter = std::clamp(rand_float(), 0.001f, 0.999f);
    float v_jitter = std::clamp(rand_float(), 0.001f, 0.999f);
    vec3 ray_dir = ActiveMapping::sample_bin(sampled_t, sampled_p, u_jitter, v_jitter);
    vec3 wo = unit_vector(ray_dir);

    if (dot(normal, wo) <= 0.0f)
    {
        wo = unit_vector(ActiveMapping::sample_bin(sampled_t, sampled_p, 0.5f, 0.5f));
    }

    // 8. Evaluate exact matching interpolated PDF
    out_pdf_guiding = get_guided_pdf_interpolated(hit_p, normal, wo);

    return wo;
}

void VoxelGrid::finalize_and_build_distributions()
{
    constexpr float INV_FOUR_PI = 1.0f / (4.0f * M_PI);
    constexpr float epsilon_floor = 0.25f; // uniform exploration floor - TRY DIFFERENT VALUES

    for (auto &vox : voxels)
    {
        float total_radiance = 0.0f;
        for (int t = 0; t < BIN_ROWS; t++)
            for (int p = 0; p < BIN_COLOMNS; p++)
                total_radiance += vox.spherical_bins[t][p];

        if (total_radiance > 0.0f)
        {
            for (int t = 0; t < BIN_ROWS; t++)
            {
                for (int p = 0; p < BIN_COLOMNS; p++)
                {
                    // 1. Trained probability mass
                    float p_mass = vox.spherical_bins[t][p] / total_radiance;

                    // 2. Solid angle for bin (t, p)
                    float solid_angle = ActiveMapping::bin_solid_angle(t, p);

                    // 3. True uniform mass for this specific bin size
                    float uniform_mass = solid_angle * INV_FOUR_PI;

                    // 4. Blend trained mass with area-weighted uniform mass
                    vox.guiding_pdf[t][p] = (1.0f - epsilon_floor) * p_mass + epsilon_floor * uniform_mass;

                    vox.spherical_bins[t][p] = 0.0f; // Reset accumulator
                }
            }
        }
        else
        {
            // Voxels with no data default to true uniform distribution per bin size
            for (int t = 0; t < BIN_ROWS; t++)
            {
                for (int p = 0; p < BIN_COLOMNS; p++)
                {
                    float solid_angle = ActiveMapping::bin_solid_angle(t, p);
                    vox.guiding_pdf[t][p] = solid_angle * INV_FOUR_PI;
                    vox.spherical_bins[t][p] = 0.0f;
                }
            }
        }
    }
}

void VoxelGrid::finalize_and_build_distributions_EMA(int iteration_m)
{
    constexpr float INV_FOUR_PI = 1.0f / (4.0f * M_PI);
    constexpr float epsilon_floor = 0.25f;

    // Choose your learning rate strategy:
    // Option A: Fixed learning rate (e.g., 0.1 for adaptive smoothing)
    float alpha_m = 0.1f;

    // Option B: Sample mean learning rate (alpha_m = 1.0 / m)
    // float alpha_m = 1.0f / static_cast<float>(std::max(1, iteration_m));

    for (auto &vox : voxels)
    {
        float total_persistent_radiance = 0.0f;

        // Step 1: Apply the EMA update formula: Y_j^(m) = (1 - alpha) * Y_j^(m-1) + alpha * Y_deposit
        for (int t = 0; t < BIN_ROWS; t++)
        {
            for (int p = 0; p < BIN_COLOMNS; p++)
            {
                float y_deposit = vox.spherical_bins[t][p]; // Current pass flux contribution

                // Blend old history with new deposit
                vox.persistent_bins[t][p] = (1.0f - alpha_m) * vox.persistent_bins[t][p] + alpha_m * y_deposit;

                total_persistent_radiance += vox.persistent_bins[t][p];

                // Clear current pass accumulator so it can collect fresh data next iteration
                vox.spherical_bins[t][p] = 0.0f;
            }
        }

        // Step 2: Build guiding distribution from the smoothed persistent history
        if (total_persistent_radiance > 0.0f)
        {
            for (int t = 0; t < BIN_ROWS; t++)
            {
                for (int p = 0; p < BIN_COLOMNS; p++)
                {
                    float p_mass = vox.persistent_bins[t][p] / total_persistent_radiance;
                    float solid_angle = ActiveMapping::bin_solid_angle(t, p);
                    float uniform_mass = solid_angle * INV_FOUR_PI;

                    vox.guiding_pdf[t][p] = (1.0f - epsilon_floor) * p_mass + epsilon_floor * uniform_mass;
                }
            }
        }
        else
        {
            // Voxels with no historical data default to uniform
            for (int t = 0; t < BIN_ROWS; t++)
            {
                for (int p = 0; p < BIN_COLOMNS; p++)
                {
                    float solid_angle = ActiveMapping::bin_solid_angle(t, p);
                    vox.guiding_pdf[t][p] = solid_angle * INV_FOUR_PI;
                }
            }
        }
    }
}

void VoxelGrid::export_voxel_heatmap_ppm(const point3 &world_p, const std::string &filename)
{
    // 1. BOUNDARY GUARD: If the test point is out of bounds, write an empty blue image and exit
    if (world_p.x() < bounds_min.x() || world_p.x() > bounds_max.x() ||
        world_p.y() < bounds_min.y() || world_p.y() > bounds_max.y() ||
        world_p.z() < bounds_min.z() || world_p.z() > bounds_max.z())
    {
        std::cout << "--> [Heatmap] Target " << world_p << " is OUT OF BOUNDS. Writing empty diagnostic map." << std::endl;

        int scale = 32;
        std::ofstream out(filename);
        out << "P3\n"
            << (BIN_COLOMNS * scale) << " " << (BIN_ROWS * scale) << "\n255\n";
        for (int i = 0; i < (BIN_COLOMNS * scale) * (BIN_ROWS * scale); i++)
        {
            out << "15 15 40\n"; // Dark Slate Blue for out-of-bounds/empty
        }
        out.close();
        return;
    }

    // 2. SURGICAL LOOKUP: Get the exact voxel
    Voxel &vox = get_voxel(world_p);

    float max_val = 0.0f;
    for (int t = 0; t < BIN_ROWS; t++)
    {
        for (int p = 0; p < BIN_COLOMNS; p++)
        {
            max_val = std::max(max_val, vox.guiding_pdf[t][p]);
        }
    }

    // If the voxel has never been visited/trained, max_val stays 0
    bool is_unvisited = (max_val <= 1e-6f);
    if (is_unvisited)
        max_val = 1.0f;

    int scale = 32;
    int img_w = BIN_COLOMNS * scale;
    int img_h = BIN_ROWS * scale;

    std::ofstream out(filename);
    out << "P3\n"
        << img_w << " " << img_h << "\n255\n";

    for (int y = 0; y < img_h; y++)
    {
        int t = y / scale;
        for (int x = 0; x < img_w; x++)
        {
            int p = x / scale;

            float intensity = vox.guiding_pdf[t][p] / max_val;

            int r = 0, g = 0, b = 0;

            if (is_unvisited)
            {
                // Unvisited voxel: Dark Blue
                r = 15;
                g = 15;
                b = 40;
            }
            else
            {
                // Beautiful Standard Jet Heatmap Color Scheme
                r = static_cast<int>(std::clamp((intensity - 0.25f) * 4.0f, 0.0f, 1.0f) * 255);
                g = static_cast<int>(std::clamp((0.5f - std::abs(intensity - 0.5f)) * 4.0f, 0.0f, 1.0f) * 255);
                b = static_cast<int>(std::clamp((0.75f - intensity) * 4.0f, 0.0f, 1.0f) * 255);
            }

            out << r << " " << g << " " << b << "\n";
        }
    }
    out.close();

    max_val = -1.0f;
    int max_t = -1, max_p = -1;

    for (int t = 0; t < BIN_ROWS; t++)
    {
        for (int p = 0; p < BIN_COLOMNS; p++)
        {
            if (vox.guiding_pdf[t][p] > max_val)
            {
                max_val = vox.guiding_pdf[t][p];
                max_t = t;
                max_p = p;
            }
        }
    }
    std::cout << "PDF max_val: " << max_val << " at max_t: " << max_t << ", max_p: " << max_p << std::endl;

    std::cout << "--> Exact Heatmap generated for " << world_p << "!" << std::endl;
}

float VoxelGrid::get_guided_pdf(const point3 &world_p, const vec3 &normal, const vec3 &dir) const
{
    float cos_dir = dot(normal, dir);
    if (cos_dir <= 0.0f)
        return 0.0f;

    int vox_idx = get_voxel_index(world_p);
    const Voxel &vox = voxels[vox_idx];

    int target_t = 0, target_p = 0;
    ActiveMapping::dir_to_bin(dir, target_t, target_p);

    float total_sum = 0.0f;

    for (int t = 0; t < BIN_ROWS; t++)
    {
        for (int p = 0; p < BIN_COLOMNS; p++)
        {
            vec3 bin_center_dir = ActiveMapping::sample_bin(t, p, 0.5f, 0.5f);
            if (dot(normal, unit_vector(bin_center_dir)) > 0.0f)
            {
                total_sum += vox.guiding_pdf[t][p];
            }
        }
    }

    // Must match the fallback branch in sample_guided_direction
    if (total_sum <= 0.0f)
    {
        return cos_dir / M_PI;
    }

    float target_weight = vox.guiding_pdf[target_t][target_p];

    // Safety check for horizon clipping
    vec3 target_center = unit_vector(ActiveMapping::sample_bin(target_t, target_p, 0.5f, 0.5f));
    if (dot(normal, target_center) <= 0.0f)
    {
        return 0.0f;
    }

    float discrete_prob = target_weight / total_sum;
    float solid_angle = ActiveMapping::bin_solid_angle(target_t, target_p);

    float final_pdf = discrete_prob / solid_angle;

    return final_pdf;
}

float VoxelGrid::get_guided_pdf_interpolated(const point3 &world_p, const vec3 &normal, const vec3 &dir) const
{
    float cos_theta = dot(normal, dir);
    if (cos_theta <= 0.0f)
        return 0.0f;

    int target_t = 0, target_p = 0;
    ActiveMapping::dir_to_bin(dir, target_t, target_p);

    float solid_angle = ActiveMapping::bin_solid_angle(target_t, target_p);
    if (solid_angle <= 0.0f)
        return 0.0f;

    // 1. Compute trilinear spatial coordinates
    float tx = (world_p.x() - bounds_min.x()) / (bounds_max.x() - bounds_min.x()) * res_x - 0.5f;
    float ty = (world_p.y() - bounds_min.y()) / (bounds_max.y() - bounds_min.y()) * res_y - 0.5f;
    float tz = (world_p.z() - bounds_min.z()) / (bounds_max.z() - bounds_min.z()) * res_z - 0.5f;

    int ix0 = std::clamp(static_cast<int>(std::floor(tx)), 0, res_x - 1);
    int iy0 = std::clamp(static_cast<int>(std::floor(ty)), 0, res_y - 1);
    int iz0 = std::clamp(static_cast<int>(std::floor(tz)), 0, res_z - 1);

    int ix1 = std::clamp(ix0 + 1, 0, res_x - 1);
    int iy1 = std::clamp(iy0 + 1, 0, res_y - 1);
    int iz1 = std::clamp(iz0 + 1, 0, res_z - 1);

    float fx = std::clamp(tx - std::floor(tx), 0.0f, 1.0f);
    float fy = std::clamp(ty - std::floor(ty), 0.0f, 1.0f);
    float fz = std::clamp(tz - std::floor(tz), 0.0f, 1.0f);

    float w[8];
    w[0] = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
    w[1] = fx * (1.0f - fy) * (1.0f - fz);
    w[2] = (1.0f - fx) * fy * (1.0f - fz);
    w[3] = fx * fy * (1.0f - fz);
    w[4] = (1.0f - fx) * (1.0f - fy) * fz;
    w[5] = fx * (1.0f - fy) * fz;
    w[6] = (1.0f - fx) * fy * fz;
    w[7] = fx * fy * fz;

    const int corner_indices[8][3] = {
        {ix0, iy0, iz0}, {ix1, iy0, iz0}, {ix0, iy1, iz0}, {ix1, iy1, iz0}, {ix0, iy0, iz1}, {ix1, iy0, iz1}, {ix0, iy1, iz1}, {ix1, iy1, iz1}};

    // 2. Geometry-aware filtering using voxel extent
    float voxel_size = (bounds_max.x() - bounds_min.x()) / static_cast<float>(res_x);
    float plane_epsilon = -0.5f * voxel_size;

    float accum_w = 0.0f;
    for (int k = 0; k < 8; ++k)
    {
        point3 corner_p = get_voxel_center(corner_indices[k][0], corner_indices[k][1], corner_indices[k][2]);

        // Zero out corners that sit deep inside walls behind the surface
        if (dot(corner_p - world_p, normal) < plane_epsilon)
        {
            w[k] = 0.0f;
        }

        accum_w += w[k];
    }

    // Fallback to nearest neighbor if all corners are rejected by geometry plane
    if (accum_w <= 0.0f)
    {
        int nearest_x = std::clamp(static_cast<int>(std::round(tx)), 0, res_x - 1);
        int nearest_y = std::clamp(static_cast<int>(std::round(ty)), 0, res_y - 1);
        int nearest_z = std::clamp(static_cast<int>(std::round(tz)), 0, res_z - 1);

        for (int k = 0; k < 8; ++k)
        {
            w[k] = (corner_indices[k][0] == nearest_x &&
                    corner_indices[k][1] == nearest_y &&
                    corner_indices[k][2] == nearest_z)
                       ? 1.0f
                       : 0.0f;
        }
    }
    else
    {
        for (int k = 0; k < 8; ++k)
        {
            w[k] /= accum_w;
        }
    }

    // 3. Evaluate matching interpolated PDF across valid corners
    float interpolated_discrete_prob = 0.0f;

    for (int k = 0; k < 8; ++k)
    {
        if (w[k] <= 0.0f)
            continue;

        int idx = corner_indices[k][0] + res_x * (corner_indices[k][1] + res_y * corner_indices[k][2]);
        const Voxel &vox = voxels[idx];

        // Recompute upper-hemisphere normalization for this corner
        float corner_hemi_sum = 0.0f;
        for (int t = 0; t < BIN_ROWS; t++)
        {
            for (int p = 0; p < BIN_COLOMNS; p++)
            {
                vec3 bin_center = ActiveMapping::sample_bin(t, p, 0.5f, 0.5f);
                if (dot(normal, bin_center) > 0.0f)
                {
                    corner_hemi_sum += vox.guiding_pdf[t][p];
                }
            }
        }

        if (corner_hemi_sum > 0.0f)
        {
            interpolated_discrete_prob += w[k] * (vox.guiding_pdf[target_t][target_p] / corner_hemi_sum);
        }
        else
        {
            // Cosine hemisphere fallback PDF value
            interpolated_discrete_prob += w[k] * (cos_theta / M_PI) * solid_angle;
        }
    }

    return interpolated_discrete_prob / solid_angle;
}

// Simple Turbo/Rainbow Colormap: Maps normalized value [0, 1] -> RGB Color
// 0.0 = Pure Blue, 0.5 = Green/Yellow, 1.0 = Pure Red
inline vec3 pdf_to_heatmap_color(float val)
{
    val = std::clamp(val, 0.0f, 1.0f);

    // Smooth blue-to-cyan-to-green-to-yellow-to-red gradient
    float r = std::clamp(1.5f - std::abs(val * 4.0f - 3.0f), 0.0f, 1.0f);
    float g = std::clamp(1.5f - std::abs(val * 4.0f - 2.0f), 0.0f, 1.0f);
    float b = std::clamp(1.5f - std::abs(val * 4.0f - 1.0f), 0.0f, 1.0f);

    return vec3(r, g, b);
};

void VoxelGrid::export_scene_with_heatmap_and_ray_bundle_to_ply( // TODO color scene_obj
    const std::string &filename,
    const object &scene_obj,
    const vec3 &position,
    const vec3 &surface_normal,
    float sphere_radius,
    VoxelGrid &grid,
    int num_samples)
{
    constexpr int theta_steps = BIN_ROWS;
    constexpr int phi_steps = BIN_COLOMNS;

    Voxel &vox = grid.get_voxel(position);

    // 1. Sample N directions from guided sampler
    std::vector<vec3> sampled_dirs;
    sampled_dirs.reserve(num_samples);
    for (int i = 0; i < num_samples; ++i)
    {
        float dummy_pdf = 0.0f;
        vec3 dir = grid.sample_guided_direction(position, surface_normal, dummy_pdf);
        sampled_dirs.push_back(dir);
    }

    // 2. Find max surface-weighted PDF for color normalization
    float max_val = 1e-4f;
    float effective_weights[theta_steps][phi_steps];

    for (int t = 0; t < theta_steps; ++t)
    {
        for (int p = 0; p < phi_steps; ++p)
        {
            // Exact bin center vector (Positive, matching sample_guided_direction)
            vec3 bin_dir = unit_vector(ActiveMapping::sample_bin(t, p, 0.5f, 0.5f));
            float cos_s = dot(surface_normal, bin_dir);

            if (cos_s > 0.0f)
            {
                effective_weights[t][p] = vox.guiding_pdf[t][p] * cos_s;
            }
            else
            {
                effective_weights[t][p] = 0.0f;
            }

            max_val = std::max(max_val, effective_weights[t][p]);
        }
    }

    // 3. Geometry counts
    int scene_vertices_count = static_cast<int>(scene_obj.mesh.size() * 3);
    int scene_faces_count = static_cast<int>(scene_obj.mesh.size());

    int sphere_vertices_count = theta_steps * phi_steps;
    int sphere_faces_count = (theta_steps - 1) * phi_steps * 2;

    int rays_vertices_count = num_samples * 4;
    int rays_faces_count = num_samples * 2;

    int total_vertices = scene_vertices_count + sphere_vertices_count + rays_vertices_count;
    int total_faces = scene_faces_count + sphere_faces_count + rays_faces_count;

    std::ofstream file(filename);
    if (!file.is_open())
        return;

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << total_vertices << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face " << total_faces << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";

    // =========================================================================
    // PASS A: SCENE GEOMETRY (Gray)
    // =========================================================================
    for (const auto &tri : scene_obj.mesh)
    {
        file << tri.v0.x() << " " << tri.v0.y() << " " << tri.v0.z() << " 180 180 180\n";
        file << tri.v1.x() << " " << tri.v1.y() << " " << tri.v1.z() << " 180 180 180\n";
        file << tri.v2.x() << " " << tri.v2.y() << " " << tri.v2.z() << " 180 180 180\n";
    }

    // =========================================================================
    // PASS B: HEATMAP SPHERE GEOMETRY (FIXED)
    // =========================================================================
    for (int t = 0; t < theta_steps; ++t)
    {
        for (int p = 0; p < phi_steps; ++p)
        {
            // FIX 1: Positive direction matching LatLong mapping (No negative sign!)
            vec3 outward_dir = unit_vector(SphereMapping::LatLong::sample_bin(t, p, 0.5f, 0.5f));
            vec3 vertex_pos = position + outward_dir * sphere_radius;

            // FIX 2: Surface-normal weighted PDF matching sampling weights
            float weight = effective_weights[t][p];
            float normalized_val = std::clamp(weight / max_val, 0.0f, 1.0f);
            vec3 rgb_color = pdf_to_heatmap_color(normalized_val);

            int r = static_cast<int>(std::clamp(rgb_color.x() * 255.0f, 0.0f, 255.0f));
            int g = static_cast<int>(std::clamp(rgb_color.y() * 255.0f, 0.0f, 255.0f));
            int b = static_cast<int>(std::clamp(rgb_color.z() * 255.0f, 0.0f, 255.0f));

            file << vertex_pos.x() << " " << vertex_pos.y() << " " << vertex_pos.z() << " "
                 << r << " " << g << " " << b << "\n";
        }
    }

    // =========================================================================
    // PASS C: BUNDLE OF SAMPLED RAY RIBBONS (Yellow)
    // =========================================================================
    for (int i = 0; i < num_samples; ++i)
    {
        vec3 dir = unit_vector(sampled_dirs[i]);
        vec3 ray_start = position;
        vec3 ray_end = position + dir * (sphere_radius * 2.5f);

        vec3 up = (std::abs(dir.y()) > 0.9f) ? vec3(1, 0, 0) : vec3(0, 1, 0);
        vec3 right = unit_vector(cross(dir, up)) * (sphere_radius * 0.03f);

        vec3 rv0 = ray_start - right;
        vec3 rv1 = ray_start + right;
        vec3 rv2 = ray_end - right;
        vec3 rv3 = ray_end + right;

        file << rv0.x() << " " << rv0.y() << " " << rv0.z() << " 255 255 0\n";
        file << rv1.x() << " " << rv1.y() << " " << rv1.z() << " 255 255 0\n";
        file << rv2.x() << " " << rv2.y() << " " << rv2.z() << " 255 255 0\n";
        file << rv3.x() << " " << rv3.y() << " " << rv3.z() << " 255 255 0\n";
    }

    // =========================================================================
    // FACES: SCENE
    // =========================================================================
    for (size_t i = 0; i < scene_obj.mesh.size(); ++i)
    {
        int base_idx = static_cast<int>(i * 3);
        file << "3 " << base_idx << " " << (base_idx + 1) << " " << (base_idx + 2) << "\n";
    }

    // =========================================================================
    // FACES: SPHERE
    // =========================================================================
    int sphere_base_idx = scene_vertices_count;
    for (int t = 0; t < theta_steps - 1; ++t)
    {
        for (int p = 0; p < phi_steps; ++p)
        {
            int p_next = (p + 1) % phi_steps;
            int v1 = sphere_base_idx + (t * phi_steps + p);
            int v2 = sphere_base_idx + (t * phi_steps + p_next);
            int v3 = sphere_base_idx + ((t + 1) * phi_steps + p);
            int v4 = sphere_base_idx + ((t + 1) * phi_steps + p_next);

            file << "3 " << v1 << " " << v3 << " " << v2 << "\n";
            file << "3 " << v2 << " " << v3 << " " << v4 << "\n";
        }
    }

    // =========================================================================
    // FACES: RAY BUNDLES
    // =========================================================================
    int rays_base_idx = scene_vertices_count + sphere_vertices_count;
    for (int i = 0; i < num_samples; ++i)
    {
        int r_idx = rays_base_idx + (i * 4);
        file << "3 " << r_idx << " " << (r_idx + 1) << " " << (r_idx + 2) << "\n";
        file << "3 " << (r_idx + 1) << " " << (r_idx + 3) << " " << (r_idx + 2) << "\n";
    }

    file.close();
    std::cout << "--> [Success] Exported heatmap with " << num_samples << " ray samples to " << filename << std::endl;
}

void VoxelGrid::export_deterministic_test_to_ply(
    const std::string &filename,
    const object &scene_obj,
    const vec3 &position,
    const vec3 &surface_normal,
    float sphere_radius,
    int test_t, // Theta bin to force to 100% (e.g. 3)
    int test_p, // Phi bin to force to 100% (e.g. 6)
    int num_samples)
{
    constexpr int theta_steps = BIN_ROWS;
    constexpr int phi_steps = BIN_COLOMNS;

    // Sanity clamp target bin
    test_t = std::clamp(test_t, 0, theta_steps - 1);
    test_p = std::clamp(test_p, 0, phi_steps - 1);

    // 1. Create a dummy test Voxel with exactly ONE bin set to 1.0f and the rest 0.0f
    Voxel test_vox;
    for (int t = 0; t < theta_steps; ++t)
    {
        for (int p = 0; p < phi_steps; ++p)
        {
            test_vox.guiding_pdf[t][p] = (t == test_t && p == test_p) ? 1.0f : 0.0f;
        }
    }
    test_vox.total_weight = 1.0f;

    // 2. Sample N directions against our single-bin distribution
    std::vector<vec3> sampled_dirs;
    sampled_dirs.reserve(num_samples);

    // Build upper-hemisphere distribution matching sample_guided_direction logic
    float local_pdf[BIN_ROWS][BIN_COLOMNS];
    float total_sum = 0.0f;

    for (int t = 0; t < theta_steps; ++t)
    {
        for (int p = 0; p < phi_steps; ++p)
        {
            // vec3 bin_dir = unit_vector(ActiveMapping::sample_bin(t, p, 0.5f, 0.5f));
            // if (dot(surface_normal, bin_dir) > 0.0f)
            //{
            local_pdf[t][p] = test_vox.guiding_pdf[t][p];
            //}
            /* else
            {
                local_pdf[t][p] = 0.0f;
            } */
            total_sum += local_pdf[t][p];
        }
    }

    // Sample rays using the exact discrete CDF + continuous bin jitter procedure
    for (int i = 0; i < num_samples; ++i)
    {
        if (total_sum <= 0.0f)
            break; // Test bin faces away from surface normal hemisphere

        float r_bin = rand_float() * total_sum;
        float running_sum = 0.0f;
        int sampled_t = 0, sampled_p = 0;
        bool found = false;

        for (int t = 0; t < theta_steps && !found; ++t)
        {
            for (int p = 0; p < phi_steps; ++p)
            {
                running_sum += local_pdf[t][p];
                if (r_bin <= running_sum)
                {
                    sampled_t = t;
                    sampled_p = p;
                    found = true;
                    break;
                }
            }
        }

        float u_jitter = rand_float();
        float v_jitter = rand_float();
        vec3 ray_dir = ActiveMapping::sample_bin(sampled_t, sampled_p, u_jitter, v_jitter);
        sampled_dirs.push_back(unit_vector(ray_dir));
    }

    // 3. Compute counts for PLY file header
    int actual_samples = static_cast<int>(sampled_dirs.size());
    int scene_vertices_count = static_cast<int>(scene_obj.mesh.size() * 3);
    int scene_faces_count = static_cast<int>(scene_obj.mesh.size());

    int sphere_vertices_count = theta_steps * phi_steps;
    int sphere_faces_count = (theta_steps - 1) * phi_steps * 2;

    int rays_vertices_count = actual_samples * 4;
    int rays_faces_count = actual_samples * 2;

    int total_vertices = scene_vertices_count + sphere_vertices_count + rays_vertices_count;
    int total_faces = scene_faces_count + sphere_faces_count + rays_faces_count;

    std::ofstream file(filename);
    if (!file.is_open())
        return;

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << total_vertices << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "element face " << total_faces << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";

    // PASS A: SCENE GEOMETRY (Gray)
    for (const auto &tri : scene_obj.mesh)
    {
        file << tri.v0.x() << " " << tri.v0.y() << " " << tri.v0.z() << " 180 180 180\n";
        file << tri.v1.x() << " " << tri.v1.y() << " " << tri.v1.z() << " 180 180 180\n";
        file << tri.v2.x() << " " << tri.v2.y() << " " << tri.v2.z() << " 180 180 180\n";
    }

    // PASS B: HEATMAP SPHERE (Red for target bin, Dark Blue for empty bins)
    for (int t = 0; t < theta_steps; ++t)
    {
        for (int p = 0; p < phi_steps; ++p)
        {
            vec3 outward_dir = unit_vector(ActiveMapping::sample_bin(t, p, 0.5f, 0.5f));
            vec3 vertex_pos = position + outward_dir * sphere_radius;

            int r = 20, g = 20, b = 60; // Default dark blue for inactive bins
            if (t == test_t && p == test_p)
            {
                r = 255;
                g = 0;
                b = 0; // Bright RED hotspot for forced target bin
            }

            file << vertex_pos.x() << " " << vertex_pos.y() << " " << vertex_pos.z() << " "
                 << r << " " << g << " " << b << "\n";
        }
    }

    // PASS C: SAMPLED RAY RIBBONS (Bright Yellow)
    for (int i = 0; i < actual_samples; ++i)
    {
        vec3 dir = sampled_dirs[i];
        vec3 ray_start = position;
        vec3 ray_end = position + dir * (sphere_radius * 2.5f);

        vec3 up = (std::abs(dir.y()) > 0.9f) ? vec3(1, 0, 0) : vec3(0, 1, 0);
        vec3 right = unit_vector(cross(dir, up)) * (sphere_radius * 0.03f);

        file << (ray_start - right).x() << " " << (ray_start - right).y() << " " << (ray_start - right).z() << " 255 255 0\n";
        file << (ray_start + right).x() << " " << (ray_start + right).y() << " " << (ray_start + right).z() << " 255 255 0\n";
        file << (ray_end - right).x() << " " << (ray_end - right).y() << " " << (ray_end - right).z() << " 255 255 0\n";
        file << (ray_end + right).x() << " " << (ray_end + right).y() << " " << (ray_end + right).z() << " 255 255 0\n";
    }

    // FACES: SCENE
    for (size_t i = 0; i < scene_obj.mesh.size(); ++i)
    {
        int base_idx = static_cast<int>(i * 3);
        file << "3 " << base_idx << " " << (base_idx + 1) << " " << (base_idx + 2) << "\n";
    }

    // FACES: SPHERE
    int sphere_base_idx = scene_vertices_count;
    for (int t = 0; t < theta_steps - 1; ++t)
    {
        for (int p = 0; p < phi_steps; ++p)
        {
            int p_next = (p + 1) % phi_steps;
            int v1 = sphere_base_idx + (t * phi_steps + p);
            int v2 = sphere_base_idx + (t * phi_steps + p_next);
            int v3 = sphere_base_idx + ((t + 1) * phi_steps + p);
            int v4 = sphere_base_idx + ((t + 1) * phi_steps + p_next);

            file << "3 " << v1 << " " << v3 << " " << v2 << "\n";
            file << "3 " << v2 << " " << v3 << " " << v4 << "\n";
        }
    }

    // FACES: RAY BUNDLES
    int rays_base_idx = scene_vertices_count + sphere_vertices_count;
    for (int i = 0; i < actual_samples; ++i)
    {
        int r_idx = rays_base_idx + (i * 4);
        file << "3 " << r_idx << " " << (r_idx + 1) << " " << (r_idx + 2) << "\n";
        file << "3 " << (r_idx + 1) << " " << (r_idx + 3) << " " << (r_idx + 2) << "\n";
    }

    file.close();
    std::cout << "--> [Sanity Test] Exported bin test (" << test_t << ", " << test_p
              << ") to " << filename << std::endl;
}

// void VoxelGrid::export_pdf_lobe_to_obj(const std::string &filename, const vec3 &position, const vec3 &normal, VoxelGrid &grid){}

void VoxelGrid::test_heatmap_directions()
{
    VoxelGrid grid;
    // Set up a clean 1x1x1 bounding box centered at the origin
    grid.initialize(point3(-1, -1, -1), point3(1, 1, 1), 1, 1, 1);

    // Use a point exactly at the origin for the lookup
    point3 test_p(0, 0, 0);
    color bright_radiance(100.0f, 100.0f, 100.0f);

    // --- TEST 1: Simulate light coming from straight UP (+Y) ---
    // If a ray went straight UP (+Y), record_radiance negates it to (-Y).
    // Let's test what your code considers "incoming light direction" when a ray goes UP.
    vec3 ray_up(0, 1, 0);
    grid.record_radiance(test_p, ray_up, bright_radiance);
    grid.finalize_and_build_distributions();
    grid.export_voxel_heatmap_ppm(test_p, "test_ray_up.ppm");

    // DOWN
    vec3 ray_down(0, -1, 0);
    grid.record_radiance(test_p, ray_down, bright_radiance);
    grid.finalize_and_build_distributions();
    grid.export_voxel_heatmap_ppm(test_p, "test_ray_down.ppm");

    // Clear grid for next test
    // grid.initialize(point3(-1, -1, -1), point3(1, 1, 1), 1, 1, 1);

    // --- TEST 2: Simulate ray going FORWARD into the room (-Z) ---
    vec3 ray_forward(0, 0, -1);
    grid.record_radiance(test_p, ray_forward, bright_radiance);
    grid.finalize_and_build_distributions();
    grid.export_voxel_heatmap_ppm(test_p, "test_ray_forward.ppm");

    // BACKWARD
    vec3 ray_backward(0, 0, 1);
    grid.record_radiance(test_p, ray_backward, bright_radiance);
    grid.finalize_and_build_distributions();
    grid.export_voxel_heatmap_ppm(test_p, "test_ray_backward.ppm");

    // Clear grid for next test
    // grid.initialize(point3(-1, -1, -1), point3(1, 1, 1), 1, 1, 1);

    // --- TEST 3: Simulate ray going RIGHT (+X) ---
    vec3 ray_right(1, 0, 0);
    grid.record_radiance(test_p, ray_right, bright_radiance);
    grid.finalize_and_build_distributions();
    grid.export_voxel_heatmap_ppm(test_p, "test_ray_right.ppm");

    // LEFT
    vec3 ray_left(-1, 0, 0);
    grid.record_radiance(test_p, ray_left, bright_radiance);
    grid.finalize_and_build_distributions();
    grid.export_voxel_heatmap_ppm(test_p, "test_ray_left.ppm");
}