#include "../hpp/path_guiding.hpp"
#include <fstream>

void VoxelGrid::initialize(point3 min_bounds, point3 max_bounds, int x_res, int y_res, int z_res)
{
    bounds_min = min_bounds;
    bounds_max = max_bounds;
    res_x = x_res;
    res_y = y_res;
    res_z = z_res;
    voxels.resize(res_x * res_y * res_z);
}

int VoxelGrid::get_voxel_index(const point3 &p) const
{
    float tx = (p.x() - bounds_min.x()) / (bounds_max.x() - bounds_min.x());
    float ty = (p.y() - bounds_min.y()) / (bounds_max.y() - bounds_min.y());
    float tz = (p.z() - bounds_min.z()) / (bounds_max.z() - bounds_min.z());

    int ix = std::clamp(static_cast<int>(tx * res_x), 0, res_x - 1);
    int iy = std::clamp(static_cast<int>(ty * res_y), 0, res_y - 1);
    int iz = std::clamp(static_cast<int>(tz * res_z), 0, res_z - 1);

    return ix + res_x * (iy + res_y * iz);
}

Voxel &VoxelGrid::get_voxel(const point3 &p)
{
    return voxels[get_voxel_index(p)];
}

void VoxelGrid::vector_to_bin(const vec3 &dir, int &out_theta_idx, int &out_phi_idx) const
{
    float theta = std::acos(std::clamp(dir.y(), -1.0f, 1.0f));
    float phi = std::atan2(dir.z(), dir.x());
    if (phi < 0.0f)
        phi += 2.0f * M_PI;

    out_theta_idx = std::clamp(static_cast<int>((theta / M_PI) * 8), 0, 7);
    out_phi_idx = std::clamp(static_cast<int>((phi / (2.0f * M_PI)) * 16), 0, 15);
}

vec3 VoxelGrid::bin_to_vector(int theta_idx, int phi_idx) const
{
    float theta = (theta_idx + 0.5f) * (M_PI / 8.0f);
    float phi = (phi_idx + 0.5f) * ((2.0f * M_PI) / 16.0f);

    return vec3(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
}

void VoxelGrid::record_radiance(const point3 &hit_p, const vec3 &ray_dir, const color &radiance)
{
    Voxel &vox = get_voxel(hit_p);

    vec3 incoming_light_dir = -unit_vector(ray_dir);

    int theta_idx, phi_idx;
    vector_to_bin(incoming_light_dir, theta_idx, phi_idx);

    float brightness = 0.2126f * radiance[0] + 0.7152f * radiance[1] + 0.0722f * radiance[2];

    vox.spherical_bins[theta_idx][phi_idx] += brightness;
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
    const Voxel &vox = get_voxel(hit_p);

    float local_pdf[8][16];
    float total_sum = 0.0f;

    // 1. Compute surface-aligned weights for all 128 bins reading from frozen guiding_pdf (FIX 1)
    for (int t = 0; t < 8; t++)
    {
        for (int p = 0; p < 16; p++)
        {
            vec3 wo = bin_to_vector(t, p);
            float cos_s = dot(normal, wo);

            if (cos_s > 0.0f)
            {
                local_pdf[t][p] = (vox.guiding_pdf[t][p] + 0.01f) * cos_s;
            }
            else
            {
                local_pdf[t][p] = 0.0f;
            }
            total_sum += local_pdf[t][p];
        }
    }

    if (total_sum <= 0.0f)
    {
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

        vec3 wo = unit_vector(x * u + y * v + z * w);

        out_pdf_guiding = std::max(0.0f, dot(normal, wo)) / M_PI;
        return wo;
    }

    // 2. Select a discrete bin
    float r = rand_float() * total_sum;
    float running_sum = 0.0f;
    int sampled_t = 0, sampled_p = 0;

    bool found = false;
    for (int t = 0; t < 8 && !found; t++)
    {
        for (int p = 0; p < 16; p++)
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

    // 3. Convert the discrete choice into a continuous solid angle PDF
    float discrete_prob = local_pdf[sampled_t][sampled_p] / total_sum;
    float solid_angle = get_bin_solid_angle(sampled_t, sampled_p);

    out_pdf_guiding = discrete_prob / solid_angle;

    // 4. Jitter within the selected bin space uniformly (FIX 3)
    float delta_theta = M_PI / 8.0f;
    float delta_phi = (2.0f * M_PI) / 16.0f;

    float random_theta = (sampled_t + rand_float()) * delta_theta;
    float random_phi = (sampled_p + rand_float()) * delta_phi;

    return vec3(
        std::sin(random_theta) * std::cos(random_phi),
        std::cos(random_theta),
        std::sin(random_theta) * std::sin(random_phi));
}

void VoxelGrid::finalize_and_build_distributions()
{
    for (auto &vox : voxels)
    {
        float total_radiance = 0.0f;

        for (int t = 0; t < 8; t++)
        {
            for (int p = 0; p < 16; p++)
            {
                total_radiance += vox.spherical_bins[t][p];
            }
        }

        for (int t = 0; t < 8; t++)
        {
            for (int p = 0; p < 16; p++)
            {
                if (total_radiance > 0.0f)
                {
                    vox.guiding_pdf[t][p] = vox.spherical_bins[t][p] / total_radiance;
                }
                else
                {
                    vox.guiding_pdf[t][p] = 1.0f / (8.0f * 16.0f);
                }

                vox.spherical_bins[t][p] = 0.0f;
            }
        }
    }
}

void VoxelGrid::export_voxel_heatmap_ppm(const point3 &world_p, const std::string &filename)
{
    Voxel &vox = get_voxel(world_p);

    float max_val = 0.0f;
    for (int t = 0; t < 8; t++)
    {
        for (int p = 0; p < 16; p++)
        {
            max_val = std::max(max_val, vox.guiding_pdf[t][p]);
        }
    }
    if (max_val == 0.0f)
        max_val = 1.0f;

    int scale = 32;
    int img_w = 16 * scale;
    int img_h = 8 * scale;

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

            int r = static_cast<int>(std::clamp(intensity * 1.5f, 0.0f, 1.0f) * 255);
            int g = static_cast<int>(std::clamp((intensity - 0.5f) * 2.0f, 0.0f, 1.0f) * 255);
            int b = static_cast<int>(std::clamp(1.0f - intensity, 0.0f, 1.0f) * 255);

            out << r << " " << g << " " << b << "\n";
        }
    }
}

void VoxelGrid::export_voxel_heatmap_ppm2(const point3 &world_p, const std::string &filename)
{
    int center_idx = get_voxel_index(world_p);
    Voxel *best_voxel = &voxels[center_idx];
    float highest_total = 0.0f;

    float step_x = (bounds_max.x() - bounds_min.x()) / res_x;
    float step_y = (bounds_max.y() - bounds_min.y()) / res_y;
    float step_z = (bounds_max.z() - bounds_min.z()) / res_z;

    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dz = -1; dz <= 1; dz++)
            {
                point3 search_p = world_p + vec3(dx * step_x, dy * step_y, dz * step_z);
                Voxel &current_vox = get_voxel(search_p);

                float current_total = 0.0f;
                for (int t = 0; t < 8; t++)
                {
                    for (int p = 0; p < 16; p++)
                    {
                        current_total += current_vox.guiding_pdf[t][p];
                    }
                }

                if (current_total > highest_total)
                {
                    highest_total = current_total;
                    best_voxel = &current_vox;
                }
            }
        }
    }

    float max_val = 0.0f;
    for (int t = 0; t < 8; t++)
    {
        for (int p = 0; p < 16; p++)
        {
            max_val = std::max(max_val, best_voxel->guiding_pdf[t][p]);
        }
    }
    if (max_val == 0.0f)
        max_val = 1.0f;

    int scale = 32;
    int img_w = 16 * scale;
    int img_h = 8 * scale;

    std::ofstream out(filename);
    out << "P3\n"
        << img_w << " " << img_h << "\n255\n";

    for (int y = 0; y < img_h; y++)
    {
        int t = y / scale;
        for (int x = 0; x < img_w; x++)
        {
            int p = x / scale;

            float intensity = best_voxel->guiding_pdf[t][p] / max_val;

            int r = static_cast<int>(std::clamp(intensity * 1.5f, 0.0f, 1.0f) * 255);
            int g = static_cast<int>(std::clamp((intensity - 0.5f) * 2.0f, 0.0f, 1.0f) * 255);
            int b = static_cast<int>(std::clamp(1.0f - intensity, 0.0f, 1.0f) * 255);

            out << r << " " << g << " " << b << "\n";
        }
    }
    std::cout << "Heatmap generated! Max radiance value found: " << highest_total << std::endl;
}

float VoxelGrid::get_guided_pdf(const point3 &world_p, const vec3 &normal, const vec3 &dir) const
{
    int vox_idx = get_voxel_index(world_p);
    const Voxel &vox = voxels[vox_idx];

    int target_t = 0, target_p = 0;
    vector_to_bin(dir, target_t, target_p);

    // Re-evaluate total_sum using guiding_pdf (FIX 1)
    float total_sum = 0.0f;
    for (int t = 0; t < 8; t++)
    {
        for (int p = 0; p < 16; p++)
        {
            vec3 wo = bin_to_vector(t, p);
            float cos_s = dot(normal, wo);
            if (cos_s > 0.0f)
            {
                total_sum += (vox.guiding_pdf[t][p] + 0.01f) * cos_s;
            }
        }
    }

    if (total_sum <= 0.0f)
        return 0.0f;

    float cos_target = dot(normal, dir);
    if (cos_target <= 0.0f)
        return 0.0f;

    float target_weight = (vox.guiding_pdf[target_t][target_p] + 0.01f) * cos_target;
    float discrete_prob = target_weight / total_sum;
    float solid_angle = get_bin_solid_angle(target_t, target_p);

    return discrete_prob / solid_angle;
}

float VoxelGrid::get_guided_pdf_interpolated(const point3 &world_p, const vec3 &normal, const vec3 &dir) const
{
    float tx = (world_p.x() - bounds_min.x()) / (bounds_max.x() - bounds_min.x()) * res_x - 0.5f;
    float ty = (world_p.y() - bounds_min.y()) / (bounds_max.y() - bounds_min.y()) * res_y - 0.5f;
    float tz = (world_p.z() - bounds_min.z()) / (bounds_max.z() - bounds_min.z()) * res_z - 0.5f;

    int ix0 = std::clamp(static_cast<int>(std::floor(tx)), 0, res_x - 1);
    int iy0 = std::clamp(static_cast<int>(std::floor(ty)), 0, res_y - 1);
    int iz0 = std::clamp(static_cast<int>(std::floor(tz)), 0, res_z - 1);

    int ix1 = std::clamp(ix0 + 1, 0, res_x - 1);
    int iy1 = std::clamp(iy0 + 1, 0, res_y - 1);
    int iz1 = std::clamp(iz0 + 1, 0, res_z - 1);

    float fx = tx - std::floor(tx);
    float fy = ty - std::floor(ty);
    float fz = tz - std::floor(tz);

    auto get_node_pdf = [&](int x, int y, int z)
    {
        int idx = z * res_y * res_x + y * res_x + x;

        int target_t = 0, target_p = 0;
        vector_to_bin(dir, target_t, target_p);

        // Re-evaluate using guiding_pdf (FIX 1)
        float total_sum = 0.0f;
        for (int t = 0; t < 8; t++)
        {
            for (int p = 0; p < 16; p++)
            {
                vec3 wo = bin_to_vector(t, p);
                float cos_s = dot(normal, wo);
                if (cos_s > 0.0f)
                {
                    total_sum += (voxels[idx].guiding_pdf[t][p] + 0.01f) * cos_s;
                }
            }
        }
        if (total_sum <= 0.0f)
            return 0.0f;
        float cos_target = dot(normal, dir);
        if (cos_target <= 0.0f)
            return 0.0f;

        return ((voxels[idx].guiding_pdf[target_t][target_p] + 0.01f) * cos_target / total_sum) / get_bin_solid_angle(target_t, target_p);
    };

    float c000 = get_node_pdf(ix0, iy0, iz0);
    float c100 = get_node_pdf(ix1, iy0, iz0);
    float c010 = get_node_pdf(ix0, iy1, iz0);
    float c110 = get_node_pdf(ix1, iy1, iz0);
    float c001 = get_node_pdf(ix0, iy0, iz1);
    float c101 = get_node_pdf(ix1, iy0, iz1);
    float c011 = get_node_pdf(ix0, iy1, iz1);
    float c111 = get_node_pdf(ix1, iy1, iz1);

    float c00 = c000 * (1.0f - fx) + c100 * fx;
    float c01 = c001 * (1.0f - fx) + c101 * fx;
    float c10 = c010 * (1.0f - fx) + c110 * fx;
    float c11 = c011 * (1.0f - fx) + c111 * fx;

    float c0 = c00 * (1.0f - fy) + c10 * fy;
    float c1 = c01 * (1.0f - fy) + c11 * fy;

    return c0 * (1.0f - fz) + c1 * fz;
}