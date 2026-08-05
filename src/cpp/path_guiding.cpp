#include "../hpp/path_guiding.hpp"
#include "../hpp/mesh_loader.hpp"
#include <fstream>

namespace SphereMapping
{

    // ====================================================
    // 1. LATITUDE-LONGITUDE (Your Current Baseline)
    // ====================================================
    struct LatLong
    {
        static inline void dir_to_bin(const vec3 &dir, int &out_t, int &out_p)
        {
            vec3 v = -unit_vector(dir);

            float theta = std::acos(std::clamp(v.y(), -1.0f, 1.0f));
            out_t = std::clamp(static_cast<int>((theta / M_PI) * 8.0f), 0, 7);

            if (std::abs(v.x()) < 1e-4f && std::abs(v.z()) < 1e-4f)
            {
                out_p = 8;
                return;
            }

            float phi = std::atan2(v.z(), v.x());
            phi += 1.5f * M_PI;

            while (phi < 0.0f)
                phi += 2.0f * M_PI;
            while (phi >= 2.0f * M_PI)
                phi -= 2.0f * M_PI;

            out_p = std::clamp(static_cast<int>((phi / (2.0f * M_PI)) * 16.0f), 0, 15);
        }

        static inline vec3 sample_bin_linear(int t, int p, float u1, float u2)
        {
            float delta_theta = M_PI / 8.0f;
            float delta_phi = (2.0f * M_PI) / 16.0f;

            float phi_min = p * delta_phi;
            float phi_max = (p + 1) * delta_phi;
            float theta_min = t * delta_theta;
            float theta_max = (t + 1) * delta_theta;

            float jittered_phi = phi_min + u1 * (phi_max - phi_min);

            // ====================================================
            // LINEAR JITTERED THETA (Biased baseline)
            // Directly interpolating the angles rather than the cosines
            // ====================================================
            float jittered_theta = theta_min + u2 * (theta_max - theta_min);

            float dx = std::sin(jittered_theta) * std::cos(jittered_phi);
            float dy = std::cos(jittered_theta);
            float dz = std::sin(jittered_theta) * std::sin(jittered_phi);
            return unit_vector(vec3(dx, dy, dz));
        }

        static inline vec3 sample_bin(int t, int p, float u1, float u2)
        {
            float delta_theta = M_PI / 8.0f;
            float delta_phi = (2.0f * M_PI) / 16.0f;

            float phi_min = p * delta_phi;
            float phi_max = (p + 1) * delta_phi;
            float theta_min = t * delta_theta;
            float theta_max = (t + 1) * delta_theta;

            // Jitter phi within bin
            float jittered_phi = phi_min + u1 * (phi_max - phi_min);
            // Match VoxelGrid offset
            jittered_phi -= 1.5f * M_PI;

            // Cosine-scaled theta jittering
            float cos_theta_min = std::cos(theta_min);
            float cos_theta_max = std::cos(theta_max);
            float jittered_cos_theta = cos_theta_min + u2 * (cos_theta_max - cos_theta_min);
            float jittered_theta = std::acos(std::clamp(jittered_cos_theta, -1.0f, 1.0f));

            float dx = std::sin(jittered_theta) * std::cos(jittered_phi);
            float dy = std::cos(jittered_theta);
            float dz = std::sin(jittered_theta) * std::sin(jittered_phi);

            return unit_vector(vec3(dx, dy, dz)); // Invert vector to match outgoing direction convention?
        }

        static inline float bin_solid_angle(int t, int p)
        {
            float delta_theta = M_PI / 8.0f;
            float delta_phi = (2.0f * M_PI) / 16.0f;
            // Solid angle of a lat-long bin: d_phi * (cos(theta_min) - cos(theta_max))
            return delta_phi * (std::cos(t * delta_theta) - std::cos((t + 1) * delta_theta));
        }
    };

    // ====================================================
    // 2. EQUAL-AREA CYLINDRICAL (An Easy Upgrade)
    // ====================================================
    struct EqualArea
    {
        static inline void dir_to_bin(const vec3 &dir, int &out_t, int &out_p)
        {
            // Map Y directly to [-1, 1] uniformly (no acos needed for elevation!)
            float cos_theta = std::clamp(dir.y(), -1.0f, 1.0f);
            float phi = std::atan2(-dir.z(), dir.x()) + 1.5f * M_PI;
            if (phi < 0.0f)
                phi += 2.0f * M_PI;

            // Map cos_theta [-1, 1] linearly to 8 bins
            float norm_cos = (cos_theta + 1.0f) * 0.5f; // [0, 1]
            out_t = std::clamp(static_cast<int>(norm_cos * 8.0f), 0, 7);
            out_p = std::clamp(static_cast<int>(phi / ((2.0f * M_PI) / 16.0f)), 0, 15);
        }

        static inline vec3 sample_bin(int t, int p, float u1, float u2)
        {
            float delta_phi = (2.0f * M_PI) / 16.0f;
            float phi_min = p * delta_phi;
            float phi_max = (p + 1) * delta_phi;
            float jittered_phi = phi_min + u1 * (phi_max - phi_min);

            // Uniformly slice in cosine space [-1, 1]
            float cos_min = -1.0f + (t / 8.0f) * 2.0f;
            float cos_max = -1.0f + ((t + 1) / 8.0f) * 2.0f;
            float jittered_cos = cos_min + u2 * (cos_max - cos_min);

            float sin_theta = std::sqrt(std::max(0.0f, 1.0f - jittered_cos * jittered_cos));
            return unit_vector(vec3(
                sin_theta * std::cos(jittered_phi),
                jittered_cos,
                sin_theta * std::sin(jittered_phi)));
        }

        static inline float bin_solid_angle(int t, int p)
        {
            // All bins have the exact same solid angle!
            // Total sphere area (4 * PI) divided by total number of bins (128)
            return (4.0f * M_PI) / 128.0f;
        }
    };

    // Octahedral TODO
    // define a flat float bins[12 * 12] array (or $10 \times 10$) inside your Voxel class.
    // when a ray hits, map its direction to $(u, v)$ using direction_to_uv, scale it by your grid resolution (e.g., $12$), and cast to integers to find your 2D bin index)
    struct Octahedral
    {
        // Helper to get the sign of a float (+1 or -1)
        inline float sign(float x)
        {
            return (x >= 0.0f) ? 1.0f : -1.0f;
        }

        // 1. Convert a 3D direction vector into 2D [0, 1] coordinates
        inline void direction_to_uv(const vec3 &dir, float &u, float &v)
        {
            vec3 d = unit_vector(dir);

            // Project onto the L1-norm octahedron plane
            float l1_norm = std::abs(d.x()) + std::abs(d.y()) + std::abs(d.z());
            float x = d.x() / l1_norm;
            float z = d.z() / l1_norm;

            // If we are in the lower hemisphere (Y < 0), fold the outer corners
            if (d.y() < 0.0f)
            {
                float temp_x = x;
                x = (1.0f - std::abs(z)) * sign(temp_x);
                z = (1.0f - std::abs(temp_x)) * sign(z);
            }

            // Map from [-1, 1]^2 to [0, 1]^2
            u = x * 0.5f + 0.5f;
            v = z * 0.5f + 0.5f;
        }

        // 2. Convert 2D [0, 1] coordinates back into a 3D direction vector
        inline vec3 uv_to_direction(float u, float v)
        {
            // Map back to [-1, 1]^2
            float x = u * 2.0f - 1.0f;
            float z = v * 2.0f - 1.0f;

            // Reconstruct the Y component (height) using the L1-norm boundary
            float y = 1.0f - std::abs(x) - std::abs(z);

            vec3 dir(x, y, z);

            // Unfold the corners if Y is negative
            if (y < 0.0f)
            {
                float temp_x = dir.x();
                dir[0] = (1.0f - std::abs(dir.z())) * sign(temp_x);
                dir[2] = (1.0f - std::abs(temp_x)) * sign(dir.z());
            }

            return unit_vector(dir);
        }

        // 3. Jittered bin sampling
        // (Assuming you updated your grid configuration to be square, e.g., N x N)
        inline vec3 sample_bin(int tx, int ty, int grid_size, float u1, float u2)
        {
            // Calculate the [0, 1] boundaries of the bin
            float delta = 1.0f / static_cast<float>(grid_size);

            float u_min = tx * delta;
            float v_min = ty * delta;

            // Jitter within the bin's 2D square bounds
            float jittered_u = u_min + u1 * delta;
            float jittered_v = v_min + u2 * delta;

            return uv_to_direction(jittered_u, jittered_v);
        }
    };
}

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
    // Cancel out the record_radiance minus sign to work with standard ray directions
    vec3 v = unit_vector(dir); // -?

    // --- 1. VERTICAL AXIS (Rows: 0 is Top, 7 is Bottom) ---
    float theta = std::acos(std::clamp(v.y(), -1.0f, 1.0f));
    out_theta_idx = std::clamp(static_cast<int>((theta / M_PI) * 8.0f), 0, 7);

    // --- 2. HORIZONTAL AXIS (Columns: 0-15) ---
    // If it's a perfectly vertical ray (Up/Down), force it to the horizontal center (Column 8)
    if (std::abs(v.x()) < 1e-4f && std::abs(v.z()) < 1e-4f)
    {
        out_phi_idx = 8;
        return;
    }

    // Standard atan2 returns -PI to +PI.
    // In our desired system: Forward (-Z) = Center, Right (+X) = Right Side, Left (-X) = Left Side
    float phi = std::atan2(v.z(), v.x());

    // To center Forward (-Z) perfectly at Column 8, we apply a 3*PI/2 shift
    phi += 1.5f * M_PI;

    // Wrap into a clean 0 to 2*PI range
    while (phi < 0.0f)
        phi += 2.0f * M_PI;
    while (phi >= 2.0f * M_PI)
        phi -= 2.0f * M_PI;

    out_phi_idx = std::clamp(static_cast<int>((phi / (2.0f * M_PI)) * 16.0f), 0, 15);
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

    int max_t = -1, max_p = -1;
    float max_pdf_val = -1.0f;

    // Only scan for the max PDF bin if we are actively debugging
    if (g_debug_print_rays)
    {
        for (int t = 0; t < 8; ++t)
        {
            for (int p = 0; p < 16; ++p)
            {
                if (vox.guiding_pdf[t][p] > max_pdf_val)
                {
                    max_pdf_val = vox.guiding_pdf[t][p];
                    max_t = t;
                    max_p = p;
                }
            }
        }

        static bool printed_once = false;
        if (!printed_once && max_pdf_val > 0.0f)
        {
            vec3 expected_dir = SphereMapping::LatLong::sample_bin(max_t, max_p, 0.5f, 0.5f);
            std::cout << "\n=========================================================\n"
                      << "[DEBUG] Max PDF Bin for voxel is (" << max_t << ", " << max_p << ")"
                      << " with value: " << max_pdf_val << "\n"
                      << "[DEBUG] Bin center direction (bin_to_vector): ("
                      << expected_dir.x() << ", " << expected_dir.y() << ", " << expected_dir.z() << ")\n"
                      << "=========================================================\n"
                      << std::endl;
            printed_once = true;
        }
    }

    // 1. Compute surface-aligned weights for all 128 bins
    for (int t = 0; t < 8; t++)
    {
        for (int p = 0; p < 16; p++)
        {
            vec3 dir = SphereMapping::LatLong::sample_bin(t, p, 0.5f, 0.5f); // HERE (sphere mapping 1)
            vec3 wo = unit_vector(dir);

            float cos_s = dot(normal, wo);

            if (cos_s > 0.0f)
            {
                local_pdf[t][p] = vox.guiding_pdf[t][p]; // + 0.01f) // * cos_s;
            }
            else
            {
                local_pdf[t][p] = 0.0f;
            }
            total_sum += local_pdf[t][p];
        }
    }

    // FALLBACK: Uniform hemisphere sampling if total_sum is zero
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

    // 2. Select a discrete bin via CDF search
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

    // 3. Jitter within the selected bin to get continuous direction
    float u_jitter = rand_float();
    float v_jitter = rand_float();
    vec3 ray_dir = SphereMapping::LatLong::sample_bin(sampled_t, sampled_p, u_jitter, v_jitter); // HERE (sphere mapping 2)
    vec3 wo = unit_vector(ray_dir);

    // 4. Calculate exact PDF for Monte Carlo
    float p_discrete = local_pdf[sampled_t][sampled_p] / total_sum;

    float solid_angle = get_bin_solid_angle(sampled_t, sampled_p);

    out_pdf_guiding = p_discrete / solid_angle;

    // =========================================================================
    // CONTROLLED DEBUG LOG: Only prints when g_debug_print_rays is true
    // =========================================================================
    if (g_debug_print_rays)
    {
        std::cout << "[DIAGNOSTIC RAY] Picked Bin (" << sampled_t << ", " << sampled_p << ") with probabiliy " << local_pdf[sampled_t][sampled_p]
                  << "-> Generated wo: (" << wo.x() << ", " << wo.y() << ", " << wo.z() << ")"
                  << std::endl;
    }

    return wo;
}

vec3 VoxelGrid::sample_guided_direction_interpolated(const point3 &hit_p, const vec3 &normal, float &out_pdf_guiding)
{
    // 1. Compute Trilinear Grid Coordinates & Indices
    float tx = (hit_p.x() - bounds_min.x()) / (bounds_max.x() - bounds_min.x()) * res_x - 0.5f;
    float ty = (hit_p.y() - bounds_min.y()) / (bounds_max.y() - bounds_min.y()) * res_y - 0.5f;
    float tz = (hit_p.z() - bounds_min.z()) / (bounds_max.z() - bounds_min.z()) * res_z - 0.5f;

    int ix0 = std::clamp(static_cast<int>(std::floor(tx)), 0, res_x - 1);
    int iy0 = std::clamp(static_cast<int>(std::floor(ty)), 0, res_y - 1);
    int iz0 = std::clamp(static_cast<int>(std::floor(tz)), 0, res_z - 1);

    int ix1 = std::clamp(ix0 + 1, 0, res_x - 1);
    int iy1 = std::clamp(iy0 + 1, 0, res_y - 1);
    int iz1 = std::clamp(iz0 + 1, 0, res_z - 1);

    float fx = tx - std::floor(tx);
    float fy = ty - std::floor(ty);
    float fz = tz - std::floor(tz);

    // 2. Compute the 8 Trilinear Weights
    float w[8];
    w[0] = (1.0f - fx) * (1.0f - fy) * (1.0f - fz); // 000
    w[1] = fx * (1.0f - fy) * (1.0f - fz);          // 100
    w[2] = (1.0f - fx) * fy * (1.0f - fz);          // 010
    w[3] = fx * fy * (1.0f - fz);                   // 110
    w[4] = (1.0f - fx) * (1.0f - fy) * fz;          // 001
    w[5] = fx * (1.0f - fy) * fz;                   // 101
    w[6] = (1.0f - fx) * fy * fz;                   // 011
    w[7] = fx * fy * fz;                            // 111

    const int corner_indices[8][3] = {
        {ix0, iy0, iz0}, {ix1, iy0, iz0}, {ix0, iy1, iz0}, {ix1, iy1, iz0}, {ix0, iy0, iz1}, {ix1, iy0, iz1}, {ix0, iy1, iz1}, {ix1, iy1, iz1}};

    // 2b. Plane test: Filter out corners behind the surface and compute valid spatial weight sum
    float total_spatial_weight = 0.0f;
    for (int k = 0; k < 8; ++k)
    {
        point3 corner_p = get_voxel_center(corner_indices[k][0], corner_indices[k][1], corner_indices[k][2]);
        vec3 dir_to_corner = corner_p - hit_p;

        if (dot(normal, dir_to_corner) < 0.0f)
        {
            w[k] = 0.0f;
        }
        total_spatial_weight += w[k];
    }

    // 3. Stochastically choose ONE voxel corner based on filtered spatial weights
    int chosen_corner = 0;

    if (total_spatial_weight > 0.0f)
    {
        float r_spatial = rand_float() * total_spatial_weight;
        float accum_w = 0.0f;

        for (int k = 0; k < 8; ++k)
        {
            accum_w += w[k];
            if (r_spatial <= accum_w)
            {
                chosen_corner = k;
                break;
            }
        }
    }
    else
    {
        // Fallback to nearest neighbor corner if all 8 are behind plane
        int nearest_x = std::clamp(static_cast<int>(std::round(tx)), 0, res_x - 1);
        int nearest_y = std::clamp(static_cast<int>(std::round(ty)), 0, res_y - 1);
        int nearest_z = std::clamp(static_cast<int>(std::round(tz)), 0, res_z - 1);

        for (int k = 0; k < 8; ++k)
        {
            if (corner_indices[k][0] == nearest_x &&
                corner_indices[k][1] == nearest_y &&
                corner_indices[k][2] == nearest_z)
            {
                chosen_corner = k;
                break;
            }
        }
    }

    // 4. Extract chosen voxel index
    int cx = corner_indices[chosen_corner][0];
    int cy = corner_indices[chosen_corner][1];
    int cz = corner_indices[chosen_corner][2];
    int idx = cx + res_x * (cy + res_y * cz);
    const Voxel &vox = voxels[idx];

    // 5. Build hemisphere CDF for the chosen voxel
    float local_pdf[8][16];
    float total_sum = 0.0f;

    for (int t = 0; t < 8; t++)
    {
        for (int p = 0; p < 16; p++)
        {
            vec3 dir = SphereMapping::LatLong::sample_bin(t, p, 0.5f, 0.5f);
            vec3 wo_bin = unit_vector(dir);
            float cos_s = dot(normal, wo_bin);

            if (cos_s > 0.0f)
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

    // FALLBACK: Uniform cosine-weighted hemisphere sampling if total_sum is zero or unlearned
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

    // 6. Select a bin inside the chosen voxel via CDF search
    float r_bin = rand_float() * total_sum;
    float running_sum = 0.0f;
    int sampled_t = 0, sampled_p = 0;

    bool found = false;
    for (int t = 0; t < 8 && !found; t++)
    {
        for (int p = 0; p < 16; p++)
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

    // 7. Continuous direction jitter within selected bin
    float u_jitter = rand_float();
    float v_jitter = rand_float();
    vec3 ray_dir = SphereMapping::LatLong::sample_bin(sampled_t, sampled_p, u_jitter, v_jitter);
    vec3 wo = unit_vector(ray_dir);

    // 8. Evaluate true interpolated PDF via O(1) lookups
    out_pdf_guiding = get_guided_pdf_interpolated(hit_p, normal, wo);

    return wo;
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

        float sum_pdf = 0.0f;
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

                sum_pdf += vox.guiding_pdf[t][p];
                vox.spherical_bins[t][p] = 0.0f; // Clear training accumulator
            }
        }

        // Cache full-sphere sum (which will be 1.0f for all voxels)
        vox.total_weight = sum_pdf;
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
            << (16 * scale) << " " << (8 * scale) << "\n255\n";
        for (int i = 0; i < (16 * scale) * (8 * scale); i++)
        {
            out << "15 15 40\n"; // Dark Slate Blue for out-of-bounds/empty
        }
        out.close();
        return;
    }

    // 2. SURGICAL LOOKUP: Get the exact voxel
    Voxel &vox = get_voxel(world_p);

    float max_val = 0.0f;
    for (int t = 0; t < 8; t++)
    {
        for (int p = 0; p < 16; p++)
        {
            max_val = std::max(max_val, vox.guiding_pdf[t][p]);
        }
    }

    // If the voxel has never been visited/trained, max_val stays 0
    bool is_unvisited = (max_val <= 1e-6f);
    if (is_unvisited)
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

    for (int t = 0; t < 8; t++)
    {
        for (int p = 0; p < 16; p++)
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

/* void VoxelGrid::export_voxel_heatmap_ppm2(const point3 &world_p, const std::string &filename)
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
} */

float VoxelGrid::get_guided_pdf(const point3 &world_p, const vec3 &normal, const vec3 &dir) const
{
    // 1. Ray direction must be in the positive hemisphere relative to surface normal
    float cos_dir = dot(normal, dir);
    if (cos_dir <= 0.0f)
        return 0.0f;

    int vox_idx = get_voxel_index(world_p);
    const Voxel &vox = voxels[vox_idx];

    // Find target bin (Ensure vector_to_bin uses positive 'dir'!)
    int target_t = 0, target_p = 0;
    vector_to_bin(dir, target_t, target_p);

    float total_sum = 0.0f;
    float target_weight = 0.0f;

    // 2. Compute total_sum using exact bin-center directions (matching sample_guided_direction)
    for (int t = 0; t < 8; t++)
    {
        for (int p = 0; p < 16; p++)
        {
            // Sample the exact bin center using the LatLong mapping (0.5, 0.5)
            vec3 bin_center_dir = SphereMapping::LatLong::sample_bin(t, p, 0.5f, 0.5f);
            vec3 wo = unit_vector(bin_center_dir);

            float cos_s = dot(normal, wo);
            if (cos_s > 0.0f)
            {
                float weight = vox.guiding_pdf[t][p]; //  + 0.01f) * cos_s;

                total_sum += weight;

                // Match the target bin
                if (t == target_t && p == target_p)
                {
                    // std::cout << "WEIGHT ASSIGNED" << std::endl;
                    target_weight = weight;
                }
            }
        }
    }

    if (total_sum <= 0.0f || target_weight <= 0.0f)
    {
        // std::cout << "sum problem " << total_sum << ", " << target_weight << std::endl;
        return 0.0f;
    }

    // 3. Compute discrete probability mass
    float discrete_prob = target_weight / total_sum;

    // 4. Convert discrete probability to continuous solid-angle PDF density (sr^-1)
    float solid_angle = get_bin_solid_angle(target_t, target_p);

    return discrete_prob / solid_angle;
}

float VoxelGrid::get_guided_pdf_interpolated(const point3 &world_p, const vec3 &normal, const vec3 &dir) const
{
    // 1. Ray direction must be in the positive hemisphere relative to surface normal
    float cos_target = dot(normal, dir);
    if (cos_target <= 0.0f)
        return 0.0f;

    // 2. Find target bin ONCE for all 8 corners
    int target_t = 0, target_p = 0;
    vector_to_bin(dir, target_t, target_p);

    float solid_angle = get_bin_solid_angle(target_t, target_p);
    if (solid_angle <= 0.0f)
        return 0.0f;

    // 3. Compute trilinear coordinates
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

    // 4. Calculate standard trilinear spatial weights w_k
    float w[8];
    w[0] = (1.0f - fx) * (1.0f - fy) * (1.0f - fz); // 000
    w[1] = fx * (1.0f - fy) * (1.0f - fz);          // 100
    w[2] = (1.0f - fx) * fy * (1.0f - fz);          // 010
    w[3] = fx * fy * (1.0f - fz);                   // 110
    w[4] = (1.0f - fx) * (1.0f - fy) * fz;          // 001
    w[5] = fx * (1.0f - fy) * fz;                   // 101
    w[6] = (1.0f - fx) * fy * fz;                   // 011
    w[7] = fx * fy * fz;                            // 111

    const int corner_indices[8][3] = {
        {ix0, iy0, iz0}, {ix1, iy0, iz0}, {ix0, iy1, iz0}, {ix1, iy1, iz0}, {ix0, iy0, iz1}, {ix1, iy0, iz1}, {ix0, iy1, iz1}, {ix1, iy1, iz1}};

    // 5. Plane test: Filter out corners behind the surface and re-normalize weights
    float total_spatial_weight = 0.0f;
    for (int k = 0; k < 8; ++k)
    {
        point3 corner_p = get_voxel_center(corner_indices[k][0], corner_indices[k][1], corner_indices[k][2]);
        vec3 dir_to_corner = corner_p - world_p;

        // Zero out weight if the voxel corner lies behind the tangent plane
        if (dot(normal, dir_to_corner) < 0.0f)
        {
            w[k] = 0.0f;
        }
        total_spatial_weight += w[k];
    }

    // Fallback: If all corners get rejected, fall back to nearest voxel center
    if (total_spatial_weight <= 0.0f)
    {
        int nearest_x = std::clamp(static_cast<int>(std::round(tx)), 0, res_x - 1);
        int nearest_y = std::clamp(static_cast<int>(std::round(ty)), 0, res_y - 1);
        int nearest_z = std::clamp(static_cast<int>(std::round(tz)), 0, res_z - 1);

        int idx = nearest_x + res_x * (nearest_y + res_y * nearest_z);
        const Voxel &vox = voxels[idx];
        if (vox.total_weight <= 0.0f)
            return 0.0f;

        float p_discrete = vox.guiding_pdf[target_t][target_p] / vox.total_weight;
        return p_discrete / solid_angle;
    }

    // 6. Evaluate node PDFs and calculate weighted sum
    auto get_node_pdf = [&](int x, int y, int z)
    {
        int idx = x + res_x * (y + res_y * z);
        const Voxel &vox = voxels[idx];

        if (vox.total_weight <= 0.0f)
            return 0.0f;

        float p_discrete = vox.guiding_pdf[target_t][target_p] / vox.total_weight;
        return p_discrete / solid_angle;
    };

    float final_pdf = 0.0f;
    for (int k = 0; k < 8; ++k)
    {
        if (w[k] > 0.0f)
        {
            float normalized_wk = w[k] / total_spatial_weight;
            float corner_pdf = get_node_pdf(corner_indices[k][0], corner_indices[k][1], corner_indices[k][2]);
            final_pdf += normalized_wk * corner_pdf;
        }
    }

    return final_pdf;
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

void VoxelGrid::export_raw_pdf_heatmap_sphere_to_ply(const std::string &filename,
                                                     const vec3 &position,
                                                     VoxelGrid &grid)
{
    int theta_steps = 64;
    int phi_steps = 128;
    float sphere_radius = 1.0f;

    // Grab the exact local voxel cell
    Voxel &vox = grid.get_voxel(position);

    // =========================================================================
    // 1. FIND MAX RAW PDF VALUE IN THE VOXEL FOR NORMALIZATION
    // =========================================================================
    float max_val = 1e-4f;
    for (int t = 0; t < 8; ++t)
    {
        for (int p = 0; p < 16; ++p)
        {
            max_val = std::max(max_val, vox.guiding_pdf[t][p]);
        }
    }

    int total_vertices = theta_steps * phi_steps;
    int total_faces = (theta_steps - 1) * phi_steps * 2;

    std::ofstream file(filename);
    if (!file.is_open())
        return;

    // PLY Header
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
    // 2. VERTEX & COLOR GENERATION (UNCONSTRAINED 360° SPHERE)
    // =========================================================================
    for (int t = 0; t < theta_steps; ++t)
    {
        float theta = (float)t * M_PI / (float)(theta_steps - 1);
        for (int p = 0; p < phi_steps; ++p)
        {
            float phi = (float)p * 2.0f * M_PI / (float)phi_steps;

            // Generate Y-up world direction vector
            float shifted_phi = phi - 1.5f * M_PI;
            vec3 dir(
                std::sin(theta) * std::cos(shifted_phi),
                std::cos(theta),
                std::sin(theta) * std::sin(shifted_phi));
            dir = unit_vector(dir);

            // Fetch the bin corresponding to this direction
            int bin_t = 0, bin_p = 0;
            vector_to_bin(dir, bin_t, bin_p);

            // Direct raw PDF query (no cosine masking, no hemisphere clamping)
            float raw_pdf = vox.guiding_pdf[bin_t][bin_p];
            float normalized_val = std::clamp(raw_pdf / max_val, 0.0f, 1.0f);

            // Convert to jet/rainbow heatmap color
            vec3 rgb_color = pdf_to_heatmap_color(normalized_val);

            // Uniform 3D Sphere geometry centered at position
            vec3 vertex_pos = position + dir * sphere_radius;

            int r = static_cast<int>(std::clamp(rgb_color.x() * 255.0f, 0.0f, 255.0f));
            int g = static_cast<int>(std::clamp(rgb_color.y() * 255.0f, 0.0f, 255.0f));
            int b = static_cast<int>(std::clamp(rgb_color.z() * 255.0f, 0.0f, 255.0f));

            file << vertex_pos.x() << " " << vertex_pos.y() << " " << vertex_pos.z() << " "
                 << r << " " << g << " " << b << "\n";
        }
    }

    // =========================================================================
    // 3. FACES
    // =========================================================================
    for (int t = 0; t < theta_steps - 1; ++t)
    {
        for (int p = 0; p < phi_steps; ++p)
        {
            int p_next = (p + 1) % phi_steps;

            int v1 = t * phi_steps + p;
            int v2 = t * phi_steps + p_next;
            int v3 = (t + 1) * phi_steps + p;
            int v4 = (t + 1) * phi_steps + p_next;

            file << "3 " << v1 << " " << v3 << " " << v2 << "\n";
            file << "3 " << v2 << " " << v3 << " " << v4 << "\n";
        }
    }

    file.close();
    std::cout << "--> [Success] Exported raw 360 heatmap sphere to " << filename << std::endl;
}

void VoxelGrid::export_scene_with_heatmap_and_ray_bundle_to_ply(
    const std::string &filename,
    const object &scene_obj,
    const vec3 &position,
    const vec3 &surface_normal,
    float sphere_radius,
    VoxelGrid &grid,
    int num_samples)
{
    constexpr int theta_steps = 8;
    constexpr int phi_steps = 16;

    Voxel &vox = grid.get_voxel(position);

    // 1. Sample N directions from guided sampler
    std::vector<vec3> sampled_dirs;
    sampled_dirs.reserve(num_samples);
    for (int i = 0; i < num_samples; ++i)
    {
        float dummy_pdf = 0.0f;
        vec3 dir = grid.sample_guided_direction_interpolated(position, surface_normal, dummy_pdf);
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
            vec3 bin_dir = unit_vector(SphereMapping::LatLong::sample_bin(t, p, 0.5f, 0.5f));
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

void VoxelGrid::export_pdf_lobe_to_obj(const std::string &filename, const vec3 &position, const vec3 &normal, VoxelGrid &grid)
{
    std::ofstream file(filename);
    if (!file.is_open())
        return;

    file << "# High Resolution Smooth Path Guiding Lobe Visualization\n";

    // High fidelity steps to render an organic, rounded balloon
    int theta_steps = 64;
    int phi_steps = 128;
    float scale = 4.0f;

    // Compute size of a single voxel cell
    float cell_size_x = (grid.bounds_max.x() - grid.bounds_min.x()) / grid.res_x;

    // CORE FIX: Shifting the lookup point 10% of a voxel cell outward along the surface normal.
    // When lookups happen exactly on a surface vertex, trilinear neighborhood gathering
    // grabs empty, untrained voxels behind walls/floors, causing jagged tier deformities.
    point3 safe_position = position + normal * (cell_size_x * 0.1f);

    // Safely look up if the local area is trained or completely unvisited
    Voxel &vox = grid.get_voxel(safe_position);
    float max_val = 0.0f;
    for (int t = 0; t < 8; t++)
    {
        for (int p = 0; p < 16; p++)
        {
            max_val = std::max(max_val, vox.guiding_pdf[t][p]);
        }
    }
    bool is_untrained = (max_val <= 0.00782f);

    // =========================================================================
    // VERTEX GENERATION LOOP (Smooth High-Res Interpolation Integration)
    // =========================================================================
    for (int t = 0; t < theta_steps; ++t)
    {
        float theta = (float)t * M_PI / (float)(theta_steps - 1);
        for (int p = 0; p < phi_steps; ++p)
        {
            float phi = (float)p * 2.0f * M_PI / (float)phi_steps;

            // Maps natively with your internal Y-Up binning scheme
            vec3 dir(
                std::sin(theta) * std::cos(phi),
                std::cos(theta),
                std::sin(theta) * std::sin(phi));

            float pdf = 0.0f;

            if (is_untrained)
            {
                // Flawless baseline sphere for untrained space
                pdf = 0.2f;
            }
            else
            {
                // Query your path-tracer's true smooth continuous interpolation math
                pdf = grid.get_guided_pdf_interpolated(safe_position, normal, dir);

                // Hemisphere clamping to make the back shell sit flat against the surface
                float cos_target = dot(normal, dir);
                if (cos_target <= 0.0f || pdf <= 1e-4f)
                {
                    pdf = 0.05f; // Flat base backing shell radius
                }
            }

            // Deform vertex position outward
            vec3 vertex_pos = position + dir * (pdf * scale);
            file << "v " << vertex_pos.x() << " " << vertex_pos.y() << " " << vertex_pos.z() << "\n";
        }
    }

    // =========================================================================
    // INDEXED FACE GENERATION LOOP
    // =========================================================================
    for (int t = 0; t < theta_steps - 1; ++t)
    {
        for (int p = 0; p < phi_steps; ++p)
        {
            int p_next = (p + 1) % phi_steps;

            int v1 = t * phi_steps + p + 1;
            int v2 = t * phi_steps + p_next + 1;
            int v3 = (t + 1) * phi_steps + p + 1;
            int v4 = (t + 1) * phi_steps + p_next + 1;

            file << "f " << v1 << " " << v3 << " " << v2 << "\n";
            file << "f " << v2 << " " << v3 << " " << v4 << "\n";
        }
    }

    file.close();
    std::cout << "--> [Success] Exported smooth continuous 3D PDF lobe mesh for point " << position << " to " << filename << std::endl;
}

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