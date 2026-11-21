#include "../hpp/spatial_hash_grid.hpp"

void SpatialHashGrid::initialize(float cell_size, point3 scene_min, point3 scene_max)
{
    this->voxel_size = cell_size;
    this->origin = scene_min;
    this->end = scene_max;
    grid.clear();

    max_res_x = std::max(0, static_cast<int>(std::ceil((end.x() - origin.x()) / voxel_size)));
    max_res_y = std::max(0, static_cast<int>(std::ceil((end.y() - origin.y()) / voxel_size)));
    max_res_z = std::max(0, static_cast<int>(std::ceil((end.z() - origin.z()) / voxel_size)));
}

CellKey SpatialHashGrid::get_cell_key(const point3 &p, const vec3 &n) const
{
    point3 offset = p - origin;

    int normal_id = 0;
    float ax = std::abs(n.x());
    float ay = std::abs(n.y());
    float az = std::abs(n.z());

    if (ax > ay && ax > az)
        normal_id = (n.x() > 0.0f) ? 0 : 1;
    else if (ay > az)
        normal_id = (n.y() > 0.0f) ? 2 : 3;
    else
        normal_id = (n.z() > 0.0f) ? 4 : 5;

    int cx = static_cast<int>(std::floor(offset.x() / voxel_size));
    int cy = static_cast<int>(std::floor(offset.y() / voxel_size));
    int cz = static_cast<int>(std::floor(offset.z() / voxel_size));

    // Clamp indices to your maximum allowed grid dimensions to prevent edge spillage
    cx = std::clamp(cx, 0, max_res_x - 1);
    cy = std::clamp(cy, 0, max_res_y - 1);
    cz = std::clamp(cz, 0, max_res_z - 1);

    return CellKey{cx, cy, cz, normal_id};
}

// all three needed?
Voxel &SpatialHashGrid::get_voxel(const point3 &p, const vec3 &n)
{
    CellKey key = get_cell_key(p, n);
    return grid[key];
}
const Voxel *SpatialHashGrid::get_voxel_const(const point3 &p, const vec3 &n) const
{
    CellKey key = get_cell_key(p, n);
    auto it = grid.find(key);
    if (it != grid.end())
    {
        return &(it->second);
    }
    return nullptr;
}
const Voxel *SpatialHashGrid::get_voxel_at_key(const CellKey &key) const
{
    auto it = grid.find(key);
    if (it != grid.end())
    {
        return &(it->second);
    }
    return nullptr;
}

void SpatialHashGrid::clear()
{
    grid.clear();
}

void SpatialHashGrid::clear_accumulators()
{
    for (auto &pair : grid)
    {
        Voxel &vox = pair.second;
        if (vox.use_quadtree)
        {
            // Reset raw energy in the quadtree while preserving split structure
            vox.dtree.clear_accumulators();
        }
        else
        {
            // Reset fixed spherical bins
            for (int t = 0; t < BIN_ROWS; t++)
            {
                for (int p = 0; p < BIN_COLOMNS; p++)
                {
                    vox.spherical_bins[t][p] = 0.0f;
                }
            }
        }
    }
}

uint64_t SpatialHashGrid::hash_position(const point3 &p, const vec3 &n, float size) const
{
    // 1. Spatial cell coordinates
    int64_t x = static_cast<int64_t>(std::floor(p.x() / size));
    int64_t y = static_cast<int64_t>(std::floor(p.y() / size));
    int64_t z = static_cast<int64_t>(std::floor(p.z() / size));

    // 2. Quantize surface normal into 6 primary direction IDs (0..5)
    int normal_id = 0;
    float ax = std::abs(n.x());
    float ay = std::abs(n.y());
    float az = std::abs(n.z());

    if (ax > ay && ax > az)
        normal_id = (n.x() > 0.0f) ? 0 : 1;
    else if (ay > az)
        normal_id = (n.y() > 0.0f) ? 2 : 3;
    else
        normal_id = (n.z() > 0.0f) ? 4 : 5;

    // 3. 64-bit prime spatial hashing constants
    const uint64_t p1 = 73856093ULL;
    const uint64_t p2 = 19349663ULL;
    const uint64_t p3 = 83492791ULL;
    const uint64_t p4 = 2654435761ULL; // Prime multiplier for normal_id

    return (static_cast<uint64_t>(x) * p1) ^
           (static_cast<uint64_t>(y) * p2) ^
           (static_cast<uint64_t>(z) * p3) ^
           (static_cast<uint64_t>(normal_id) * p4);
}

size_t SpatialHashGrid::get_size() const
{
    size_t sz = 0;
#pragma omp critical(spatial_hash_access)
    {
        sz = grid.size();
    }
    return sz;
}

void SpatialHashGrid::record_radiance(const point3 &hit_p, const vec3 &normal, const vec3 &dir, const color &radiance)
{
    vec3 outgoing_dir = unit_vector(dir);
    int theta_idx, phi_idx;
    ActiveMapping::dir_to_bin(outgoing_dir, theta_idx, phi_idx);

    // Standard ITU-R BT.709 perceived luminance conversion
    float luminance = 0.2126f * radiance[0] + 0.7152f * radiance[1] + 0.0722f * radiance[2];

    // Nudge spatial lookup off surface boundaries using the geometric normal
    // point3 query_p = hit_p + normal * 1e-3f;

#pragma omp critical(spatial_hash_access)
    {
        Voxel &vox = get_voxel(hit_p, normal); // Use query_p here!
        vox.spherical_bins[theta_idx][phi_idx] += luminance;
    }
}

void SpatialHashGrid::finalize_and_build_distributions() // for bins and quadtree TODO EMA
{
    constexpr float INV_FOUR_PI = 1.0f / (4.0f * M_PI);
    constexpr float epsilon_floor = 0.05f; // 5% uniform floor

    for (auto &pair : grid)
    {
        Voxel &vox = pair.second;

        // =========================================================
        // PATH A: QUAD TREE VOXEL
        // =========================================================
        if (vox.use_quadtree)
        {
            // Finalize, normalize, and build CDF for directional quadtree
            vox.dtree.build_distribution(epsilon_floor);
            // vox.dtree.clear_accumulators();
        }
        // =========================================================
        // PATH B: FIXED SPHERICAL BINS VOXEL
        // =========================================================
        else
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
                        float p_mass = vox.spherical_bins[t][p] / total_radiance;
                        float solid_angle = ActiveMapping::bin_solid_angle(t, p);
                        float uniform_mass = solid_angle * INV_FOUR_PI;

                        vox.guiding_pdf[t][p] = (1.0f - epsilon_floor) * p_mass + epsilon_floor * uniform_mass;
                        // vox.spherical_bins[t][p] = 0.0f; // Reset accumulator
                    }
                }
            }
            else
            {
                for (int t = 0; t < BIN_ROWS; t++)
                {
                    for (int p = 0; p < BIN_COLOMNS; p++)
                    {
                        float solid_angle = ActiveMapping::bin_solid_angle(t, p);
                        vox.guiding_pdf[t][p] = solid_angle * INV_FOUR_PI;
                        // vox.spherical_bins[t][p] = 0.0f;
                    }
                }
            }
        }
    }
    // clear_accumulators();
}

void SpatialHashGrid::finalize_and_build_distributions_EMA(int iteration_m)
{
    constexpr float INV_FOUR_PI = 1.0f / (4.0f * M_PI);
    constexpr float epsilon_floor = 0.05f; // 5% uniform floor

    // Select your learning rate strategy (pass-weighted sample mean or fixed decay)
    float alpha_m = 1.0f / static_cast<float>(std::max(1, iteration_m));
    // Alternatively for fixed decay: float alpha_m = 0.15f;

    for (auto &pair : grid)
    {
        Voxel &vox = pair.second;

        // =========================================================
        // PATH A: QUAD TREE VOXEL
        // =========================================================
        if (vox.use_quadtree)
        {
            // Note: Inside your dtree.build_distribution or prior to it,
            // you should apply the same EMA update:
            // persistent_weight = (1 - alpha_m) * old_weight + alpha_m * current_deposit;

            vox.dtree.build_distribution_EMA(epsilon_floor, alpha_m);
            vox.dtree.clear_accumulators(); // Reset current pass bin deposits
        }
        // =========================================================
        // PATH B: FIXED SPHERICAL BINS VOXEL
        // =========================================================
        else
        {
            float total_persistent_radiance = 0.0f;

            // Step 1: Apply EMA update to blend history with current pass deposit
            for (int t = 0; t < BIN_ROWS; t++)
            {
                for (int p = 0; p < BIN_COLOMNS; p++)
                {
                    float y_deposit = vox.spherical_bins[t][p]; // Current pass flux

                    // Blend history with new deposit
                    vox.persistent_bins[t][p] = (1.0f - alpha_m) * vox.persistent_bins[t][p] + alpha_m * y_deposit;

                    total_persistent_radiance += vox.persistent_bins[t][p];

                    // Clear current pass accumulator for the next iteration
                    vox.spherical_bins[t][p] = 0.0f;
                }
            }

            // Step 2: Build guiding PDF using the smoothed persistent history
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
                // Voxels with no historical data default to uniform distribution
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
}

float SpatialHashGrid::get_guided_pdf(const point3 &world_p, const vec3 &normal, const vec3 &dir) const
{
    float cos_dir = dot(normal, dir);
    if (cos_dir <= 0.0f) // Keep clean hemisphere clip
        return 0.0f;

    // Nudge consistently to avoid boundary coordinate jitter
    point3 query_p = world_p + normal * 1e-3f;
    const Voxel *vox = get_voxel_const(query_p, normal);

    // FIX 1: Match sample's fallback PDF exactly (Uniform solid-angle: 1 / (2 * pi))
    // Alternatively, if you want cosine-weighted, change sample's fallback to cos_dir / M_PI.
    if (!vox)
        return 1.0f / (2.0f * M_PI);

    int target_t = 0, target_p = 0;
    ActiveMapping::dir_to_bin(unit_vector(dir), target_t, target_p);

    float total_sum = 0.0f;
    for (int t = 0; t < BIN_ROWS; t++)
    {
        for (int p = 0; p < BIN_COLOMNS; p++)
        {
            vec3 bin_center = ActiveMapping::sample_bin(t, p, 0.5f, 0.5f);
            if (dot(normal, unit_vector(bin_center)) > -0.1f)
            {
                total_sum += vox->guiding_pdf[t][p];
            }
        }
    }

    if (total_sum <= 1e-7f)
        return 1.0f / (2.0f * M_PI); // Must match sample's fallback!

    float target_weight = vox->guiding_pdf[target_t][target_p];
    float discrete_prob = target_weight / total_sum;
    float solid_angle = ActiveMapping::bin_solid_angle(target_t, target_p);

    return (solid_angle > 1e-7f) ? (discrete_prob / solid_angle) : (1.0f / (2.0f * M_PI));
}

vec3 SpatialHashGrid::sample_guided_direction(const point3 &hit_p, const vec3 &normal, float &out_pdf_guiding) const
{
    // Nudge consistently
    point3 query_p = hit_p + normal * 1e-3f;
    const Voxel *vox = get_voxel_const(query_p, normal);

    if (!vox)
    {
        // Uniform hemisphere fallback
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

        out_pdf_guiding = 1.0f / (2.0f * M_PI);
        return unit_vector(x * u + y * v + z * w);
    }

    std::vector<float> local_pdf(BIN_ROWS * BIN_COLOMNS, 0.0f);
    float total_sum = 0.0f;

    for (int t = 0; t < BIN_ROWS; t++)
    {
        for (int p = 0; p < BIN_COLOMNS; p++)
        {
            vec3 dir = ActiveMapping::sample_bin(t, p, 0.5f, 0.5f);
            int idx = t * BIN_COLOMNS + p;

            if (dot(normal, unit_vector(dir)) > -0.1f)
            {
                local_pdf[idx] = vox->guiding_pdf[t][p];
            }
            else
            {
                local_pdf[idx] = 0.0f;
            }
            total_sum += local_pdf[idx];
        }
    }

    if (total_sum <= 1e-7f)
    {
        // Uniform hemisphere fallback matching get_guided_pdf
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

        out_pdf_guiding = 1.0f / (2.0f * M_PI);
        return unit_vector(x * u + y * v + z * w);
    }

    float r = rand_float() * total_sum;
    float running_sum = 0.0f;
    int sampled_t = 0, sampled_p = 0;
    bool found = false;

    for (int t = 0; t < BIN_ROWS && !found; t++)
    {
        for (int p = 0; p < BIN_COLOMNS; p++)
        {
            int idx = t * BIN_COLOMNS + p;
            running_sum += local_pdf[idx];
            if (r <= running_sum)
            {
                sampled_t = t;
                sampled_p = p;
                found = true;
                break;
            }
        }
    }

    float u_jitter = std::clamp(rand_float(), 0.01f, 0.99f);
    float v_jitter = std::clamp(rand_float(), 0.01f, 0.99f);
    vec3 wo = unit_vector(ActiveMapping::sample_bin(sampled_t, sampled_p, u_jitter, v_jitter));

    if (dot(normal, wo) <= 0.0f)
    {
        wo = unit_vector(ActiveMapping::sample_bin(sampled_t, sampled_p, 0.5f, 0.5f));
    }

    // FIX 2: Pass hit_p so get_guided_pdf applies the exact same query_p nudge!
    out_pdf_guiding = get_guided_pdf(hit_p, normal, wo);
    return wo;
}

// ============================================================================
// 4. REWRITTEN TRILINEAR INTERPOLATED LOOKUPS
// ============================================================================
/* float SpatialHashGrid::get_guided_pdf_interpolated(const point3 &world_p, const vec3 &normal, const vec3 &dir) const
{
    float cos_theta = dot(normal, dir);
    if (cos_theta <= 0.0f)
        return 0.0f;

    int target_t = 0, target_p = 0;
    ActiveMapping::dir_to_bin(dir, target_t, target_p);

    float solid_angle = ActiveMapping::bin_solid_angle(target_t, target_p);
    if (solid_angle <= 0.0f)
        return 0.0f;

    // Calculate normalized fractional coordinate within grid cell spaces
    point3 offset = world_p - origin;
    float tx = (offset.x() / voxel_size) - 0.5f;
    float ty = (offset.y() / voxel_size) - 0.5f;
    float tz = (offset.z() / voxel_size) - 0.5f;

    int ix0 = static_cast<int>(std::floor(tx));
    int iy0 = static_cast<int>(std::floor(ty));
    int iz0 = static_cast<int>(std::floor(tz));

    float fx = tx - std::floor(tx);
    float fy = ty - std::floor(ty);
    float fz = tz - std::floor(tz);

    float w[8] = {
        (1.0f - fx) * (1.0f - fy) * (1.0f - fz),
        fx * (1.0f - fy) * (1.0f - fz),
        (1.0f - fx) * fy * (1.0f - fz),
        fx * fy * (1.0f - fz),
        (1.0f - fx) * (1.0f - fy) * fz,
        fx * (1.0f - fy) * fz,
        (1.0f - fx) * fy * fz,
        fx * fy * fz};

    const CellKey corner_keys[8] = {
        {ix0, iy0, iz0}, {ix0 + 1, iy0, iz0}, {ix0, iy0 + 1, iz0}, {ix0 + 1, iy0 + 1, iz0}, {ix0, iy0, iz0 + 1}, {ix0 + 1, iy0, iz0 + 1}, {ix0, iy0 + 1, iz0 + 1}, {ix0 + 1, iy0 + 1, iz0 + 1}};

    float plane_epsilon = -0.5f * voxel_size;
    float accum_w = 0.0f;

    for (int k = 0; k < 8; ++k)
    {
        point3 corner_p = origin + point3(
                                       (corner_keys[k].x + 0.5f) * voxel_size,
                                       (corner_keys[k].y + 0.5f) * voxel_size,
                                       (corner_keys[k].z + 0.5f) * voxel_size);

        if (dot(corner_p - world_p, normal) < plane_epsilon)
            w[k] = 0.0f;

        accum_w += w[k];
    }

    if (accum_w > 0.0f)
    {
        for (int k = 0; k < 8; ++k)
            w[k] /= accum_w;
    }
    else
    {
        return get_guided_pdf(world_p, normal, dir); // Fallback to nearest neighbor
    }

    float interpolated_discrete_prob = 0.0f;
    for (int k = 0; k < 8; ++k)
    {
        if (w[k] <= 0.0f)
            continue;

        const Voxel *vox = get_voxel_at_key(corner_keys[k]);
        if (!vox)
        {
            interpolated_discrete_prob += w[k] * (cos_theta / M_PI) * solid_angle;
            continue;
        }

        float corner_hemi_sum = 0.0f;
        for (int t = 0; t < bin_rows; t++)
        {
            for (int p = 0; p < bin_colomns; p++)
            {
                vec3 bin_center = ActiveMapping::sample_bin(t, p, 0.5f, 0.5f);
                if (dot(normal, bin_center) > 0.0f)
                    corner_hemi_sum += vox->guiding_pdf[t][p];
            }
        }

        if (corner_hemi_sum > 0.0f)
            interpolated_discrete_prob += w[k] * (vox->guiding_pdf[target_t][target_p] / corner_hemi_sum);
        else
            interpolated_discrete_prob += w[k] * (cos_theta / M_PI) * solid_angle;
    }

    return interpolated_discrete_prob / solid_angle;
} */

// ============================================================================
// 5. REWRITTEN HEATMAP PPM DIAGNOSTIC EXPORT
// ============================================================================
/* void SpatialHashGrid::export_voxel_heatmap_ppm(const point3 &world_p, const std::string &filename)
{
    const Voxel *vox = get_voxel_const(world_p);

    int scale = 32;
    int img_w = bin_colomns * scale;
    int img_h = bin_rows * scale;

    std::ofstream out(filename);
    out << "P3\n"
        << img_w << " " << img_h << "\n255\n";

    if (!vox)
    {
        std::cout << "--> [Heatmap] Target " << world_p << " is UNVISITED. Writing default map." << std::endl;
        for (int i = 0; i < img_w * img_h; i++)
            out << "15 15 40\n";
        out.close();
        return;
    }

    float max_val = 0.0f;
    for (int t = 0; t < bin_rows; t++)
        for (int p = 0; p < bin_colomns; p++)
            max_val = std::max(max_val, vox->guiding_pdf[t][p]);

    if (max_val <= 1e-6f)
        max_val = 1.0f;

    for (int y = 0; y < img_h; y++)
    {
        int t = y / scale;
        for (int x = 0; x < img_w; x++)
        {
            int p = x / scale;
            float intensity = vox->guiding_pdf[t][p] / max_val;

            int r = static_cast<int>(std::clamp((intensity - 0.25f) * 4.0f, 0.0f, 1.0f) * 255);
            int g = static_cast<int>(std::clamp((0.5f - std::abs(intensity - 0.5f)) * 4.0f, 0.0f, 1.0f) * 255);
            int b = static_cast<int>(std::clamp((0.75f - intensity) * 4.0f, 0.0f, 1.0f) * 255);

            out << r << " " << g << " " << b << "\n";
        }
    }
    out.close();
} */

void SpatialHashGrid::record_radiance_QT(const point3 &hit_p, const vec3 &normal, const vec3 &dir, const color &rad)
{
    float lum = 0.2126f * rad[0] + 0.7152f * rad[1] + 0.0722f * rad[2];

#pragma omp critical(spatial_hash_access)
    {
        Voxel &vox = get_voxel(hit_p, normal);

        vox.dtree.record_radiance(dir, lum);
    }
}

vec3 SpatialHashGrid::sample_guided_direction_QT(const point3 &hit_p, const vec3 &normal, float &out_pdf) const
{
    const Voxel *vox = get_voxel_const(hit_p, normal);

    if (!vox || !vox->dtree.has_distribution)
    {
        // Cosine-weighted hemisphere sampling fallback
        float r1 = rand_float();
        float r2 = rand_float();

        float phi = 2.0f * M_PI * r1;
        float r = std::sqrt(r2);

        float x = r * std::cos(phi);
        float y = r * std::sin(phi);
        float z = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y));

        vec3 w = unit_vector(normal);
        vec3 a = (std::abs(w.x()) > 0.9f)
                     ? vec3(0, 1, 0)
                     : vec3(1, 0, 0);

        vec3 v = unit_vector(cross(w, a));
        vec3 u = cross(w, v);

        out_pdf = z / M_PI;

        return unit_vector(x * u + y * v + z * w);
    }

    // Sample from the full directional quadtree
    return vox->dtree.sample_direction(
        rand_float(), rand_float(), out_pdf);

    // Sample from the directional quadtree
    vec3 sampled_dir = vox->dtree.sample_direction(rand_float(), rand_float(), out_pdf);

    // If the sampled ray points into the geometric surface, fall back to hemisphere normal, TODO fix?
    /* if (dot(normal, sampled_dir) <= 0.0f)
    {
        sampled_dir = unit_vector(normal);
        out_pdf = get_guided_pdf_QT(hit_p, normal, sampled_dir);
    } */

    return sampled_dir;
}

float SpatialHashGrid::get_guided_pdf_QT(const point3 &hit_p, const vec3 &normal, const vec3 &dir) const
{
    // Rays pointing below the surface plane receive zero PDF
    /* float cos_dir = dot(normal, dir);
    if (cos_dir <= 0.0f)
    {
        return 0.0f;
    } */

    const Voxel *vox = get_voxel_const(hit_p, normal);

    if (!vox || !vox->dtree.has_distribution)
    {
        // Cosine-weighted hemisphere fallback
        float cos_dir = dot(normal, dir);

        if (cos_dir <= 0.0f)
            return 0.0f;

        return cos_dir / M_PI;
    }

    // Return the original quadtree PDF, including below-surface directions
    return vox->dtree.get_pdf(dir);
}

vec3 SpatialHashGrid::sample_guided_direction_QT_debug(const point3 &hit_p, const vec3 &normal, float &out_pdf) const
{
    const Voxel *vox = get_voxel_const(hit_p, normal);

    if (!vox)
    {
        // Cosine-weighted hemisphere sampling fallback for unvisited voxels
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

        out_pdf = 1.0f / (2.0f * M_PI);

        return unit_vector(x * u + y * v + z * w);
    }

    // Sample from the directional quadtree
    vec3 sampled_dir = vox->dtree.sample_direction_debug(rand_float(), rand_float(), out_pdf);

    // If the sampled ray points into the geometric surface, fall back to hemisphere normal
    if (dot(normal, sampled_dir) <= 0.0f)
    {
        sampled_dir = unit_vector(normal);
        out_pdf = get_guided_pdf_QT(hit_p, normal, sampled_dir);
    }

    return sampled_dir;
}

float SpatialHashGrid::get_guided_pdf_QT_debug(const point3 &hit_p, const vec3 &normal, const vec3 &dir) const
{
    // Rays pointing below the surface plane receive zero PDF
    if (dot(normal, dir) <= 0.0f)
    {
        return 0.0f;
    }

    const Voxel *vox = get_voxel_const(hit_p, normal);

    if (!vox)
    {
        return 1.0f / (2.0f * M_PI); // Uniform hemisphere fallback
    }

    return vox->dtree.get_pdf_debug(dir);
}

/* void SpatialHashGrid::test_first_quadtree_consistency() const
{
    const Voxel *sample_voxel = nullptr;

    // Find the first voxel that actually uses a quadtree
    for (const auto &pair : grid)
    {
        if (pair.second.use_quadtree)
        {
            sample_voxel = &pair.second;
            break;
        }
    }

    if (!sample_voxel)
    {
        std::cout << "[QT DIAGNOSTIC] No Quadtree voxels found in hash grid!" << std::endl;
        return;
    }

    const DirectionalQuadtree &dtree = sample_voxel->dtree;

    int num_samples = 50000;
    int mismatches = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < num_samples; ++i)
    {
        float sample_pdf = 0.0f;

        // 1. Generate sample direction using tree importance sampling
        vec3 dir = dtree.sample_direction(rand_float(), rand_float(), sample_pdf);

        // 2. Query PDF of that same direction independently
        float eval_pdf = dtree.get_pdf(dir);

        float diff = std::abs(sample_pdf - eval_pdf);

        // Use relative difference check to handle scale variations cleanly
        float rel_diff = diff / std::max(sample_pdf, 1e-5f);
        if (rel_diff > 1e-3f)
        {
            mismatches++;
            max_diff = std::max(max_diff, diff);
        }
    }

    // 3. SEPARATE UNIFORM MONTE CARLO INTEGRATION TEST (Integral should = 1.0)
    double pdf_integral_sum = 0.0;
    for (int i = 0; i < num_samples; ++i)
    {
        // Sample UNIFORMLY in UV space [0, 1)^2
        float u = rand_float();
        float v = rand_float();

        vec3 dir = dtree.uv_to_dir(u, v);
        pdf_integral_sum += static_cast<double>(dtree.get_pdf(dir));
    }

    // Equal-Area Cylindrical conversion factor (4 * PI)
    double estimated_integral = (4.0 * M_PI * pdf_integral_sum) / num_samples;

    std::cout << "\n=============================================" << std::endl;
    std::cout << "        QUADTREE CONSISTENCY DIAGNOSTIC       " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Tested Samples:         " << num_samples << std::endl;
    std::cout << "PDF Mismatch Count:     " << mismatches << " (" << (100.0f * mismatches / num_samples) << "%)" << std::endl;
    std::cout << "Max PDF Difference:     " << max_diff << std::endl;
    std::cout << "Integrated PDF Value:   " << estimated_integral << " (Expected: ~1.000)" << std::endl;
    std::cout << "=============================================\n"
              << std::endl;
} */

void SpatialHashGrid::inspect_quadtree(const DirectionalQuadtree &dtree)
{
    std::cout << "--- QUADTREE STRUCTURE ---" << std::endl;
    std::cout << "Root Radiance: " << dtree.get_root_radiance() << std::endl;
    std::cout << "Max Depth:     " << dtree.get_max_depth() << std::endl;
    std::cout << "Total Nodes:   " << dtree.get_node_count() << std::endl;
}

void SpatialHashGrid::run_mapping_diagnostic()
{
    std::cout << "\n=============================================\n";
    std::cout << "        EQUAL-AREA MAPPING DIAGNOSTIC        \n";
    std::cout << "=============================================\n";

    int test_samples = 10000;
    float max_reconstruction_error = 0.0f;

    for (int i = 0; i < test_samples; ++i)
    {
        // Generate uniform point on sphere
        float z = 1.0f - 2.0f * ((float)rand() / RAND_MAX);
        float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        float phi = 2.0f * M_PI * ((float)rand() / RAND_MAX);
        vec3 original_dir(r * std::cos(phi), r * std::sin(phi), z);

        // Map to UV
        float u, v;
        ActiveMapping::direction_to_uv(original_dir, u, v);

        // Map back to Dir
        vec3 reconstructed_dir = ActiveMapping::uv_to_direction(u, v);

        // Measure error
        float error = (original_dir - reconstructed_dir).length();
        max_reconstruction_error = std::max(max_reconstruction_error, error);
    }

    std::cout << "Max Inverse Mapping Error: " << max_reconstruction_error << " (Target: < 0.0001)" << std::endl;
    std::cout << "=============================================\n\n";
}

void SpatialHashGrid::diagnose_test_voxel(const point3 &test_p, const vec3 &test_normal, const std::string &heatmap_filename) const
{
    const Voxel *vox = get_voxel_const(test_p, test_normal);
    if (!vox)
    {
        std::cout << "[DIAGNOSTIC WARN] No voxel found at test_point ("
                  << test_p.x() << ", " << test_p.y() << ", " << test_p.z() << ")!" << std::endl;
        return;
    }

    std::cout << "\n=================== QUADTREE DIAGNOSTICS ===================" << std::endl;
    std::cout << "Test Voxel Position: " << test_p << std::endl;
    // FIX 1: Use public getter instead of accessing private root member directly
    std::cout << "Root sum_radiance: " << vox->dtree.get_root_radiance() << std::endl;

    // print node count
    std::cout << "Total QuadTree Nodes: " << vox->dtree.get_node_count() << std::endl;
    std::cout << "QuadTree Leafs: " << vox->dtree.get_leaf_count() << std::endl;

    // FIX 2: Can now be called with 0 arguments due to default parameters
    // vox->dtree.print_tree_structure();

    // Generate PPM image of the UV tree PDF density
    vox->dtree.export_heatmap_ppm(heatmap_filename);
    std::cout << "============================================================\n"
              << std::endl;
}