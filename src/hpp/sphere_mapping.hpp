#ifndef SPHERE_MAPPING_HPP
#define SPHERE_MAPPING_HPP
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// set spherical bin dimensions
static const int BIN_ROWS = 8;
static const int BIN_COLOMNS = 16;

#include "vec3.hpp"
#include <algorithm>

namespace SphereMapping
{
    // 1. LATITUDE-LONGITUDE
    struct LatLong
    {
        static inline void dir_to_bin(const vec3 &dir, int &out_t, int &out_p)
        {
            // BUG FIX 1: Keep positive vector direction consistently
            vec3 v = unit_vector(dir);

            // 1. Theta (Elevation along Y axis)
            float theta = std::acos(std::clamp(v.y(), -1.0f, 1.0f));
            out_t = std::clamp(static_cast<int>((theta / M_PI) * 8.0f), 0, 7);

            // Handle pole singularity
            if (std::abs(v.x()) < 1e-5f && std::abs(v.z()) < 1e-5f)
            {
                out_p = 0;
                return;
            }

            // 2. Phi (Azimuth in X-Z plane)
            float phi = std::atan2(v.z(), v.x()); // Range [-PI, PI]
            phi += 1.5f * M_PI;                   // Apply forward shift

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
            // BUG FIX 3: Epsilon clamp keeps sample strictly inside bin bounds [0, 1)
            u1 = std::clamp(u1, 1e-5f, 1.0f - 1e-5f);
            u2 = std::clamp(u2, 1e-5f, 1.0f - 1e-5f);

            float delta_theta = M_PI / 8.0f;
            float delta_phi = (2.0f * M_PI) / 16.0f;

            // Theta jittering (Cosine space for solid-angle uniformity)
            float theta_min = t * delta_theta;
            float theta_max = (t + 1) * delta_theta;
            float cos_theta_min = std::cos(theta_min);
            float cos_theta_max = std::cos(theta_max);

            float jittered_cos_theta = cos_theta_min + u2 * (cos_theta_max - cos_theta_min);
            float jittered_theta = std::acos(std::clamp(jittered_cos_theta, -1.0f, 1.0f));

            // Phi jittering
            float phi_min = p * delta_phi;
            float phi_max = (p + 1) * delta_phi;
            float jittered_phi = phi_min + u1 * (phi_max - phi_min);

            // BUG FIX 2: Undo phase shift and wrap back into standard atan2 range [-PI, PI]
            float raw_phi = jittered_phi - 1.5f * M_PI;
            while (raw_phi < -M_PI)
                raw_phi += 2.0f * M_PI;
            while (raw_phi >= M_PI)
                raw_phi -= 2.0f * M_PI;

            // Cartesian reconstruction
            float sin_theta = std::sin(jittered_theta);
            float dx = sin_theta * std::cos(raw_phi);
            float dy = std::cos(jittered_theta);
            float dz = sin_theta * std::sin(raw_phi);

            return unit_vector(vec3(dx, dy, dz));
        }

        static inline float bin_solid_angle(int t, int p)
        {
            float delta_theta = M_PI / 8.0f;
            float delta_phi = (2.0f * M_PI) / 16.0f;
            return delta_phi * (std::cos(t * delta_theta) - std::cos((t + 1) * delta_theta));
        }
    };

    // 2. EQUAL-AREA CYLINDRICAL
    struct EqualArea
    {
        static constexpr float INV_FOUR_PI = 1.0f / (4.0f * M_PI);

        static inline float jacobian(const vec3 &dir)
        {
            // For Lambert equal-area cylindrical projection,
            // the mapping Jacobian relative to solid angle is strictly constant (1 / 4pi).
            return INV_FOUR_PI;
        }

        static inline void dir_to_bin(const vec3 &dir, int &out_t, int &out_p)
        {
            // 1. Elevation: dir.y() corresponds to cos_theta in [-1, 1]
            float cos_theta = std::clamp(dir.y(), -1.0f, 1.0f);

            // 2. Azimuth: Standard spherical azimuth mapping in [0, 2pi]
            float phi = std::atan2(dir.z(), dir.x());
            if (phi < 0.0f)
                phi += 2.0f * M_PI;

            // 3. Map linearly across grid dimensions
            float norm_cos = (cos_theta + 1.0f) * 0.5f; // [-1, 1] -> [0, 1]

            out_t = std::clamp(static_cast<int>(norm_cos * BIN_ROWS), 0, BIN_ROWS - 1);
            out_p = std::clamp(static_cast<int>(phi / ((2.0f * M_PI) / BIN_COLOMNS)), 0, BIN_COLOMNS - 1);
        }

        static inline vec3 sample_bin(int t, int p, float u1, float u2)
        {
            float delta_phi = (2.0f * M_PI) / static_cast<float>(BIN_COLOMNS);
            float phi_min = p * delta_phi;
            float jittered_phi = phi_min + u1 * delta_phi;

            // Uniformly slice in cosine space [-1, 1]
            float delta_cos = 2.0f / static_cast<float>(BIN_ROWS);
            float cos_min = -1.0f + t * delta_cos;
            float jittered_cos = cos_min + u2 * delta_cos;

            float sin_theta = std::sqrt(std::max(0.0f, 1.0f - jittered_cos * jittered_cos));

            // Matching standard spherical coordinate reconstruction
            return unit_vector(vec3(
                sin_theta * std::cos(jittered_phi),
                jittered_cos,
                sin_theta * std::sin(jittered_phi)));
        }

        static inline float bin_solid_angle(int t, int p)
        {
            // Equal-area guarantee: Every bin has the exact same solid angle!
            return (4.0f * M_PI) / static_cast<float>(BIN_ROWS * BIN_COLOMNS);
        }

        static inline vec3 uv_to_direction(float u, float v)
        {
            float phi = u * 2.0f * M_PI;
            float cos_theta = v * 2.0f - 1.0f;
            float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));

            return unit_vector(vec3(
                sin_theta * std::cos(phi),
                cos_theta,
                sin_theta * std::sin(phi)));
        }

        static inline void direction_to_uv(const vec3 &dir, float &u, float &v)
        {
            vec3 d = unit_vector(dir);
            float cos_theta = std::clamp(d.y(), -1.0f, 1.0f);
            float phi = std::atan2(d.z(), d.x());
            if (phi < 0.0f)
                phi += 2.0f * M_PI;

            u = phi / (2.0f * M_PI);
            v = (cos_theta + 1.0f) * 0.5f;
        }
    };

    // 3. OCTAHEDRAL
    struct Octahedral
    {
        // Helper to handle sign symmetry correctly
        static inline float oct_sign(float x)
        {
            return (x >= 0.0f) ? 1.0f : -1.0f;
        }

        // 1. Direction -> UV
        static inline void direction_to_uv(const vec3 &dir, float &u, float &v)
        {
            vec3 d = unit_vector(dir);

            // Project onto L1-norm octahedron
            float l1_norm = std::abs(d.x()) + std::abs(d.y()) + std::abs(d.z());
            float x = d.x() / l1_norm;
            float z = d.z() / l1_norm;

            // Fold outer corners if lower hemisphere
            if (d.y() < 0.0f)
            {
                float temp_x = x;
                x = (1.0f - std::abs(z)) * oct_sign(temp_x);
                z = (1.0f - std::abs(temp_x)) * oct_sign(z);
            }

            u = x * 0.5f + 0.5f;
            v = z * 0.5f + 0.5f;
        }

        // 2. UV -> Direction
        static inline vec3 uv_to_direction(float u, float v)
        {
            float x = u * 2.0f - 1.0f;
            float z = v * 2.0f - 1.0f;
            float y = 1.0f - std::abs(x) - std::abs(z);

            vec3 dir(x, y, z);

            if (y < 0.0f)
            {
                float temp_x = dir.x();
                dir[0] = (1.0f - std::abs(dir.z())) * oct_sign(temp_x);
                dir[2] = (1.0f - std::abs(temp_x)) * oct_sign(dir.z());
            }

            return unit_vector(dir);
        }

        // 3. Direction -> Bin (Matching VoxelGrid API)
        static inline void dir_to_bin(const vec3 &dir, int &out_t, int &out_p)
        {
            float u, v;
            direction_to_uv(dir, u, v);

            out_t = std::clamp(static_cast<int>(u * BIN_ROWS), 0, BIN_ROWS - 1);
            out_p = std::clamp(static_cast<int>(v * BIN_COLOMNS), 0, BIN_COLOMNS - 1);
        }

        // 4. Sample Bin (Matching VoxelGrid API)
        static inline vec3 sample_bin(int t, int p, float u1, float u2)
        {
            float delta_u = 1.0f / static_cast<float>(BIN_ROWS);
            float delta_v = 1.0f / static_cast<float>(BIN_COLOMNS);

            float jittered_u = (t + u1) * delta_u;
            float jittered_v = (p + u2) * delta_v;

            return uv_to_direction(jittered_u, jittered_v);
        }

        // 5. Bin Solid Angle (Analytical integration of 2D UV quad on sphere)
        static inline float bin_solid_angle(int t, int p)
        {
            // Sample corners of the UV quad
            float delta_u = 1.0f / static_cast<float>(BIN_ROWS);
            float delta_v = 1.0f / static_cast<float>(BIN_COLOMNS);

            float u0 = t * delta_u;
            float u1 = (t + 1) * delta_u;
            float v0 = p * delta_v;
            float v1 = (p + 1) * delta_v;

            vec3 p00 = uv_to_direction(u0, v0);
            vec3 p10 = uv_to_direction(u1, v0);
            vec3 p01 = uv_to_direction(u0, v1);
            vec3 p11 = uv_to_direction(u1, v1);

            // Approximate solid angle using two spherical triangles
            auto spherical_triangle_area = [](const vec3 &a, const vec3 &b, const vec3 &c)
            {
                float num = std::abs(dot(a, cross(b, c)));
                float den = 1.0f + dot(a, b) + dot(b, c) + dot(c, a);
                return 2.0f * std::atan2(num, den);
            };

            return spherical_triangle_area(p00, p10, p11) + spherical_triangle_area(p00, p11, p01);
        }
    };
}

#endif