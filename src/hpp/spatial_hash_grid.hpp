#define _USE_MATH_DEFINES

#pragma once

#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "vec3.hpp"
#include "path_guiding.hpp"
#include "spatial_datastructure.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

using ActiveMapping = SphereMapping::EqualArea;

struct CellKey
{
    int x, y, z;
    int normal_id; // 0..5 für die 6 Hauptrichtungen

    bool operator==(const CellKey &other) const
    {
        return x == other.x && y == other.y && z == other.z && normal_id == other.normal_id;
    }
};

struct CellKeyHash
{
    std::size_t operator()(const CellKey &key) const
    {
        const std::size_t p1 = 73856093;
        const std::size_t p2 = 19349663;
        const std::size_t p3 = 83492791;
        const std::size_t p4 = 2654435761;

        return (static_cast<std::size_t>(key.x) * p1) ^
               (static_cast<std::size_t>(key.y) * p2) ^
               (static_cast<std::size_t>(key.z) * p3) ^
               (static_cast<std::size_t>(key.normal_id) * p4);
    }
};

class SpatialHashGrid : public SpatialDatastructure
{
public:
    bool is_training = true;
    float voxel_size; // = 0.5f;

    point3 origin, end;
    std::unordered_map<CellKey, Voxel, CellKeyHash> grid;

    // Precomputed maximum integer indices for fast clamping
    int max_res_x = 0;
    int max_res_y = 0;
    int max_res_z = 0;

    void initialize(float cell_size, point3 scene_min, point3 scene_max);
    CellKey get_cell_key(const point3 &p, const vec3 &n) const;

    // Key Comparison helper replacement for legacy get_voxel_index()
    /* bool is_same_cell(const point3 &p1, const point3 &p2) const
    {
        return get_cell_key(p1) == get_cell_key(p2);
    } */

    Voxel &get_voxel(const point3 &p, const vec3 &n);
    const Voxel *get_voxel_const(const point3 &p, const vec3 &n) const;
    const Voxel *get_voxel_at_key(const CellKey &key) const;

    void record_radiance(const point3 &hit_p, const vec3 &normal, const vec3 &dir, const color &radiance);
    void finalize_and_build_distributions();
    void finalize_and_build_distributions_EMA(int iteration_m);

    float get_guided_pdf(const point3 &world_p, const vec3 &normal, const vec3 &dir) const;
    // float get_guided_pdf_interpolated(const point3 &world_p, const vec3 &normal, const vec3 &dir) const;

    vec3 sample_guided_direction(const point3 &hit_p, const vec3 &normal, float &out_pdf_guiding) const;
    // vec3 sample_guided_direction_interpolated(const point3 &hit_p, const vec3 &normal, float &out_pdf_guiding);

    // void export_voxel_heatmap_ppm(const point3 &world_p, const std::string &filename);

    void record_radiance_QT(const point3 &hit_p, const vec3 &normal, const vec3 &dir, const color &rad);
    vec3 sample_guided_direction_QT(const point3 &hit_p, const vec3 &normal, float &out_pdf) const;
    vec3 sample_guided_direction_QT_debug(const point3 &hit_p, const vec3 &normal, float &out_pdf) const;
    float get_guided_pdf_QT(const point3 &hit_p, const vec3 &normal, const vec3 &dir) const;
    float get_guided_pdf_QT_debug(const point3 &hit_p, const vec3 &normal, const vec3 &dir) const;

    // void test_first_quadtree_consistency() const;
    void inspect_quadtree(const DirectionalQuadtree &dtree);

    void clear();
    void clear_accumulators();

    // Public diagnostic & hashing helpers
    uint64_t hash_position(const point3 &p, const vec3 &n, float size) const;
    size_t get_size() const;

    void run_mapping_diagnostic();
    void diagnose_test_voxel(const point3 &test_p, const vec3 &test_normal, const std::string &heatmap_filename) const;

private:
};