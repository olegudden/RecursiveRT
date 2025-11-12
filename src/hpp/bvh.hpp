#ifndef BVH_HPP
#define BVH_HPP

#include "aabb.hpp"
#include "triangle.hpp"
#include "hittable.hpp"

#include <vector>
#include <memory>
#include <limits>
#include <iostream>

struct bvh_node
{
    aabb box;
    std::unique_ptr<bvh_node> left = nullptr;
    std::unique_ptr<bvh_node> right = nullptr;
    std::vector<triangle> triangles; // Only filled in leaves

    bvh_node() = default;
};

struct bvh_stats
{
    int total_nodes = 0;
    int leaf_nodes = 0;

    int min_tris_per_leaf = std::numeric_limits<int>::max();
    int max_tris_per_leaf = 0;
    double sum_tris_per_leaf = 0;
    int leaf_count = 0;

    int min_depth = std::numeric_limits<int>::max();
    int max_depth = 0;

    mutable int total_aabb_tests = 0;
    mutable int64_t total_triangle_tests = 0;

    double build_time_ms = 0.0;
    double traverse_time_ms = 0.0;

    int64_t total_surface_area = 0;
    double leafs_surface_area = 0.0;
    float overlap_volume = 0.0f;

    void printStats() const;
    void collect_bvh_stats(const bvh_node *node, int depth, bvh_stats &stats);
};

class bvh : public hittable
{
public:
    std::unique_ptr<bvh_node> root;
    bvh_stats stats;

    bvh() = default;

    std::unique_ptr<bvh_node> build_bvh_binned_sah(std::vector<triangle> &tris, int depth = 0);
    std::unique_ptr<bvh_node> build_bvh_cycle_axis_equal(std::vector<triangle> &tris, int depth = 0);
    std::unique_ptr<bvh_node> build_bvh_longest_axis_midpoint(std::vector<triangle> &tris, int depth = 0);
    std::unique_ptr<bvh_node> build_bvh_exact_sah(std::vector<triangle> &tris, int depth = 0);

    bool hit(const ray &r, float t_min, float t_max, hit_info &hi) const override;

private:
    bool hit_node(bvh_node *node, const ray &r, float t_min, float t_max, hit_info &final_hi) const;
};

#endif // BVH_H
