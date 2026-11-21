#pragma once

#include <memory>
#include <array>
#include "vec3.hpp"
#include "sphere_mapping.hpp"

struct DTreeNode
{
    float sum_radiance = 0.0f;
    float pdf_weight = 0.0f;
    float current_radiance = 0.0f;

    std::array<std::unique_ptr<DTreeNode>, 4> children = {nullptr, nullptr, nullptr, nullptr};

    bool is_leaf() const { return children[0] == nullptr; }

    void reset()
    {
        sum_radiance = 0.0f;
        for (auto &child : children)
        {
            if (child)
                child->reset();
            child.reset();
        }
    }

    void reset_accumulators()
    {
        current_radiance = 0.0f;
        // sum_radiance = 0.0f;
        // pdf_weight = 0.0f;

        for (auto &child : children)
        {
            if (child)
            {
                child->reset_accumulators(); // Recursively clear children without deleting them
            }
        }
    }
};

class DirectionalQuadtree
{
public:
    bool has_distribution = false;
    DirectionalQuadtree() = default;

    void record_radiance(const vec3 &dir, float radiance);
    vec3 sample_direction(float u1, float u2, float &out_pdf) const;
    vec3 sample_direction_debug(float u1, float u2, float &out_pdf) const;
    float get_pdf(const vec3 &dir) const;
    float get_pdf_debug(const vec3 &dir) const;
    void reset();
    void clear_accumulators();
    void build_distribution(float epsilon_floor = 0.05f);
    void build_distribution_EMA(float epsilon_floor, float alpha_m);

    const DTreeNode *get_root() const { return &root; }

    float get_root_radiance() const;
    int get_max_depth() const;
    int get_node_count() const;
    int get_leaf_count() const;

    void export_heatmap_ppm(const std::string &filename) const;
    void print_tree_structure(const DTreeNode *node = nullptr, int depth = 0) const;

    void dir_to_uv(const vec3 &dir, float &u, float &v) const;
    vec3 uv_to_dir(float u, float v) const;

private:
    DTreeNode root;
    int sample_count = 0;

    void record_recursive(DTreeNode &node, float u, float v, float radiance, int depth);
    float get_pdf_recursive(const DTreeNode &node, uint32_t ix, uint32_t iy, int depth) const;
    float get_pdf_recursive_uv(const DTreeNode &node, float u, float v, float u_min, float u_max, float v_min, float v_max, int depth) const;
    float get_pdf_recursive_debug(const DTreeNode &node, float u, float v, int depth) const;
    void build_distribution_recursive(DTreeNode &node, float epsilon_floor);
    void apply_ema_and_update_sums(DTreeNode &node, float alpha_m);
    void build_distribution_recursive_EMA(DTreeNode &node, float epsilon_floor);

    int get_max_depth_recursive(const DTreeNode &node, int current_depth) const;
    int get_node_count_recursive(const DTreeNode &node) const;
    int get_leaf_count_recursive(const DTreeNode &node) const;
};