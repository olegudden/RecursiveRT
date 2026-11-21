#include "../hpp/directional_quadtree.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

using ActiveMapping = SphereMapping::EqualArea;

void DirectionalQuadtree::dir_to_uv(const vec3 &dir, float &u, float &v) const
{

    ActiveMapping::direction_to_uv(dir, u, v);
}

vec3 DirectionalQuadtree::uv_to_dir(float u, float v) const

{

    return ActiveMapping::uv_to_direction(u, v);
}

void DirectionalQuadtree::reset()
{

    root.reset();
}

void DirectionalQuadtree::clear_accumulators()
{
    root.reset_accumulators();
    // has_distribution = false;
}

void DirectionalQuadtree::record_radiance(const vec3 &dir, float radiance)

{
    sample_count++; // Track how many total samples/rays hit this voxel
    if (radiance <= 0.0f)

        return;

    float u, v;

    dir_to_uv(unit_vector(dir), u, v);

    record_recursive(root, u, v, radiance, 0);
}

void DirectionalQuadtree::record_recursive(DTreeNode &node, float u, float v, float radiance, int depth)
{
    node.sum_radiance += radiance;

    // Max depth cap (depth 4 = 16x16 resolution, depth 5 = 32x32)
    /* if (depth >= 5) // CHANGE DEPTH (1)
        return;

    // 1. If it's a leaf, check if we should subdivide it
    if (node.is_leaf())
    {
        // Require a safe minimum energy accumulation before allowing a split
        // (Prevents hyper-splitting on the first few rays of a new iteration)
        const float MIN_ENERGY_FOR_SPLIT = 1e-3f;

        // Use a safe floor for root sum radiance so it never evaluates to 0.0f on early rays
        float safe_root_radiance = std::max(root.sum_radiance, 1.0f);
        float relative_split_threshold = safe_root_radiance * 0.05f; // CHANGE THRESHOLD

        // Only split if we cross a meaningful energy threshold
        if (node.sum_radiance > std::max(MIN_ENERGY_FOR_SPLIT, relative_split_threshold))
        {
            // Subdivide leaf into 4 children with ZERO initial radiance for the new batch
            for (auto &child : node.children)
            {
                child = std::make_unique<DTreeNode>();
                child->sum_radiance = 0.0f;
            }

            // Distribute current accumulated energy to children evenly so tree remains balanced
            float distributed_radiance = node.sum_radiance / 4.0f;
            for (auto &child : node.children)
            {
                child->sum_radiance = distributed_radiance;
            }
        }
        else
        {
            // Stop here: stays a leaf, radiance is recorded in node.sum_radiance
            return;
        }
    }

    // 2. If it was already an internal node OR was just split above, recurse into child
    int child_idx = (u >= 0.5f ? 1 : 0) + (v >= 0.5f ? 2 : 0);

    float next_u = (u >= 0.5f) ? (u - 0.5f) * 2.0f : u * 2.0f;
    float next_v = (v >= 0.5f) ? (v - 0.5f) * 2.0f : v * 2.0f;

    record_recursive(*node.children[child_idx], next_u, next_v, radiance, depth + 1); */

    // Record the observation in the current-pass accumulator.
    node.current_radiance += radiance;

    // Maximum tree depth reached.
    if (depth >= 5)
        return;

    // Determine the child corresponding to this direction.
    int child_idx =
        (u >= 0.5f ? 1 : 0) +
        (v >= 0.5f ? 2 : 0);

    float next_u = (u >= 0.5f)
                       ? (u - 0.5f) * 2.0f
                       : u * 2.0f;

    float next_v = (v >= 0.5f)
                       ? (v - 0.5f) * 2.0f
                       : v * 2.0f;

    // Case 1: Current node is a leaf.
    if (node.is_leaf())
    {
        float updated_radiance =
            node.sum_radiance + node.current_radiance;

        const float MIN_ENERGY_FOR_SPLIT = 1e-3f;

        float relative_split_threshold = root.sum_radiance * 0.05f;

        float split_threshold = std::max(MIN_ENERGY_FOR_SPLIT, relative_split_threshold);

        if (updated_radiance > split_threshold)
        {
            // Distribute historical persistent energy.
            float historical_share =
                node.sum_radiance / 4.0f;

            // Distribute current-pass observations as well.
            float current_share =
                node.current_radiance / 4.0f;

            for (auto &child : node.children)
            {
                child = std::make_unique<DTreeNode>();

                child->sum_radiance = historical_share;
                child->current_radiance = current_share;
            }

            // The parent retains its aggregate values.
            // The current observation has already been recorded.
            // Do not add it again during recursion.
            return;
        }

        return;
    }

    // Case 2: Node is already internal.
    record_recursive(
        *node.children[child_idx],
        next_u,
        next_v,
        radiance,
        depth + 1);

    // Record current-pass observation.
    /* node.current_radiance += radiance;

    // Maximum tree depth reached.
    if (depth >= 5)
        return;

    // Determine the corresponding child.
    int child_idx =
        (u >= 0.5f ? 1 : 0) +
        (v >= 0.5f ? 2 : 0);

    float next_u = (u >= 0.5f)
                       ? (u - 0.5f) * 2.0f
                       : u * 2.0f;

    float next_v = (v >= 0.5f)
                       ? (v - 0.5f) * 2.0f
                       : v * 2.0f;

    if (node.is_leaf())
    {
        // Persistent + current-pass accumulated energy.
        float node_energy =
            node.sum_radiance + node.current_radiance;

        // Estimate average energy at the current depth.
        float expected_energy =
            root.sum_radiance /
            std::pow(4.0f, depth);

        const float MIN_ENERGY_FOR_SPLIT = 1e-3f;
        const float SPLIT_FACTOR = 1.99f;

        float split_threshold =
            std::max(
                MIN_ENERGY_FOR_SPLIT,
                SPLIT_FACTOR * expected_energy);

        if (node_energy > split_threshold)
        {
            std::cout
                << "[QT SPLIT] depth=" << depth
                << " node_energy=" << node_energy
                << " threshold=" << split_threshold
                << " root_energy=" << root.sum_radiance
                << std::endl;

            // Preserve existing accumulated information.
            float historical_share =
                node.sum_radiance / 4.0f;

            float current_share =
                node.current_radiance / 4.0f;

            for (auto &child : node.children)
            {
                child = std::make_unique<DTreeNode>();

                child->sum_radiance = historical_share;
                child->current_radiance = current_share;
            }
        }

        return;
    }

    // Internal node: continue into the appropriate child.
    record_recursive(
        *node.children[child_idx],
        next_u,
        next_v,
        radiance,
        depth + 1); */
}

float DirectionalQuadtree::get_pdf(const vec3 &dir) const
{
    if (!has_distribution)
        return 1.0f / (4.0f * M_PI);
    float u, v;
    dir_to_uv(dir, u, v);

    u = std::clamp(u, 0.0f, 0.999999f);
    v = std::clamp(v, 0.0f, 0.999999f);

    float pdf_uv = get_pdf_recursive_uv(root, u, v, 0.0f, 1.0f, 0.0f, 1.0f, 0);
    return pdf_uv * ActiveMapping::jacobian(dir);
}

float DirectionalQuadtree::get_pdf_recursive_uv(const DTreeNode &node, float u, float v, float u_min, float u_max, float v_min, float v_max, int depth) const
{
    if (node.is_leaf() || depth >= 5) // CHANGE MAX_DEPTH (2) HERE
        return 1.0f;

    float total_weight = 0.0f;
    std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 4; ++i)
    {
        if (node.children[i])
        {
            weights[i] = node.children[i]->pdf_weight;
            total_weight += weights[i];
        }
    }

    if (total_weight <= 0.0f)
        return 1.0f;

    // Find which quadrant (child) contains (u, v) based on the current bounding box midpoints
    float u_mid = 0.5f * (u_min + u_max);
    float v_mid = 0.5f * (v_min + v_max);

    int child_idx = (u >= u_mid ? 1 : 0) + (v >= v_mid ? 2 : 0);
    const auto &child = node.children[child_idx];

    if (!child || weights[child_idx] <= 0.0f)
    {
        std::cout << "PDF traversal encountered invalid child!"
                  << " depth: " << depth
                  << " child_idx: " << child_idx
                  << " total_weight: " << total_weight
                  << " weight: " << weights[child_idx]
                  << std::endl;

        return 0.0f;
    }

    float p_child = weights[child_idx] / total_weight;

    // Update bounds for the chosen child (matching the sampler's bounding box logic)
    float next_u_min = (child_idx & 1) ? u_mid : u_min;
    float next_u_max = (child_idx & 1) ? u_max : u_mid;
    float next_v_min = (child_idx & 2) ? v_mid : v_min;
    float next_v_max = (child_idx & 2) ? v_max : v_mid;

    return (p_child * 4.0f) * get_pdf_recursive_uv(*child, u, v, next_u_min, next_u_max, next_v_min, next_v_max, depth + 1);
}

float DirectionalQuadtree::get_pdf_recursive(const DTreeNode &node, uint32_t ix, uint32_t iy, int depth) const
{
    if (node.is_leaf() || node.sum_radiance <= 0.0f || depth >= 16)
        return 1.0f;

    float total_weight = 0.0f;
    std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 4; ++i)
    {
        if (node.children[i])
        {
            weights[i] = node.children[i]->pdf_weight;
            total_weight += weights[i];
        }
    }

    if (total_weight <= 0.0f)
        return 1.0f;

    // Extract exact child branch bit for this depth
    int bit = 15 - depth;
    int child_idx = ((ix >> bit) & 1) | (((iy >> bit) & 1) << 1);

    const auto &child = node.children[child_idx];

    if (!child || weights[child_idx] <= 0.0f)
        return 0.0f; // Return 0.0f for unlit regions to prevent non-zero PDF on 0 radiance

    float p_child = weights[child_idx] / total_weight;

    return (p_child * 4.0f) * get_pdf_recursive(*child, ix, iy, depth + 1);
}

float DirectionalQuadtree::get_pdf_debug(const vec3 &dir) const
{
    std::cout << "\n--- [DEBUG EVALUATION TRACE] ---\n";
    if (root.sum_radiance <= 0.0f)
    {
        std::cout << "Root sum_radiance <= 0. Root PDF = 1 / 4pi\n";
        return 1.0f / (4.0f * M_PI);
    }

    float u, v;
    dir_to_uv(dir, u, v);

    std::cout << "Input Vector Direction : (" << dir.x() << ", " << dir.y() << ", " << dir.z() << ")\n";
    std::cout << "Reprojected UV         : (" << u << ", " << v << ")\n";

    float clamped_u = std::clamp(u, 0.0f, 0.999999f);
    float clamped_v = std::clamp(v, 0.0f, 0.999999f);

    if (clamped_u != u || clamped_v != v)
    {
        std::cout << "WARNING: UV clamped from (" << u << ", " << v << ") to ("
                  << clamped_u << ", " << clamped_v << ")\n";
    }

    float pdf_uv = get_pdf_recursive_debug(root, clamped_u, clamped_v, 0);
    float jacobian = ActiveMapping::jacobian(dir);
    float final_pdf = pdf_uv * jacobian;

    std::cout << "Final PDF UV  : " << pdf_uv << "\n";
    std::cout << "Jacobian      : " << jacobian << "\n";
    std::cout << "Final PDF     : " << final_pdf << "\n";
    std::cout << "--------------------------------\n";

    return final_pdf;
}

float DirectionalQuadtree::get_pdf_recursive_debug(const DTreeNode &node, float u, float v, int depth) const
{
    if (node.is_leaf())
    {
        std::cout << "  [Depth " << depth << "] Leaf node reached at " << &node << ". Returning 1.0\n";
        return 1.0f;
    }

    float total_weight = 0.0f;
    std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 4; ++i)
    {
        if (node.children[i])
        {
            weights[i] = node.children[i]->pdf_weight;
            total_weight += weights[i];
        }
    }

    if (total_weight <= 0.0f)
    {
        std::cout << "  [Depth " << depth << "] Node " << &node << " has 0 total weight. Returning 1.0\n";
        return 1.0f;
    }

    int child_idx = (u >= 0.5f ? 1 : 0) + (v >= 0.5f ? 2 : 0);
    const auto &child = node.children[child_idx];

    std::cout << "  [Depth " << depth << "] Node " << &node
              << " | UV: (" << u << ", " << v << ")"
              << " | Target Child Index: " << child_idx;

    if (!child)
    {
        std::cout << " | Child is NULL! Returning 1e-6\n";
        return 1e-6f;
    }

    float p_child = weights[child_idx] / total_weight;

    float next_u = (u >= 0.5f) ? (u - 0.5f) * 2.0f : u * 2.0f;
    float next_v = (v >= 0.5f) ? (v - 0.5f) * 2.0f : v * 2.0f;

    next_u = std::clamp(next_u, 0.0f, 0.999999f);
    next_v = std::clamp(next_v, 0.0f, 0.999999f);

    std::cout
        << " | Child Radiance Ratio: " << p_child
        << " | Branch Factor (*4): " << (p_child * 4.0f)
        << " | Next UV: ("
        << next_u << ", "
        << next_v << ")\n";

    return (p_child * 4.0f) * get_pdf_recursive_debug(*child, next_u, next_v, depth + 1);
}

vec3 DirectionalQuadtree::sample_direction_debug(float u1, float u2, float &out_pdf) const
{
    if (root.sum_radiance <= 0.0f)
    {
        out_pdf = 1.0f / (4.0f * M_PI);
        return uv_to_dir(u1, u2);
    }

    float u_min = 0.0f, u_max = 1.0f;
    float v_min = 0.0f, v_max = 1.0f;
    float pdf_uv = 1.0f;

    const DTreeNode *current = &root;
    float r = std::clamp(u1, 0.0f, 0.999999f);

    // DEBUG
    std::cout << "\n--- [DEBUG SAMPLING TRACE] ---\n";
    std::cout << "Input u1 = " << u1 << ", u2 = " << u2 << "\n";

    int depth = 0;
    while (current && !current->is_leaf() && current->sum_radiance > 0.0f)
    {
        std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
        float total_weight = 0.0f;

        for (int i = 0; i < 4; ++i)
        {
            if (current->children[i])
            {
                weights[i] = current->children[i]->sum_radiance;
                total_weight += weights[i];
            }
        }

        if (total_weight <= 0.0f)
            break;

        int chosen_child = 0;
        float r_scaled = r * total_weight;
        float accum = 0.0f;

        for (int i = 0; i < 4; ++i)
        {
            accum += weights[i];
            if (r_scaled <= accum)
            {
                chosen_child = i;
                break;
            }
        }

        float prev_accum = accum - weights[chosen_child];
        r = (weights[chosen_child] > 0.0f) ? (r_scaled - prev_accum) / weights[chosen_child] : 0.5f;
        r = std::clamp(r, 0.0f, 0.999999f);

        float p_child = weights[chosen_child] / total_weight;
        pdf_uv *= (p_child * 4.0f);

        // DEBUG
        std::cout
            << "  [Depth " << depth << "]"
            << " Node " << current
            << " | r = " << r
            << " | chosen child = " << chosen_child
            << " | p_child = " << p_child
            << " | branch factor = " << (p_child * 4.0f)
            << " | pdf_uv = " << pdf_uv
            << "\n";

        float u_mid = 0.5f * (u_min + u_max);
        float v_mid = 0.5f * (v_min + v_max);

        if (chosen_child & 1)
            u_min = u_mid;
        else
            u_max = u_mid;

        if (chosen_child & 2)
            v_min = v_mid;
        else
            v_max = v_mid;

        current = current->children[chosen_child].get();

        depth++;
    }

    // Nudge parameters slightly inwards to guarantee exact cell matching when reprojected
    float final_u = std::clamp(u_min + r * (u_max - u_min), 0.0f, 0.999999f);
    float final_v = std::clamp(v_min + u2 * (v_max - v_min), 0.0f, 0.999999f);

    // DEBUG
    std::cout
        << "Final UV = ("
        << final_u << ", "
        << final_v << ")\n";

    std::cout
        << "Sampler PDF UV = "
        << pdf_uv << "\n";

    vec3 dir = uv_to_dir(final_u, final_v);
    out_pdf = pdf_uv * ActiveMapping::jacobian(dir);

    std::cout
        << "Sampler final PDF = "
        << out_pdf << "\n";

    std::cout << "--------------------------------\n";

    return dir;
}

vec3 DirectionalQuadtree::sample_direction(float u1, float u2, float &out_pdf) const
{
    // DEBUG
    std::array<int, 32> sampled_children{};
    std::array<float, 32> sampled_r{};
    std::array<float, 32> sampled_p{};
    int sampled_depth = 0;
    std::array<float, 32> sampled_u_min{};
    std::array<float, 32> sampled_u_max{};
    std::array<float, 32> sampled_v_min{};
    std::array<float, 32> sampled_v_max{};

    if (!has_distribution)
    {
        out_pdf = 1.0f / (4.0f * M_PI);
        return uv_to_dir(u1, u2);
    }

    float u_min = 0.0f, u_max = 1.0f;
    float v_min = 0.0f, v_max = 1.0f;
    float pdf_uv = 1.0f;
    const DTreeNode *current = &root;

    float rx = std::clamp(u1, 0.0f, 0.999999f);
    float ry = std::clamp(u2, 0.0f, 0.999999f);

    while (current && !current->is_leaf())
    {
        std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
        float total_weight = 0.0f;

        for (int i = 0; i < 4; ++i)
        {
            if (current->children[i])
            {
                weights[i] = current->children[i]->pdf_weight;
                total_weight += weights[i];
            }
        }

        if (total_weight <= 0.0f)
            break;

        // 1. Marginal U selection (Left: children 0 & 2, Right: children 1 & 3)
        float w_left = weights[0] + weights[2];
        float w_right = weights[1] + weights[3];
        float w_u = w_left + w_right;

        if (w_u <= 0.0f)
            break;

        float p_left = w_left / w_u;
        int chosen_u = 0;
        if (rx < p_left)
        {
            chosen_u = 0;
            rx = (p_left > 0.0f) ? (rx / p_left) : 0.5f;
        }
        else
        {
            chosen_u = 1;
            rx = (p_left < 1.0f) ? ((rx - p_left) / (1.0f - p_left)) : 0.5f;
        }
        rx = std::clamp(rx, 0.0f, 0.999999f);

        // 2. Conditional V selection given chosen U (Bottom vs Top)
        float w_bottom = (chosen_u == 0) ? weights[0] : weights[1];
        float w_top = (chosen_u == 0) ? weights[2] : weights[3];
        float w_v = w_bottom + w_top;

        float p_bottom = (w_v > 0.0f) ? (w_bottom / w_v) : 0.5f;
        int chosen_v = 0;
        if (ry < p_bottom)
        {
            chosen_v = 0;
            ry = (p_bottom > 0.0f) ? (ry / p_bottom) : 0.5f;
        }
        else
        {
            chosen_v = 1;
            ry = (p_bottom < 1.0f) ? ((ry - p_bottom) / (1.0f - p_bottom)) : 0.5f;
        }
        ry = std::clamp(ry, 0.0f, 0.999999f);

        int chosen_child = chosen_u + (chosen_v * 2);
        const auto &child = current->children[chosen_child];

        if (!child || weights[chosen_child] <= 0.0f)
        {
            std::cout << "Sample traversal encountered invalid child!" << std::endl;

            break;
        }

        float p_child = weights[chosen_child] / total_weight;
        pdf_uv *= (p_child * 4.0f);

        float u_mid = 0.5f * (u_min + u_max);
        float v_mid = 0.5f * (v_min + v_max);

        if (chosen_child & 1)
            u_min = u_mid;
        else
            u_max = u_mid;

        if (chosen_child & 2)
            v_min = v_mid;
        else
            v_max = v_mid;

        current = child.get();

        // DEBUG
        sampled_u_min[sampled_depth] = u_min;
        sampled_u_max[sampled_depth] = u_max;
        sampled_v_min[sampled_depth] = v_min;
        sampled_v_max[sampled_depth] = v_max;
        sampled_children[sampled_depth] = chosen_child;
        // sampled_r[sampled_depth] = r;
        sampled_p[sampled_depth] = p_child;
        sampled_depth++;
    }

    // Keep the sampled UV safely inside the selected leaf.
    // Avoid rounding onto a quadtree boundary during direction conversion.

    constexpr float UV_MARGIN = 1e-4f;

    float final_u = u_min + std::clamp(rx, UV_MARGIN, 1.0f - UV_MARGIN) * (u_max - u_min);

    float final_v = v_min + std::clamp(ry, UV_MARGIN, 1.0f - UV_MARGIN) * (v_max - v_min);

    vec3 dir = uv_to_dir(final_u, final_v);
    out_pdf = pdf_uv * ActiveMapping::jacobian(dir);

    // DEBUG (FIX)
    float evaluated_pdf = get_pdf(dir);
    if (std::abs(out_pdf - evaluated_pdf) > 1e-2f)
    {
        std::cout << "\n========== SAMPLER MISMATCH ==========\n";
        std::cout << "u1 = " << u1 << "\n";
        std::cout << "u2 = " << u2 << "\n";

        std::cout << "Final UV = ("
                  << final_u << ", "
                  << final_v << ")\n";

        std::cout << "Sampler PDF UV = " << pdf_uv << "\n";
        std::cout << "Sampler PDF    = " << out_pdf << "\n";
        std::cout << "Evaluated PDF  = " << evaluated_pdf << "\n";

        for (int d = 0; d < sampled_depth; ++d)
        {
            std::cout << "  [Depth " << d << "] "
                      << "child = " << sampled_children[d]
                      << ", r = " << sampled_r[d]
                      << ", p_child = " << sampled_p[d]
                      << "\n";
        }
        get_pdf_debug(dir);

        std::setprecision(12);
        float roundtrip_u, roundtrip_v;
        dir_to_uv(dir, roundtrip_u, roundtrip_v);
        if (std::abs(roundtrip_u - final_u) > 0.0f || std::abs(roundtrip_v - final_v) > 0.0f)
        {
            std::cout << "Sampler final UV before mapping = ("
                      << final_u << ", "
                      << final_v << ")\n";
            std::cout << "Roundtrip UV = ("
                      << roundtrip_u << ", "
                      << roundtrip_v << ")\n";
            std::cout << "UV error = ("
                      << roundtrip_u - final_u << ", "
                      << roundtrip_v - final_v << ")\n";
            std::exit(1);
        }

        // std::exit(1);
    }

    return dir;
}

void DirectionalQuadtree::build_distribution(float epsilon_floor)
{
    if (root.sum_radiance <= 0.0f)
        return;

    // Recursively calculate normalized weights and store them safely
    build_distribution_recursive(root, epsilon_floor);

    has_distribution = true;
}

void DirectionalQuadtree::build_distribution_recursive(DTreeNode &node, float epsilon_floor)
{
    if (node.is_leaf())
        return;

    float total_child_radiance = 0.0f;
    for (int i = 0; i < 4; ++i)
    {
        if (node.children[i])
            total_child_radiance += node.children[i]->sum_radiance;
    }

    if (total_child_radiance <= 0.0f)
        return;

    for (int i = 0; i < 4; ++i)
    {
        if (node.children[i])
        {
            float trained_ratio = node.children[i]->sum_radiance / total_child_radiance;
            float blended_ratio = (1.0f - epsilon_floor) * trained_ratio + epsilon_floor * 0.25f;

            // STORE IN A SEPARATE PROBABILITY FIELD (e.g., pdf_weight)
            // DO NOT overwrite node.children[i]->sum_radiance here!
            node.children[i]->pdf_weight = blended_ratio;

            build_distribution_recursive(*node.children[i], epsilon_floor);
        }
    }
}

void DirectionalQuadtree::build_distribution_EMA(float epsilon_floor, float alpha_m)
{
    // Step 1: Recursively apply EMA and compute historical sums from bottom-up or top-down
    apply_ema_and_update_sums(root, alpha_m);

    if (root.sum_radiance <= 0.0f)
        return;

    // Step 2: Recursively calculate normalized weights and store them safely
    build_distribution_recursive(root, epsilon_floor);

    has_distribution = true;
}

void DirectionalQuadtree::apply_ema_and_update_sums(
    DTreeNode &node,
    float alpha_m)
{
    node.sum_radiance =
        (1.0f - alpha_m) * node.sum_radiance + alpha_m * node.current_radiance;

    node.current_radiance = 0.0f;

    if (node.is_leaf())
        return;

    for (int i = 0; i < 4; ++i)
    {
        if (node.children[i])
        {
            apply_ema_and_update_sums(
                *node.children[i], alpha_m);
        }
    }
}

void DirectionalQuadtree::build_distribution_recursive_EMA(DTreeNode &node, float epsilon_floor)
{
    if (node.is_leaf())
        return;

    float total_child_radiance = 0.0f;
    for (int i = 0; i < 4; ++i)
    {
        if (node.children[i])
            total_child_radiance += node.children[i]->sum_radiance; // Uses smoothed historical sum
    }

    if (total_child_radiance <= 0.0f)
        return;

    for (int i = 0; i < 4; ++i)
    {
        if (node.children[i])
        {
            float trained_ratio = node.children[i]->sum_radiance / total_child_radiance;
            float blended_ratio = (1.0f - epsilon_floor) * trained_ratio + epsilon_floor * 0.25f;

            // Store safely in pdf_weight; DO NOT touch sum_radiance here!
            node.children[i]->pdf_weight = blended_ratio;

            build_distribution_recursive_EMA(*node.children[i], epsilon_floor);
        }
    }
}

// 1. Get the total radiance stored at the root
float DirectionalQuadtree::get_root_radiance() const
{
    return root.sum_radiance;
}

// 2. Get total number of nodes (inner nodes + leaves)
int DirectionalQuadtree::get_node_count() const
{
    return get_node_count_recursive(root);
}

int DirectionalQuadtree::get_node_count_recursive(const DTreeNode &node) const
{
    int count = 1; // Count this node
    if (!node.is_leaf())
    {
        for (int i = 0; i < 4; ++i)
        {
            if (node.children[i])
            {
                count += get_node_count_recursive(*node.children[i]);
            }
        }
    }
    return count;
}

int DirectionalQuadtree::get_leaf_count() const
{
    return get_leaf_count_recursive(root);
}

int DirectionalQuadtree::get_leaf_count_recursive(const DTreeNode &node) const
{
    if (node.is_leaf())
    {
        return 1;
    }
    int count = 0;
    for (int i = 0; i < 4; ++i)
    {
        if (node.children[i])
        {
            count += get_leaf_count_recursive(*node.children[i]);
        }
    }
    return count;
}

// 3. Get maximum depth achieved by the tree
int DirectionalQuadtree::get_max_depth() const
{
    return get_max_depth_recursive(root, 1);
}

int DirectionalQuadtree::get_max_depth_recursive(const DTreeNode &node, int current_depth) const
{
    if (node.is_leaf())
    {
        return current_depth;
    }

    int max_d = current_depth;
    for (int i = 0; i < 4; ++i)
    {
        if (node.children[i])
        {
            max_d = std::max(max_d, get_max_depth_recursive(*node.children[i], current_depth + 1));
        }
    }
    return max_d;
}

// Helper function to recursively draw quadtree cell outlines
inline void draw_node_bounds(const DTreeNode *node, std::vector<float> &overlay_grid,
                             int res, float u0, float u1, float v0, float v1)
{
    if (!node)
        return;

    // Convert UV coordinates [0, 1] to pixel grid coordinates
    int x0 = std::clamp(static_cast<int>(u0 * res), 0, res - 1);
    int x1 = std::clamp(static_cast<int>(u1 * res), 0, res - 1);
    int y0 = std::clamp(static_cast<int>(v0 * res), 0, res - 1);
    int y1 = std::clamp(static_cast<int>(v1 * res), 0, res - 1);

    // Draw cell boundary pixels (setting outline intensity flag)
    for (int x = x0; x <= x1; ++x)
    {
        overlay_grid[y0 * res + x] = -1.0f; // Top edge
        overlay_grid[y1 * res + x] = -1.0f; // Bottom edge
    }
    for (int y = y0; y <= y1; ++y)
    {
        overlay_grid[y * res + x0] = -1.0f; // Left edge
        overlay_grid[y * res + x1] = -1.0f; // Right edge
    }

    // If internal node, recurse into children
    if (!node->is_leaf())
    {
        float u_mid = 0.5f * (u0 + u1);
        float v_mid = 0.5f * (v0 + v1);

        draw_node_bounds(node->children[0].get(), overlay_grid, res, u0, u_mid, v0, v_mid);
        draw_node_bounds(node->children[1].get(), overlay_grid, res, u_mid, u1, v0, v_mid);
        draw_node_bounds(node->children[2].get(), overlay_grid, res, u0, u_mid, v_mid, v1);
        draw_node_bounds(node->children[3].get(), overlay_grid, res, u_mid, u1, v_mid, v1);
    }
}

void DirectionalQuadtree::export_heatmap_ppm(const std::string &filename) const
{
    constexpr int res = 128; // 128x128 resolution grid
    std::vector<float> pdf_grid(res * res, 0.0f);
    float max_pdf = 0.0f;

    // 1. Evaluate directional PDFs across grid
    for (int y = 0; y < res; ++y)
    {
        float v = (y + 0.5f) / static_cast<float>(res);
        for (int x = 0; x < res; ++x)
        {
            float u = (x + 0.5f) / static_cast<float>(res);

            vec3 dir = uv_to_dir(u, v);
            float pdf = get_pdf(dir); // PDF per steradian (solid angle)

            pdf_grid[y * res + x] = pdf;
            if (pdf > max_pdf)
                max_pdf = pdf;
        }
    }

    // 2. Overlay boundary outlines onto pdf_grid (-1.0f flags boundary pixels)
    draw_node_bounds(get_root(), pdf_grid, res, 0.0f, 1.0f, 0.0f, 1.0f);

    // 3. Write PPM Header
    std::ofstream out(filename); // Plain text output mode
    if (!out.is_open())
        return;

    out << "P3\n"
        << res << " " << res << "\n255\n";

    // 4. Output single stream of pixels
    for (int i = 0; i < res * res; ++i)
    {
        // Draw boundary line in white
        if (pdf_grid[i] < 0.0f)
        {
            out << "255 255 255\n";
            continue;
        }

        float val = (max_pdf > 0.0f) ? (pdf_grid[i] / max_pdf) : 0.0f;

        // Normal Heatmap Colors: Dark Blue -> Cyan -> Yellow -> Red -> White
        int r = static_cast<int>(std::clamp(val * 255.0f * 1.5f, 0.0f, 255.0f));
        int g = static_cast<int>(std::clamp((val - 0.25f) * 255.0f * 1.3f, 0.0f, 255.0f));
        int b = static_cast<int>(std::clamp((0.75f - val) * 255.0f * 2.0f, 0.0f, 255.0f));

        out << r << " " << g << " " << b << "\n";
    }

    out.close();
    std::cout << "[DIAGNOSTIC] Saved ASCII Heatmap: " << filename
              << " | Peak PDF: " << max_pdf << " sr^-1" << std::endl;
}

// 2. Prints the ASCII Tree Topology & Node Probabilities to std::cout
void DirectionalQuadtree::print_tree_structure(const DTreeNode *node, int depth) const
{
    if (!node)
        node = &root;

    std::string indent(depth * 2, ' ');
    std::cout << indent << "└─ [Depth " << depth << "] Leaf=" << (node->is_leaf() ? "YES" : "NO")
              << " | sum_radiance/p_i = " << node->sum_radiance << std::endl;

    if (!node->is_leaf())
    {
        for (int i = 0; i < 4; ++i)
        {
            if (node->children[i])
            {
                std::cout << indent << "   Child [" << i << "]:" << std::endl;
                print_tree_structure(node->children[i].get(), depth + 1);
            }
        }
    }
}