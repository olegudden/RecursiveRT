#include "../hpp/bvh.hpp"
#include <algorithm>

// ----------------------- bvh_stats -----------------------
void bvh_stats::printStats() const
{
    std::cout << "Total nodes: " << total_nodes << "\n";
    std::cout << "Leaf nodes: " << leaf_nodes << "\n";
    std::cout << "Triangles per leaf (min/avg/max): "
              << min_tris_per_leaf << " / "
              << (sum_tris_per_leaf / leaf_count) << " / "
              << max_tris_per_leaf << "\n";
    std::cout << "Tree depth (min/max): " << min_depth << " / " << max_depth << "\n";
    std::cout << "Triangle tests: " << total_triangle_tests << "\n";
    std::cout << "Total surface area: " << total_surface_area << "\n";
    std::cout << "Leafs surface area: " << leafs_surface_area << "\n";
    std::cout << "Overlap volume siblings: " << overlap_volume << "\n";
    std::cout << "Build time (ms): " << build_time_ms << "\n";
    std::cout << "Traverse time (ms): " << traverse_time_ms << "\n";
}

void bvh_stats::collect_bvh_stats(const bvh_node *node, int depth, bvh_stats &stats)
{
    if (!node)
        return;

    stats.total_nodes++;
    stats.total_surface_area += node->box.surface_area();

    if (node->left && node->right)
        stats.overlap_volume += aabb::intersection_volume(node->left->box, node->right->box);

    if (!node->left && !node->right) // Leaf
    {
        stats.min_depth = std::min(stats.min_depth, depth);
        stats.max_depth = std::max(stats.max_depth, depth);
        stats.leaf_nodes++;
        int count = static_cast<int>(node->triangles.size());
        stats.min_tris_per_leaf = std::min(stats.min_tris_per_leaf, count);
        stats.max_tris_per_leaf = std::max(stats.max_tris_per_leaf, count);
        stats.sum_tris_per_leaf += count;
        stats.leaf_count++;
        stats.leafs_surface_area += node->box.surface_area();
    }

    collect_bvh_stats(node->left.get(), depth + 1, stats);
    collect_bvh_stats(node->right.get(), depth + 1, stats);
}

// ----------------------- bvh -----------------------
bool bvh::hit(const ray &r, float t_min, float t_max, hit_info &hi) const
{
    return hit_node(root.get(), r, t_min, t_max, hi);
}

bool bvh::hit_node(bvh_node *node, const ray &r, float t_min, float t_max, hit_info &final_hi) const
{
    if (!node)
        return false;

    stats.total_aabb_tests++;

    // hit_info dummy; // here
    if (!node->box.hit(r, t_min, t_max, final_hi))
        return false;

    bool hit_anything = false;
    float closest_so_far = t_max;
    hit_info temp_hi;

    if (node->left || node->right)
    {
        hit_info left_hi, right_hi;
        bool hit_left = node->left && hit_node(node->left.get(), r, t_min, closest_so_far, left_hi);
        if (hit_left)
        {
            closest_so_far = left_hi.t;
            final_hi = left_hi;
            hit_anything = true;
        }
        bool hit_right = node->right && hit_node(node->right.get(), r, t_min, closest_so_far, right_hi);
        if (hit_right && (!hit_left || right_hi.t < final_hi.t))
        {
            final_hi = right_hi;
            hit_anything = true;
        }
    }
    else
    {
        for (const auto &tri : node->triangles)
        {
            stats.total_triangle_tests++;
            if (tri.hit(r, t_min, closest_so_far, temp_hi))
            {
                closest_so_far = temp_hi.t;
                final_hi = temp_hi;
                hit_anything = true;
            }
        }
    }
    return hit_anything;
}

// ----------------------- BVH Builders -----------------------

std::unique_ptr<bvh_node> bvh::build_bvh_binned_sah(std::vector<triangle> &tris, int depth)
{
    auto node = std::make_unique<bvh_node>();
    node->box = aabb(tris);

    // base case: small leaf or max depth
    if (tris.size() <= 10 || depth >= 20)
    {
        node->triangles = tris;
        return node;
    }

    // number of buckets (clamped by triangle count)
    int NUM_BUCKETS = 12;

    // SAH cost constants
    const float C_trav = 0.125f;
    const float C_isect = 1.0f;

    // centroid bounds
    vec3 min_c = tris[0].centroid();
    vec3 max_c = tris[0].centroid();
    for (const triangle &tri : tris)
    {
        vec3 c = tri.centroid();
        min_c = min_vec(min_c, c);
        max_c = max_vec(max_c, c);
    }

    vec3 extent = max_c - min_c;
    aabb centroid_bounds(min_c, max_c);

    int best_axis = -1;
    int best_split = -1;
    float best_cost = C_isect * (float)tris.size();

    for (int axis = 0; axis < 3; axis++)
    {
        if (extent[axis] == 0.0f)
            continue;

        struct Bucket
        {
            int triangleCount = 0;
            aabb bounds;
            bool empty = true;
        };
        std::vector<Bucket> buckets(NUM_BUCKETS);

        // assign triangles to buckets
        for (triangle &tri : tris)
        {
            float centroid = tri.centroid()[axis];
            int b = std::min(NUM_BUCKETS - 1,
                             int(NUM_BUCKETS * (centroid - min_c[axis]) / extent[axis]));

            aabb tri_box = aabb::from_triangle(tri);
            if (buckets[b].empty)
            {
                buckets[b].bounds = tri_box;
                buckets[b].empty = false;
            }
            else
            {
                buckets[b].bounds = aabb::surrounding_box(buckets[b].bounds, tri_box);
            }
            buckets[b].triangleCount++;
        }

        // precompute prefix sums of left/right partitions
        std::vector<aabb> left_bounds(NUM_BUCKETS - 1);
        std::vector<int> left_count(NUM_BUCKETS - 1);
        std::vector<aabb> right_bounds(NUM_BUCKETS - 1);
        std::vector<int> right_count(NUM_BUCKETS - 1);

        aabb left_box;
        int count = 0;
        for (int i = 0; i < NUM_BUCKETS - 1; i++)
        {
            if (i == 0)
            {
                if (!buckets[i].empty)
                    left_box = buckets[i].bounds;
                count = buckets[i].triangleCount;
            }
            else
            {
                if (!buckets[i].empty)
                    left_box = aabb::surrounding_box(left_box, buckets[i].bounds);
                count += buckets[i].triangleCount;
            }
            left_bounds[i] = left_box;
            left_count[i] = count;
        }

        aabb right_box;
        count = 0;
        for (int i = NUM_BUCKETS - 1; i > 0; i--)
        {
            if (i == NUM_BUCKETS - 1)
            {
                if (!buckets[i].empty)
                    right_box = buckets[i].bounds;
                count = buckets[i].triangleCount;
            }
            else
            {
                if (!buckets[i].empty)
                    right_box = aabb::surrounding_box(right_box, buckets[i].bounds);
                count += buckets[i].triangleCount;
            }
            right_bounds[i - 1] = right_box;
            right_count[i - 1] = count;
        }

        float parent_area = node->box.surface_area();
        for (int i = 0; i < NUM_BUCKETS - 1; i++)
        {
            if (left_count[i] == 0 || right_count[i] == 0)
                continue;

            float cost = C_trav +
                         (left_bounds[i].surface_area() / parent_area) * (float)left_count[i] * C_isect +
                         (right_bounds[i].surface_area() / parent_area) * (float)right_count[i] * C_isect;

            if (cost < best_cost)
            {
                best_cost = cost;
                best_axis = axis;
                best_split = i;
            }
        }
    }

    // no valid split -> leaf
    if (best_axis == -1)
    {
        node->triangles = tris;
        return node;
    }

    // partition triangles
    std::vector<triangle> left_tris, right_tris;
    float axis_min = min_c[best_axis];
    float axis_extent = extent[best_axis];

    for (triangle &tri : tris)
    {
        float centroid = tri.centroid()[best_axis];
        int b = std::min(NUM_BUCKETS - 1,
                         int(NUM_BUCKETS * (centroid - axis_min) / axis_extent));

        if (b <= best_split)
            left_tris.push_back(tri);
        else
            right_tris.push_back(tri);
    }

    // fallback for degenerate splits
    if (left_tris.empty() || right_tris.empty())
    {
        std::sort(tris.begin(), tris.end(),
                  [best_axis](const triangle &a, const triangle &b)
                  {
                      return a.centroid()[best_axis] < b.centroid()[best_axis];
                  });
        size_t mid = tris.size() / 2;
        left_tris.assign(tris.begin(), tris.begin() + mid);
        right_tris.assign(tris.begin() + mid, tris.end());
    }

    node->left = build_bvh_binned_sah(left_tris, depth + 1);
    node->right = build_bvh_binned_sah(right_tris, depth + 1);
    return node;
}

std::unique_ptr<bvh_node> bvh::build_bvh_cycle_axis_equal(std::vector<triangle> &tris, int depth)
{
    auto node = std::make_unique<bvh_node>();
    node->box = aabb(tris);

    if (tris.size() <= 10 || depth >= 20)
    {
        node->triangles = tris;
        return node;
    }

    int axis = depth % 3; // cycle through axis, TODO: try other methods
    std::vector<triangle> sorted_tris = tris;

    std::sort(sorted_tris.begin(), sorted_tris.end(), [axis](const triangle &a, const triangle &b)
              { return a.centroid()[axis] < b.centroid()[axis]; });

    size_t mid = sorted_tris.size() / 2;
    std::vector<triangle> left_tris(sorted_tris.begin(), sorted_tris.begin() + mid);
    std::vector<triangle> right_tris(sorted_tris.begin() + mid, sorted_tris.end());

    node->left = build_bvh_cycle_axis_equal(left_tris, depth + 1);
    node->right = build_bvh_cycle_axis_equal(right_tris, depth + 1);

    return node;
}

std::unique_ptr<bvh_node> bvh::build_bvh_longest_axis_midpoint(std::vector<triangle> &tris, int depth)
{
    auto node = std::make_unique<bvh_node>();
    node->box = aabb(tris);

    // Leaf condition
    if (tris.size() <= 10 || depth >= 20)
    {
        node->triangles = tris;
        return node;
    }

    // 1. Find longest axis
    vec3 extent = node->box.max_point() - node->box.min_point();
    int axis = 0;
    if (extent.y() > extent.x() && extent.y() > extent.z())
        axis = 1;
    else if (extent.z() > extent.x() && extent.z() > extent.y())
        axis = 2;

    // 2. Find midpoint along that axis
    float midpoint = 0.5f * (node->box.min_point()[axis] + node->box.max_point()[axis]);

    // 3. Partition into left/right sets by spatial midpoint
    std::vector<triangle> left_tris;
    std::vector<triangle> right_tris;
    left_tris.reserve(tris.size());
    right_tris.reserve(tris.size());

    for (auto &tri : tris)
    {
        if (tri.centroid()[axis] < midpoint)
            left_tris.push_back(tri);
        else
            right_tris.push_back(tri);
    }

    // Handle degenerate split (all tris go to one side)
    if (left_tris.empty() || right_tris.empty())
    {
        // fallback: split evenly by count
        size_t mid = tris.size() / 2;
        std::sort(tris.begin(), tris.end(), [axis](const triangle &a, const triangle &b)
                  { return a.centroid()[axis] < b.centroid()[axis]; });
        left_tris.assign(tris.begin(), tris.begin() + mid);
        right_tris.assign(tris.begin() + mid, tris.end());
    }

    node->left = build_bvh_longest_axis_midpoint(left_tris, depth + 1);
    node->right = build_bvh_longest_axis_midpoint(right_tris, depth + 1);

    return node;
}

std::unique_ptr<bvh_node> bvh::build_bvh_exact_sah(std::vector<triangle> &tris, int depth)
{
    auto node = std::make_unique<bvh_node>();
    node->box = aabb(tris);

    // Leaf criteria
    if (tris.size() <= 10 || depth >= 20)
    {
        node->triangles = tris;
        return node;
    }

    const float C_trav = 0.125f;
    const float C_isect = 1.0f;
    float best_cost = std::numeric_limits<float>::infinity();
    int best_axis = -1;
    int best_split = -1;

    float parent_area = node->box.surface_area();
    if (parent_area == 0.0f)
    {
        node->triangles = tris;
        return node;
    }

    // Try each axis
    for (int axis = 0; axis < 3; ++axis)
    {
        // Sort by centroid
        std::sort(tris.begin(), tris.end(),
                  [axis](const triangle &a, const triangle &b)
                  {
                      return a.centroid()[axis] < b.centroid()[axis];
                  });

        // Precompute left
        std::vector<aabb> left_bounds(tris.size());
        std::vector<int> left_count(tris.size());
        aabb left_box;
        for (size_t i = 0; i < tris.size(); ++i)
        {
            left_box = (i == 0) ? aabb::from_triangle(tris[i])
                                : aabb::surrounding_box(left_box, aabb::from_triangle(tris[i]));
            left_bounds[i] = left_box;
            left_count[i] = (int)i + 1;
        }

        // Precompute right
        std::vector<aabb> right_bounds(tris.size());
        std::vector<int> right_count(tris.size());
        aabb right_box;
        for (int i = (int)tris.size() - 1; i >= 0; --i)
        {
            right_box = (i == (int)tris.size() - 1) ? aabb::from_triangle(tris[i])
                                                    : aabb::surrounding_box(right_box, aabb::from_triangle(tris[i]));
            right_bounds[i] = right_box;
            right_count[i] = (int)tris.size() - i;
        }

        // Evaluate splits
        for (size_t i = 0; i < tris.size() - 1; ++i)
        {
            if (left_count[i] == 0 || right_count[i + 1] == 0)
                continue;

            float cost = C_trav +
                         (left_bounds[i].surface_area() / parent_area) * (left_count[i] * C_isect) +
                         (right_bounds[i + 1].surface_area() / parent_area) * (right_count[i + 1] * C_isect);

            if (cost < best_cost)
            {
                best_cost = cost;
                best_axis = axis;
                best_split = (int)i;
            }
        }
    }

    // Check against making a leaf
    float leaf_cost = tris.size() * C_isect;
    if (best_axis == -1 || best_cost >= leaf_cost)
    {
        node->triangles = tris;
        return node;
    }

    // Partition along best split
    std::sort(tris.begin(), tris.end(),
              [best_axis](const triangle &a, const triangle &b)
              {
                  return a.centroid()[best_axis] < b.centroid()[best_axis];
              });

    std::vector<triangle> left_tris(tris.begin(), tris.begin() + best_split + 1);
    std::vector<triangle> right_tris(tris.begin() + best_split + 1, tris.end());

    // Degenerate fallback
    if (left_tris.empty() || right_tris.empty())
    {
        size_t mid = tris.size() / 2;
        left_tris.assign(tris.begin(), tris.begin() + mid);
        right_tris.assign(tris.begin() + mid, tris.end());
    }

    node->left = build_bvh_exact_sah(left_tris, depth + 1);
    node->right = build_bvh_exact_sah(right_tris, depth + 1);

    return node;
}