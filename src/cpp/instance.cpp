#include "../hpp/instance.hpp"

instance::instance(std::shared_ptr<bvh> local_bvh, const glm::mat4 &transform)
    : local_bvh(local_bvh), tmat(transform)
{
    inv_tmat = glm::inverse(tmat);
    normal_mat = glm::transpose(glm::mat3(inv_tmat));

    // Safely get local box from the root node (handles null root if BVH is empty)
    aabb local_box = (local_bvh && local_bvh->root) ? local_bvh->root->box : aabb();

    // Transform local AABB to world space
    world_box = compute_world_aabb(local_box, tmat);
}

bool instance::hit(const ray &r, float t_min, float t_max, hit_info &hi) const
{
    // 1. Transform World Ray -> Object (Local) Ray
    glm::vec4 local_o = inv_tmat * glm::vec4(r.origin().x(), r.origin().y(), r.origin().z(), 1.0f);
    glm::vec4 local_d = inv_tmat * glm::vec4(r.direction().x(), r.direction().y(), r.direction().z(), 0.0f);

    ray local_ray(
        point3(local_o.x, local_o.y, local_o.z),
        vec3(local_d.x, local_d.y, local_d.z));

    // 2. Test intersection on untransformed mesh BVH
    if (!local_bvh->hit(local_ray, t_min, t_max, hi))
    {
        return false;
    }

    // 3. Convert Hit Point & Normal back to World Space
    hi.p = r.at(hi.t);

    // Transform local normal using inverse-transpose matrix
    glm::vec3 world_n = normal_mat * glm::vec3(hi.normal.x(), hi.normal.y(), hi.normal.z());
    hi.normal = unit_vector(vec3(world_n.x, world_n.y, world_n.z));

    return true;
}

aabb instance::compute_world_aabb(const aabb &local_box, const glm::mat4 &m)
{
    point3 min_p(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
    point3 max_p(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());

    // Transform all 8 corners of the local bounding box to find the new world extent
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                float x = i ? local_box.max_point().x() : local_box.min_point().x();
                float y = j ? local_box.max_point().y() : local_box.min_point().y();
                float z = k ? local_box.max_point().z() : local_box.min_point().z();

                glm::vec4 w_corner = m * glm::vec4(x, y, z, 1.0f);

                min_p = point3(std::min(min_p.x(), w_corner.x),
                               std::min(min_p.y(), w_corner.y),
                               std::min(min_p.z(), w_corner.z));

                max_p = point3(std::max(max_p.x(), w_corner.x),
                               std::max(max_p.y(), w_corner.y),
                               std::max(max_p.z(), w_corner.z));
            }
        }
    }
    return aabb(min_p, max_p);
}