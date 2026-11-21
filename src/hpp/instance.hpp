#ifndef INSTANCE_HPP
#define INSTANCE_HPP

#include <glm/glm.hpp>
#include <memory>
#include <algorithm>
#include <limits>

// Replace/include your actual header files for ray, hit_info, aabb, bvh
#include "ray.hpp"
#include "hittable.hpp"
#include "aabb.hpp"
#include "bvh.hpp"

class instance
{
public:
    std::shared_ptr<bvh> local_bvh; // BLAS
    glm::mat4 tmat;                 // Object-to-World
    glm::mat4 inv_tmat;             // World-to-Object (inverse)
    glm::mat3 normal_mat;           // Inverse transpose (for normals)
    aabb world_box;                 // Transformed AABB (world)

    instance(std::shared_ptr<bvh> local_bvh, const glm::mat4 &transform);

    bool hit(const ray &r, float t_min, float t_max, hit_info &hi) const;
    aabb box() const { return world_box; }

private:
    static aabb compute_world_aabb(const aabb &local_box, const glm::mat4 &m);
};

#endif // INSTANCE_H