/* #ifndef OBJECT_H
#define OBJECT_H

#include "triangle.h"
#include "aabb.h"
#include "mat4.h"

#include ".\glm\glm.hpp"
#include <vector>

struct object
{
    std::vector<triangle> mesh;
    aabb box;
    material_type mt;

    glm::mat4 transform;
    glm::mat4 inverse_transform;
    glm::mat4 normal_transfrom;

    object() = default;

    object(std::vector<triangle> mesh, const aabb &box, material_type mt, const glm::mat4 &transform)
        : mesh(std::move(mesh)),
          box(box),
          mt(mt),
          transform(transform),
          inverse_transform(glm::inverse(transform)),
          normal_transfrom(glm::transpose(glm::inverse(transform))) {}
};

#endif */
