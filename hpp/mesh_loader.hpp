#ifndef MESH_LOADER_HPP
#define MESH_LOADER_HPP

#include "vec3.hpp"
#include "triangle.hpp"
#include "aabb.hpp"
#include "material.hpp"
#include "color.hpp"
#include "light_source.hpp"

#include "../../glm/glm.hpp"
#include "../../glm/gtc/matrix_transform.hpp"

#include <vector>

struct object
{
    std::vector<triangle> mesh;
    aabb box;
    material_preset mt; // material_type
    std::vector<area_light> lights;

    glm::mat4 transform;
    glm::mat4 inverse_transform;
    glm::mat4 transpose_transform;

    object();
    // object(const std::vector<triangle> &m, const aabb &b, material_preset mt, const glm::mat4 &tform, const std::vector<area_light> &als); // material_type
    object(std::vector<triangle> mesh,
           aabb box,
           glm::mat4 transform,
           const std::vector<area_light> &lights);
};

std::vector<std::vector<area_light>> cluster_light_triangles(const std::vector<area_light> &tris);
area_light merge_light_triangles(const std::vector<area_light> &tris);

object obj_to_mesh(const char *objpath,
                   const char *mtlpath,
                   vec3 offset,
                   material_preset preset,
                   material_type mt, // material_type
                   glm::mat4 tmat,
                   color c);

#endif // MESH_LOADER_HPP
