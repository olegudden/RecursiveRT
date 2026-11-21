#ifndef MESH_LOADER_HPP
#define MESH_LOADER_HPP

#include "light_source.hpp"
#include "texture.hpp"

#include <vector>

struct object
{
    std::vector<triangle> mesh;
    aabb box;
    // material_preset mt; // material_type
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

object obj_to_mesh(const char *objpath, // std::shared_ptr<bvh>
                   const char *mtlpath,
                   vec3 offset,
                   glm::mat4 tmat,
                   color c,
                   TextureManager &tex_manager); // std::vector<area_light> &out_lights);

#endif // MESH_LOADER_HPP
