#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include "vec3.h"
#include "triangle.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include ".\tinyobjloader\tiny_obj_loader.h"
#include "aabb.h"

// #include "object.h"

#include ".\glm\glm.hpp"
#include ".\glm\gtc\matrix_transform.hpp"

#include <cfloat>

struct object
{
    std::vector<triangle> mesh;
    aabb box;
    material_type mt;
    // Transformations
    glm::mat4 transform;
    glm::mat4 inverse_transform;
    glm::mat4 transpose_transform;

    object() {}

    object(const std::vector<triangle> &m, const aabb &b, material_type mt, const glm::mat4 &tform)
        : mesh(m), box(b), mt(mt), transform(tform)
    {
        inverse_transform = glm::inverse(tform);
        transpose_transform = glm::transpose(inverse_transform);
    }

    /* bool hit(const ray &world_ray, float t_min, float t_max, hit_info &out_hit) const
    {
        // Transform ray to object space
        vec3 origin_os = glm::vec3(inverse_transform * glm::vec4(world_ray.origin(), 1.0f));
        vec3 dir_os = glm::vec3(inverse_transform * glm::vec4(world_ray.direction(), 0.0f));
        ray ray_os(origin_os, dir_os);

        hit_info local_hit;
        if (bvh_tree.hit(ray_os, t_min, t_max, local_hit))
        {
            // Transform intersection info back to world space
            out_hit.p = glm::vec3(transform * glm::vec4(local_hit.p, 1.0f));
            out_hit.normal = glm::normalize(glm::vec3(transpose_transform) * glm::vec4(local_hit.normal, 0.0f));
            out_hit.t = local_hit.t;
            out_hit.mt = mt;
            return true;
        }
        return false;
    } */
};

/* inline bool triangle::hit(const ray &world_ray, float t_min, float t_max, hit_info &hi) // TRANSFORMATION (Brute-force)
{
    if (!obj_ptr)
        return false;

    glm::vec3 o = to_glm(world_ray.origin());
    glm::vec3 d = to_glm(world_ray.direction());

    glm::vec3 origin_os = glm::vec3(obj_ptr->inverse_transform * glm::vec4(o, 1.0f));
    glm::vec3 direction_os = glm::vec3(obj_ptr->inverse_transform * glm::vec4(d, 0.0f));
    ray local_ray(from_glm(origin_os), from_glm(direction_os));

    const float eps = 1e-5f;
    vec3 edge10 = v1 - v0;
    vec3 edge20 = v2 - v0;

    vec3 pvec = cross(local_ray.direction(), edge20);
    float det = dot(edge10, pvec);
    if (std::abs(det) < eps)
        return false;

    float inv_det = 1.0f / det;
    vec3 tvec = local_ray.origin() - v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < -eps || u > 1.0f + eps)
        return false;

    vec3 qvec = cross(tvec, edge10);
    float v = dot(local_ray.direction(), qvec) * inv_det;
    if (v < -eps || u + v > 1.0f + eps)
        return false;

    float t = dot(edge20, qvec) * inv_det;
    if (t < t_min || t > t_max)
        return false;

    vec3 normal_os = unit_vector((1 - u - v) * n0 + u * n1 + v * n2);
    vec3 hit_os = local_ray.at(t);
    vec3 hit_world = from_glm(glm::vec3(obj_ptr->transform * glm::vec4(to_glm(hit_os), 1.0f)));
    vec3 normal_world = from_glm(glm::normalize(glm::vec3(obj_ptr->transpose_transform * glm::vec4(to_glm(normal_os), 0.0f))));

    hi.t = t;
    hi.p = hit_world;
    hi.normal = normal_world;
    hi.mat = this->mat;
    hi.mt = this->mt;
    return true;
} */

object obj_to_mesh(const char *objpath, const char *mtlpath, vec3 offset, material_type mt, glm::mat4 tmat, color c)
{
    std::vector<triangle> triangle_mesh;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    point3 min_point(FLT_MAX, FLT_MAX, FLT_MAX);
    point3 max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    const float AABB_EPS = 1e-5f;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objpath, mtlpath);

    if (!warn.empty())
        std::cout << "warning: " << warn << std::endl;
    if (!err.empty())
        std::cerr << "error: " << err << std::endl;
    if (!ret)
        exit(1);

    int total_triangles = 0;
    for (const auto &shape : shapes)
    {
        size_t index_offset = 0;

        int counter_triangles = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3)
                continue;

            point3 v[3];
            vec3 n[3];
            int mat_id = shape.mesh.material_ids[f];
            material mat;

            for (int i = 0; i < 3; i++)
            {
                // get triangle vertices
                tinyobj::index_t idx = shape.mesh.indices[index_offset + i];
                float vx = attrib.vertices[3 * idx.vertex_index + 0] + offset[0];
                float vy = attrib.vertices[3 * idx.vertex_index + 1] + offset[1];
                float vz = attrib.vertices[3 * idx.vertex_index + 2] + offset[2];

                glm::vec4 v4(vx, vy, vz, 1.0f); // 1.0 for vertex
                glm::vec4 transformed_v4 = tmat * v4;
                v[i] = point3(transformed_v4.x, transformed_v4.y, transformed_v4.z);
                // v[i] = point3(vx, vy, vz);

                float tx = transformed_v4.x;
                float ty = transformed_v4.y;
                float tz = transformed_v4.z;

                // Update AABB with transformed vertex
                min_point = point3(std::min(min_point.x(), tx),
                                   std::min(min_point.y(), ty),
                                   std::min(min_point.z(), tz));
                max_point = point3(std::max(max_point.x(), tx),
                                   std::max(max_point.y(), ty),
                                   std::max(max_point.z(), tz));

                // get normals of vertices
                if (idx.normal_index >= 0)
                {
                    float nx = attrib.normals[3 * idx.normal_index + 0];
                    float ny = attrib.normals[3 * idx.normal_index + 1];
                    float nz = attrib.normals[3 * idx.normal_index + 2];

                    glm::vec4 n4(nx, ny, nz, 0.0f); // 0.0f for vector
                    glm::vec4 transformed_n4 = glm::transpose(glm::inverse(tmat)) * n4;
                    n[i] = unit_vector(vec3(transformed_n4.x, transformed_n4.y, transformed_n4.z));
                    // n[i] = vec3(nx, ny, nz);
                }
                else
                {
                    std::cout << "fallback normal" << '\n';
                    n[i] = unit_vector(cross(v[1] - v[0], v[2] - v[0])); // fallback
                }

                // get material properties of triangle
                if (mat_id >= 0 && mat_id < materials.size())
                {
                    const auto &m = materials[mat_id];
                    mat.ambient = color(m.ambient[0], m.ambient[1], m.ambient[2]);
                    mat.diffuse = color(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
                    mat.specular = color(m.specular[0], m.specular[1], m.specular[2]);
                    mat.spec_exponent = m.shininess;

                    /* if (m.emission[0] > 0 || m.emission[1] > 0 || m.emission[2] > 0)
                    {
                        color emission = color(m.emission[0], m.emission[1], m.emission[2]);

                        // you can represent this as a "light_source" positioned at triangle center
                        point3 center = (v[0] + v[1] + v[2]) / 3.0f;
                        lights.push_back(light_source{center - vec3(0, 1e-3f, 0), emission, 1.0f});

                        // optionally: also add the triangle to world with pure emissive material
                        mat.diffuse = emission;
                        mat.specular = color(0, 0, 0);
                    } */
                }
                // more properties
                else // fallback
                {
                    mat.ambient = color(fmax(0.0f, c.x() - 0.1f), fmax(0.0f, c.y() - 0.1f), fmax(0.0f, c.z() - 0.1f));
                    mat.diffuse = color(c.x(), c.y(), c.z());
                    // mat.ambient = color(0.2, 0.1, 0.3);
                    // mat.diffuse = color(0.3, 0.1, 0.7);
                    mat.specular = color(0.9, 0.9, 0.9);
                    mat.spec_exponent = 16.0f;
                }

                if (mt == material_type::mirror)
                {
                    mat.reflectivity = 1.0f;
                    mat.transparency = 0.0f;
                    mat.ior = 1.0f;
                }
                else if (mt == material_type::transparent)
                {
                    mat.reflectivity = 0.05f; // small reflection like glass
                    mat.transparency = 0.95f; // mostly transparent
                    mat.transmission_color = color(0.8, 0.2, 0.2);
                    mat.ior = 1.5f; // index of refraction (default: glass)
                }
                else
                { // diffuse
                    mat.reflectivity = 0.0f;
                    mat.transparency = 0.0f;
                    mat.ior = 1.0f;
                }
                // mat.transparency = 1.0f - mat.reflectivity;
            }

            triangle_mesh.push_back(triangle(v[0], v[1], v[2], n[0], n[1], n[2], mat, mt, nullptr));
            index_offset += fv;

            counter_triangles++;
        }
        std::cout << "number of triangles:" << counter_triangles << '\n';
        total_triangles += counter_triangles;
    }

    // Optional: inflate slightly for numerical robustness
    min_point = min_point - point3(AABB_EPS, AABB_EPS, AABB_EPS);
    max_point = max_point + point3(AABB_EPS, AABB_EPS, AABB_EPS);

    std::cout << "Total triangles:" << total_triangles << '\n';

    // object obj{triangle_mesh, aabb(min_point, max_point), mt, tmat}; // point3(0, 0, 0), point3(0, 0, 0) // min_point, max_point
    object obj(triangle_mesh, aabb(min_point, max_point), mt, tmat);

    for (triangle &tri : obj.mesh)
    {
        tri.set_obj(&obj);
    }
    return obj;
}

#endif