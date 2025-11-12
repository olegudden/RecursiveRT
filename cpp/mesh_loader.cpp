#include "../hpp/mesh_loader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "..\..\tinyobjloader\tiny_obj_loader.h"

#include <iostream>
#include <cfloat>

object::object() = default;

object::object(const std::vector<triangle> &m, const aabb &b, material_type mt, const glm::mat4 &tform, const std::vector<area_light> &als)
    : mesh(m), box(b), mt(mt), transform(tform), lights(als)
{
    inverse_transform = glm::inverse(tform);
    transpose_transform = glm::transpose(inverse_transform);
}

area_light merge_light_triangles(const std::vector<area_light> &tris)
{
    area_light merged;
    if (tris.empty())
        return merged;

    // Compute average normal (robust against rounding)
    vec3 avg_normal(0, 0, 0);
    for (const auto &t : tris)
        avg_normal += unit_vector(t.normal);
    avg_normal = unit_vector(avg_normal / float(tris.size()));

    // Gather all triangle vertices
    std::vector<point3> all_points;
    for (const auto &t : tris)
    {
        all_points.push_back(t.origin);
        all_points.push_back(t.origin + t.e1);
        all_points.push_back(t.origin + t.e2);
    }

    // Project all points into a 2D coordinate system in the light’s plane
    vec3 tangent = unit_vector(perpendicular(avg_normal));
    vec3 bitangent = cross(avg_normal, tangent);

    std::vector<std::pair<float, float>> uv_coords;
    uv_coords.reserve(all_points.size());
    for (const auto &p : all_points)
    {
        float u = dot(p, tangent);
        float v = dot(p, bitangent);
        uv_coords.emplace_back(u, v);
    }

    // Compute min/max in 2D projection
    float min_u = FLT_MAX, max_u = -FLT_MAX;
    float min_v = FLT_MAX, max_v = -FLT_MAX;
    for (auto [u, v] : uv_coords)
    {
        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    // Reconstruct rectangle corners from the 2D bounds
    point3 origin_world = tangent * min_u + bitangent * min_v;
    vec3 e1 = tangent * (max_u - min_u);
    vec3 e2 = bitangent * (max_v - min_v);

    // Store in area_light
    merged.origin = origin_world;
    merged.e1 = e1;
    merged.e2 = e2;
    merged.normal = avg_normal;
    merged.emission = tris[0].emission;
    merged.area = cross(e1, e2).length();

    return merged;
}

object obj_to_mesh(const char *objpath, const char *mtlpath, vec3 offset, material_type mt, glm::mat4 tmat, color c)
{
    std::vector<triangle> triangle_mesh;
    std::vector<area_light> lights;
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

        bool is_light = (shape.name == "light"); // for area lights

        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3)
                continue;

            point3 v[3];
            vec3 n[3];
            int mat_id = shape.mesh.material_ids[f];
            material mat;
            bool have_vertex_normals = true;

            for (int i = 0; i < 3; i++)
            {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + i];
                float vx = attrib.vertices[3 * idx.vertex_index + 0] + offset[0];
                float vy = attrib.vertices[3 * idx.vertex_index + 1] + offset[1];
                float vz = attrib.vertices[3 * idx.vertex_index + 2] + offset[2];

                glm::vec4 v4(vx, vy, vz, 1.0f); // vertex transformation
                glm::vec4 transformed_v4 = tmat * v4;
                v[i] = point3(transformed_v4.x, transformed_v4.y, transformed_v4.z);

                min_point = point3(std::min(min_point.x(), v[i].x()),
                                   std::min(min_point.y(), v[i].y()),
                                   std::min(min_point.z(), v[i].z()));
                max_point = point3(std::max(max_point.x(), v[i].x()),
                                   std::max(max_point.y(), v[i].y()),
                                   std::max(max_point.z(), v[i].z()));

                if (idx.normal_index >= 0)
                {
                    float nx = attrib.normals[3 * idx.normal_index + 0];
                    float ny = attrib.normals[3 * idx.normal_index + 1];
                    float nz = attrib.normals[3 * idx.normal_index + 2];

                    glm::vec4 n4(nx, ny, nz, 0.0f); // normal transformation
                    glm::vec4 transformed_n4 = glm::transpose(glm::inverse(tmat)) * n4;
                    n[i] = unit_vector(vec3(transformed_n4.x, transformed_n4.y, transformed_n4.z));
                }
                else
                {
                    have_vertex_normals = false;
                }
                if (!have_vertex_normals)
                {
                    vec3 face_normal = unit_vector(cross(v[1] - v[0], v[2] - v[0]));
                    // If you previously negated normals globally, keep the same convention (choose one)
                    // For example, to flip all computed normals to match your previous code:
                    // face_normal = -face_normal;

                    n[0] = face_normal;
                    n[1] = face_normal;
                    n[2] = face_normal;
                }

                if (mat_id >= 0 && mat_id < materials.size())
                {
                    const auto &m = materials[mat_id];
                    mat.ambient = color(m.ambient[0], m.ambient[1], m.ambient[2]);
                    mat.diffuse = color(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
                    mat.specular = color(m.specular[0], m.specular[1], m.specular[2]);
                    mat.spec_exponent = m.shininess;
                }
                else
                {
                    mat.ambient = color(fmax(0.0f, c.x() - 0.1f), fmax(0.0f, c.y() - 0.1f), fmax(0.0f, c.z() - 0.1f));
                    mat.diffuse = color(c.x(), c.y(), c.z());
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
                    mat.reflectivity = 0.05f;
                    mat.transparency = 0.95f;
                    mat.transmission_color = color(0.8, 0.2, 0.2);
                    mat.ior = 1.5f;
                }
                else
                {
                    mat.reflectivity = 0.0f;
                    mat.transparency = 0.0f;
                    mat.ior = 1.0f;
                }

                if (is_light) // area light
                {
                    struct area_light al;
                    al.origin = v[0];
                    al.e1 = v[2] - v[0];
                    al.e2 = v[1] - v[0];
                    al.normal = n[0];

                    lights.push_back(al);
                    mat.emission = color(15, 15, 15);

                    // mt = material_type::light;
                    mat.diffuse = color(0, 0, 0);
                    mat.specular = color(0, 0, 0);
                    mat.reflectivity = 0.0f;
                    mat.transparency = 0.0f;
                }
            }

            triangle_mesh.push_back(triangle(v[0], v[1], v[2], n[0], n[1], n[2], mat, mt, nullptr));

            index_offset += fv;
            counter_triangles++;
        }

        std::cout << "number of triangles: " << counter_triangles << '\n';
        total_triangles += counter_triangles;
    }

    min_point = min_point - point3(AABB_EPS, AABB_EPS, AABB_EPS);
    max_point = max_point + point3(AABB_EPS, AABB_EPS, AABB_EPS);

    std::cout << "Total triangles: " << total_triangles << '\n';

    object obj(triangle_mesh, aabb(min_point, max_point), mt, tmat, lights);

    for (triangle &tri : obj.mesh)
        tri.set_obj(&obj);

    return obj;
}
