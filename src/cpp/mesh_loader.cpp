#include "../hpp/mesh_loader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "..\..\tinyobjloader\tiny_obj_loader.h"

#include <iostream>
#include <cfloat>

object::object() = default;

/* object::object(const std::vector<triangle> &m, const aabb &b, material_preset &mt, const glm::mat4 &tform, const std::vector<area_light> &als) // material_type
    : mesh(m), box(b), mt(mt), transform(tform), lights(als)
{
    inverse_transform = glm::inverse(tform);
    transpose_transform = glm::transpose(inverse_transform);
} */

object::object(std::vector<triangle> mesh, aabb box, glm::mat4 transform, const std::vector<area_light> &lights) : mesh(mesh), box(box), transform(transform), lights(lights)
{
    inverse_transform = glm::inverse(transform);
    transpose_transform = glm::transpose(inverse_transform);
}

std::vector<std::vector<area_light>> cluster_light_triangles(const std::vector<area_light> &tris) // HERE
{
    std::vector<std::vector<area_light>> clusters;

    const float NORMAL_EPS = 0.01f;
    const float POS_EPS = 0.1f; // more forgiving

    for (const auto &t : tris)
    {
        bool added = false;

        for (auto &cluster : clusters)
        {
            const auto &ref = cluster[0];

            bool similar_normal =
                dot(unit_vector(t.normal), unit_vector(ref.normal)) > (1.0f - NORMAL_EPS);

            bool close =
                (t.origin - ref.origin).length() < POS_EPS;

            if (similar_normal && close)
            {
                cluster.push_back(t);
                added = true;
                break;
            }
        }

        if (!added)
            clusters.push_back({t});
    }

    return clusters;
}

/* area_light merge_light_triangles(const std::vector<area_light> &tris) // FIX HERE
{
    area_light merged;
    if (tris.empty())
        return merged;

    // 1. Average normal
    vec3 avg_normal(0, 0, 0);
    for (const auto &t : tris)
        avg_normal += unit_vector(t.normal);
    merged.normal = unit_vector(avg_normal);

    // 2. Collect all points
    std::vector<point3> pts;
    pts.reserve(tris.size() * 3);

    for (const auto &t : tris)
    {
        pts.push_back(t.origin);
        pts.push_back(t.origin + t.e1);
        pts.push_back(t.origin + t.e2);
    }

    point3 ref = pts[0];

    // 3. Tangent frame
    vec3 tangent = unit_vector(perpendicular(merged.normal));
    vec3 bitangent = cross(merged.normal, tangent);

    // 4. Project
    float min_u = FLT_MAX, max_u = -FLT_MAX;
    float min_v = FLT_MAX, max_v = -FLT_MAX;

    for (const auto &p : pts)
    {
        vec3 d = p - ref;
        float u = dot(d, tangent);
        float v = dot(d, bitangent);

        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    // 5. Reconstruct rectangle
    merged.origin = ref + tangent * min_u + bitangent * min_v;
    merged.e1 = tangent * (max_u - min_u);
    merged.e2 = bitangent * (max_v - min_v);
    merged.area = cross(merged.e1, merged.e2).length();

    // 6. Emission
    merged.emission = tris[0].emission;

    // 7. Copy triangles
    for (const auto &t : tris)
        merged.tris.push_back(t.tris[0]); // each area_light contains exactly 1 triangle

    return merged;
} */

area_light merge_light_triangles(const std::vector<area_light> &tris) // TODO multiple lights (only merge triangles actually connected)
{
    area_light merged;
    if (tris.empty())
        return merged;

    // --- 1. Compute average normal
    vec3 avg_normal(0, 0, 0);
    for (const auto &t : tris)
        avg_normal += unit_vector(t.normal);
    avg_normal = unit_vector(avg_normal);

    // --- 2. Collect all world points from the triangles
    std::vector<point3> all_points;
    all_points.reserve(tris.size() * 3);

    for (const auto &t : tris)
    {
        all_points.push_back(t.origin);
        all_points.push_back(t.origin + t.e1);
        all_points.push_back(t.origin + t.e2);
    }

    // Pick a reference point
    point3 ref = all_points[0];

    // --- 3. Build local tangent frame for projection
    vec3 tangent = unit_vector(perpendicular(avg_normal));
    vec3 bitangent = cross(avg_normal, tangent);

    // Special-case lights parallel to the XZ plane
    if (fabs(avg_normal.y()) > 0.999f)
    {
        tangent = vec3(1, 0, 0);
        bitangent = vec3(0, 0, -1); // matches Cornell box orientation
    }
    else
    {
        tangent = unit_vector(perpendicular(avg_normal));
        bitangent = cross(avg_normal, tangent);
    }

    // --- 4. Project relative to reference point!
    float min_u = FLT_MAX, max_u = -FLT_MAX;
    float min_v = FLT_MAX, max_v = -FLT_MAX;

    for (const auto &p : all_points)
    {
        vec3 d = p - ref;
        float u = dot(d, tangent);
        float v = dot(d, bitangent);

        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    // --- 5. Reconstruct the world rectangle
    point3 origin_world = ref + tangent * min_u + bitangent * min_v;
    vec3 e1 = tangent * (max_u - min_u);
    vec3 e2 = bitangent * (max_v - min_v);

    // --- 6. Fill result
    merged.origin = origin_world;
    merged.e1 = e1;
    merged.e2 = e2;
    merged.normal = avg_normal;
    merged.emission = tris[0].emission;
    merged.area = cross(e1, e2).length();

    return merged;
}

object obj_to_mesh(const char *objpath, const char *mtlpath,
                   vec3 offset, material_preset preset, material_type mt,
                   glm::mat4 tmat, color override_color)
{
    std::vector<triangle> triangle_mesh;
    std::vector<area_light> lights;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                objpath, mtlpath);

    if (!warn.empty())
        std::cout << "warning: " << warn << "\n";
    if (!err.empty())
        std::cerr << "error: " << err << "\n";
    if (!ret)
        exit(1);

    point3 min_point(FLT_MAX, FLT_MAX, FLT_MAX);
    point3 max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    const float AABB_EPS = 1e-4f;

    int total_triangles = 0;
    int light_ids = 0;

    for (const auto &shape : shapes)
    {
        bool is_light = (shape.name == "light" || shape.name == "light2");

        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3)
            {
                index_offset += fv;
                continue;
            }

            // -----------------------------
            // 1. Load material ONCE per face
            // -----------------------------
            int mat_id = shape.mesh.material_ids[f];
            material mat = make_material(preset, override_color);

            if (mat_id >= 0 && mat_id < materials.size())
            {
                const auto &m = materials[mat_id];

                mat.diffuse = color(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
                mat.specular = color(m.specular[0], m.specular[1], m.specular[2]);
                mat.spec_exponent = m.shininess;

                if (materials[mat_id].name == "mirror")
                {
                    mat.is_delta_specular = true;
                    mat.specular_prob = 1.0f;
                }
            }

            if (is_light)
            {
                mat.emission = color(25, 25, 25); // 25
                mat.diffuse = color(0, 0, 0);
                mat.specular = color(0, 0, 0);
                mat.light_id = light_ids++;
            }

            // -----------------------------
            // 2. Load vertices and normals
            // -----------------------------
            point3 v[3];
            vec3 n[3];
            bool have_normals = true;

            for (int i = 0; i < 3; i++)
            {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + i];

                // Vertex
                float vx = attrib.vertices[3 * idx.vertex_index + 0] + offset.x();
                float vy = attrib.vertices[3 * idx.vertex_index + 1] + offset.y();
                float vz = attrib.vertices[3 * idx.vertex_index + 2] + offset.z();

                glm::vec4 v4(vx, vy, vz, 1.0f);
                glm::vec4 tv = tmat * v4;
                v[i] = point3(tv.x, tv.y, tv.z);

                // Track AABB
                min_point = point3(std::min(min_point.x(), v[i].x()),
                                   std::min(min_point.y(), v[i].y()),
                                   std::min(min_point.z(), v[i].z()));
                max_point = point3(std::max(max_point.x(), v[i].x()),
                                   std::max(max_point.y(), v[i].y()),
                                   std::max(max_point.z(), v[i].z()));

                // Normal
                if (idx.normal_index >= 0)
                {
                    float nx = attrib.normals[3 * idx.normal_index + 0];
                    float ny = attrib.normals[3 * idx.normal_index + 1];
                    float nz = attrib.normals[3 * idx.normal_index + 2];

                    glm::vec4 n4(nx, ny, nz, 0.0f);
                    glm::vec4 tn = glm::transpose(glm::inverse(tmat)) * n4;
                    n[i] = unit_vector(vec3(tn.x, tn.y, tn.z));
                }
                else
                {
                    have_normals = false;
                }
            }

            // -----------------------------
            // 3. Compute face normal if needed
            // -----------------------------
            if (!have_normals)
            {
                vec3 face_normal = unit_vector(cross(v[1] - v[0], v[2] - v[0]));
                n[0] = n[1] = n[2] = face_normal;
            }

            // -----------------------------
            // 4. Register area light
            // -----------------------------
            if (is_light)
            {
                area_light al;
                al.origin = v[0];
                al.e1 = v[1] - v[0];
                al.e2 = v[2] - v[0];
                al.normal = n[0];
                al.emission = mat.emission;

                // Ensure normal faces downward (Cornell box)
                if (dot(al.normal, vec3(0, -1, 0)) < 0)
                    al.normal = -al.normal;

                lights.push_back(al);
            }

            // -----------------------------
            // 5. Add triangle to mesh
            // -----------------------------
            triangle_mesh.emplace_back(
                v[0], v[1], v[2],
                n[0], n[1], n[2],
                mat, mt, nullptr);

            index_offset += fv;
            total_triangles++;
        }
    }

    min_point = min_point - point3(AABB_EPS, AABB_EPS, AABB_EPS);
    max_point = max_point + point3(AABB_EPS, AABB_EPS, AABB_EPS);

    std::cout << "Total triangles: " << total_triangles << "\n";

    object obj(triangle_mesh, aabb(min_point, max_point), tmat, lights);

    for (triangle &tri : obj.mesh)
        tri.set_obj(&obj);

    return obj;
}
