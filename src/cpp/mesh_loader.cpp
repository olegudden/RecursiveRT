#include "../hpp/mesh_loader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "..\..\external\tinyobjloader\tiny_obj_loader.h"

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
                   vec3 offset,
                   glm::mat4 tmat, color override_color, TextureManager &tex_manager)
{
    std::vector<triangle> triangle_mesh;
    std::vector<area_light> lights;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objpath, mtlpath);

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

    // -------------------------------------------------------------
    // Track separate light instances per material based on proximity
    // -------------------------------------------------------------
    std::map<int, int> mat_id_to_active_light_index;
    std::map<int, point3> mat_id_last_centroid;
    const float LIGHT_DISTANCE_THRESHOLD = 5.0f;

    for (const auto &shape : shapes)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3)
            {
                index_offset += fv;
                continue;
            }

            int mat_id = shape.mesh.material_ids[f];
            material mat{};

            bool is_face_emissive = false;
            color face_emission(0, 0, 0);
            int current_light_index = -1;

            // -------------------------------------------------------------
            // 1. Parse Vertices, Normals, and UVs First
            // -------------------------------------------------------------
            point3 v[3];
            vec3 n[3];
            vec3 uv[3];
            bool have_normals = true;

            for (int i = 0; i < 3; i++)
            {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + i];
                float vx = attrib.vertices[3 * idx.vertex_index + 0] + offset.x();
                float vy = attrib.vertices[3 * idx.vertex_index + 1] + offset.y();
                float vz = attrib.vertices[3 * idx.vertex_index + 2] + offset.z();

                glm::vec4 tv = tmat * glm::vec4(vx, vy, vz, 1.0f);
                v[i] = point3(tv.x, tv.y, tv.z);

                min_point = point3(std::min(min_point.x(), v[i].x()), std::min(min_point.y(), v[i].y()), std::min(min_point.z(), v[i].z()));
                max_point = point3(std::max(max_point.x(), v[i].x()), std::max(max_point.y(), v[i].y()), std::max(max_point.z(), v[i].z()));

                if (idx.normal_index >= 0)
                {
                    float nx = attrib.normals[3 * idx.normal_index + 0];
                    float ny = attrib.normals[3 * idx.normal_index + 1];
                    float nz = attrib.normals[3 * idx.normal_index + 2];

                    glm::vec4 tn = glm::transpose(glm::inverse(tmat)) * glm::vec4(nx, ny, nz, 0.0f);
                    n[i] = unit_vector(vec3(tn.x, tn.y, tn.z));
                }
                else
                {
                    have_normals = false;
                }

                if (idx.texcoord_index >= 0)
                {
                    uv[i] = vec3(attrib.texcoords[2 * idx.texcoord_index + 0], attrib.texcoords[2 * idx.texcoord_index + 1], 0);
                }
                else
                {
                    uv[i] = vec3(0.5f, 0.5f, 0);
                }
            }

            vec3 cross_product = cross(v[1] - v[0], v[2] - v[0]);
            if (!have_normals)
            {
                vec3 face_normal = unit_vector(cross_product);
                n[0] = n[1] = n[2] = face_normal;
            }

            float tri_area = 0.5f * cross_product.length();
            point3 current_face_centroid = (v[0] + v[1] + v[2]) / 3.0f;

            // -------------------------------------------------------------
            // 2. Parse Material & Textures
            // -------------------------------------------------------------
            if (mat_id >= 0 && mat_id < static_cast<int>(materials.size()))
            {
                const auto &m = materials[mat_id];

                mat.ambient = color(m.ambient[0], m.ambient[1], m.ambient[2]);
                mat.diffuse = color(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
                mat.specular = color(m.specular[0], m.specular[1], m.specular[2]);
                mat.spec_exponent = m.shininess;
                mat.roughness = m.roughness;
                mat.metallic = m.metallic;
                mat.clearcoat_roughness = m.clearcoat_roughness;
                mat.clearcoat_thickness = m.clearcoat_thickness;
                mat.sheen = m.sheen;

                if (shape.name == "monkey" || shape.name == "pyramid")
                {
                    mat.subsurface_weight = 0.5f;
                    mat.subsurface_scale = 0.25f;
                }

                // Diffuse Texture & Missing Kd Fallback
                if (!m.diffuse_texname.empty())
                {
                    mat.diffuse_texname = m.diffuse_texname;
                    mat.has_diffuse_tex = true;
                    std::string full_path = "./objects/" + m.diffuse_texname;
                    mat.diffuse_tex_id = tex_manager.load_texture(full_path);

                    if (mat.diffuse_tex_id == -1)
                    {
                        mat.has_diffuse_tex = false;
                    }
                    else if (mat.diffuse.x() == 0.0f && mat.diffuse.y() == 0.0f && mat.diffuse.z() == 0.0f)
                    {
                        mat.diffuse = color(1.0f, 1.0f, 1.0f);
                    }
                }

                // Emissive Texture (map_Ke)
                if (!m.emissive_texname.empty())
                {
                    mat.emissive_texname = m.emissive_texname;
                    mat.has_emissive_tex = true;
                    std::string full_path = "./objects/" + m.emissive_texname;
                    mat.emissive_tex_id = tex_manager.load_texture(full_path);

                    if (mat.emissive_tex_id == -1)
                    {
                        std::cerr << "[ERROR] Failed to load emissive texture: " << full_path << "\n";
                        mat.has_emissive_tex = false;
                    }
                }

                // Prevent pure-white Ks 1.0 1.0 1.0 washout on standard diffuse materials
                if (m.illum == 2 && !mat.is_delta_specular)
                {
                    mat.specular = color(0.04f, 0.04f, 0.04f);
                }

                // Mirror / Perfect Specular Detection
                if (m.illum == 5)
                {
                    mat.is_delta_specular = true;
                    mat.specular_prob = 1.0f;
                    mat.specular = color(m.specular[0], m.specular[1], m.specular[2]);
                    mat.roughness = 0.0f;
                }

                // Glass / Transparent Dielectric Detection
                bool is_transparent = (m.illum == 7);
                if (is_transparent)
                {
                    mat.transparency = 1.0f;
                    mat.ior = (m.ior > 0.0f) ? m.ior : 1.5f;
                    mat.specular = color(m.specular[0], m.specular[1], m.specular[2]);
                    mat.roughness = 0.0f;
                }

                // -------------------------------------------------------------
                // 3. Emissive Light Instance Grouping & Mask Filtering
                // -------------------------------------------------------------
                if (m.emission[0] > 0.001f || m.emission[1] > 0.001f || m.emission[2] > 0.001f)
                {
                    face_emission = color(m.emission[0], m.emission[1], m.emission[2]);
                    mat.emission = face_emission;

                    bool triangle_actually_emits = true;
                    if (mat.has_emissive_tex)
                    {
                        vec3 uv_center = (uv[0] + uv[1] + uv[2]) / 3.0f;
                        color c_mid = tex_manager.sample_texture_from_cache(tex_manager, mat.emissive_tex_id, uv_center.x(), uv_center.y());
                        color c0 = tex_manager.sample_texture_from_cache(tex_manager, mat.emissive_tex_id, uv[0].x(), uv[0].y());
                        color c1 = tex_manager.sample_texture_from_cache(tex_manager, mat.emissive_tex_id, uv[1].x(), uv[1].y());
                        color c2 = tex_manager.sample_texture_from_cache(tex_manager, mat.emissive_tex_id, uv[2].x(), uv[2].y());

                        float max_val = std::max({c_mid.x(), c_mid.y(), c_mid.z(),
                                                  c0.x(), c0.y(), c0.z(),
                                                  c1.x(), c1.y(), c1.z(),
                                                  c2.x(), c2.y(), c2.z()});

                        if (max_val < 0.01f)
                        {
                            triangle_actually_emits = false;
                        }
                    }

                    if (triangle_actually_emits)
                    {
                        is_face_emissive = true;
                        bool create_new_light_instance = false;

                        if (mat_id_to_active_light_index.find(mat_id) == mat_id_to_active_light_index.end())
                        {
                            create_new_light_instance = true;
                        }
                        else
                        {
                            point3 last_centroid = mat_id_last_centroid[mat_id];
                            float distance = (current_face_centroid - last_centroid).length();
                            if (distance > LIGHT_DISTANCE_THRESHOLD)
                            {
                                create_new_light_instance = true;
                            }
                        }

                        if (create_new_light_instance)
                        {
                            area_light new_light;
                            new_light.light_id = static_cast<int>(lights.size());
                            new_light.emission = face_emission;
                            new_light.name = m.name + "_instance_" + std::to_string(new_light.light_id);

                            mat_id_to_active_light_index[mat_id] = new_light.light_id;
                            lights.push_back(new_light);

                            std::cout << "[SUCCESS] Created Unique Light Instance: " << new_light.name << "\n";
                        }

                        current_light_index = mat_id_to_active_light_index[mat_id];
                        mat_id_last_centroid[mat_id] = current_face_centroid;
                        mat.light_id = current_light_index;
                    }
                }
            }

            triangle tri(v[0], v[1], v[2], n[0], n[1], n[2], uv[0], uv[1], uv[2], mat, nullptr, tri_area);

            if (is_face_emissive && current_light_index >= 0)
            {
                lights[current_light_index].tris.push_back(tri);
            }

            triangle_mesh.push_back(tri);
            index_offset += fv;
            total_triangles++;
        }
    }

    // -------------------------------------------------------------
    // 4. Post-processing: Finalize metadata uniquely per isolated light instance
    // -------------------------------------------------------------
    for (size_t i = 0; i < lights.size(); i++)
    {
        if (!lights[i].tris.empty())
        {
            float running_area_sum = 0.0f;
            vec3 accumulated_normal(0, 0, 0);
            point3 accumulated_center(0, 0, 0);

            for (auto &tri : lights[i].tris)
            {
                running_area_sum += tri.area;
                lights[i].tris_cdf.push_back(running_area_sum);
                accumulated_normal += tri.n0 * tri.area;
                accumulated_center += (tri.v0 + tri.v1 + tri.v2) / 3.0f;
            }

            lights[i].area = running_area_sum;
            lights[i].normal = unit_vector(accumulated_normal);
            lights[i].origin = accumulated_center / static_cast<float>(lights[i].tris.size());
        }
    }

    min_point = min_point - point3(AABB_EPS, AABB_EPS, AABB_EPS);
    max_point = max_point + point3(AABB_EPS, AABB_EPS, AABB_EPS);

    std::cout << "Total triangles: " << total_triangles << "\n";
    std::cout << "Total distinct lights: " << lights.size() << "\n";

    object obj(triangle_mesh, aabb(min_point, max_point), tmat, lights);

    for (triangle &tri : obj.mesh)
        tri.obj_ptr = &obj;

    return obj;
}
