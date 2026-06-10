#include "../hpp/mesh_loader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "..\..\tinyobjloader\tiny_obj_loader.h"

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

/* object obj_to_mesh(const char *objpath, const char *mtlpath,
                   vec3 offset, material_preset preset, material_type mt,
                   glm::mat4 tmat, color override_color, TextureManager &tex_manager)

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
    // average normal (light)
    vec3 avg_normal(0, 0, 0);

    for (const auto &shape : shapes)
    {
        // -------------------------------------------------------------
        // Robust Light Identification: Check First Face Material's Emission
        // -------------------------------------------------------------
        bool is_light = false;
        color shape_emission(0, 0, 0);

        // If the shape has faces, peek at the material of the very first face (face 0)
        if (!shape.mesh.material_ids.empty())
        {
            int first_face_mat_id = shape.mesh.material_ids[0];
            if (first_face_mat_id >= 0 && first_face_mat_id < materials.size())
            {
                const auto &m = materials[first_face_mat_id];

                // If Ke is non-zero, this shape acts as a light source!
                if (m.emission[0] > 0.0f || m.emission[1] > 0.0f || m.emission[2] > 0.0f)
                {
                    is_light = true;
                    shape_emission = color(m.emission[0], m.emission[1], m.emission[2]);
                }
            }
        }

        // Fallback: If material isn't emissive but name contains "light" or "staklo" (optional safety)
        if (!is_light)
        {
            is_light = (shape.name.rfind("light", 0) == 0) || (shape.name.rfind("staklo", 0) == 0);
        }

        if (is_light)
            std::cout << "light found: " << shape.name << " with emission: " << shape_emission << std::endl;

        // initialize light
        area_light current_light;
        int current_light_index = (int)lights.size();
        float totalArea = 0.0f;
        if (is_light)
        {
            current_light.name = shape.name;
            current_light.light_id = current_light_index;
            current_light.emission = shape_emission; // Pre-populate the light's emission color
        }

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
            material mat{};
            // mat = make_material(preset, override_color);

            if (mat_id >= 0 && mat_id < materials.size())

            {
                const auto &m = materials[mat_id];

                mat.ambient = color(m.ambient[0], m.ambient[1], m.ambient[2]);
                mat.diffuse = color(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
                mat.specular = color(m.specular[0], m.specular[1], m.specular[2]);
                mat.spec_exponent = m.shininess;

                if (m.emission[0] > 0 || m.emission[1] > 0 || m.emission[2] > 0)
                {
                    current_light.emission[0] = m.emission[0]; // 25
                    current_light.emission[1] = m.emission[1];
                    current_light.emission[2] = m.emission[2];
                }

                if (m.illum == 4 || m.illum == 6)
                {
                    mat.transparency = 1.0f;
                    mat.ior = 1.5;
                }

                // Check if the diffuse texture string is NOT empty
                if (!m.diffuse_texname.empty())
                {
                    // 1. Store the filename
                    mat.diffuse_texname = m.diffuse_texname;
                    // 2. Set a flag so the shader knows to use UVs
                    mat.has_diffuse_tex = true;
                    // 3. Combine the directory path with the parsed filename
                    std::string full_path = "./objects/" + m.diffuse_texname;
                    // Load and store the integer index inside the material
                    mat.diffuse_tex_id = tex_manager.load_texture(full_path);

                    if (mat.diffuse_tex_id == -1)
                    {
                        mat.has_diffuse_tex = false; // Fallback to flat color if image file missing
                    }
                    // std::cout << "Found texture: " << m.diffuse_texname << std::endl;
                }

                if (materials[mat_id].name == "mirror")
                {
                    mat.is_delta_specular = true;
                    mat.specular_prob = 1.0f;
                }
            }

            // -----------------------------
            // 2. Load vertices and normals
            // -----------------------------
            point3 v[3];
            vec3 n[3];
            vec3 uv[3];
            bool have_normals = true;

            for (int i = 0; i < 3; i++)
            {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + i];

                // Vertex
                float vx = attrib.vertices[3 * idx.vertex_index + 0] + offset.x();
                float vy = attrib.vertices[3 * idx.vertex_index + 1] + offset.y();
                float vz = attrib.vertices[3 * idx.vertex_index + 2] + offset.z();
                glm::vec4 v4(vx, vy, vz, 1.0f);

                // transformation
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

                    // transformation
                    glm::vec4 tn = glm::transpose(glm::inverse(tmat)) * n4;
                    n[i] = unit_vector(vec3(tn.x, tn.y, tn.z));
                }

                else
                {
                    have_normals = false;
                }

                // textures (uv)
                if (idx.texcoord_index >= 0)
                {
                    uv[i] = vec3(
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        attrib.texcoords[2 * idx.texcoord_index + 1],
                        0);
                }
                else
                {
                    uv[i] = vec3(0.5f, 0.5f, 0);
                }
            }

            // -----------------------------
            // 3. Compute face normal if needed (fallback)
            // -----------------------------
            // cross product
            vec3 cross_product = cross(v[1] - v[0], v[2] - v[0]);
            if (!have_normals)
            {
                vec3 face_normal = unit_vector(cross_product);
                n[0] = n[1] = n[2] = face_normal;
            }

            if (is_light)
            {
                mat.emission = current_light.emission;
                mat.light_id = current_light_index;
            }

            // create triangle
            triangle tri(v[0], v[1], v[2], n[0], n[1], n[2], uv[0], uv[1], uv[2], mat, mt, nullptr);

            // -----------------------------
            // 4. Connect triangle to light
            // -----------------------------
            if (is_light)
            {
                // triangle area formula
                float triangle_area = 0.5f * cross_product.length();
                avg_normal += n[0] * triangle_area;
                totalArea += triangle_area;
                current_light.tris.push_back(tri);
            }

            // -----------------------------
            // 5. Add triangle to mesh
            // -----------------------------
            triangle_mesh.push_back(tri);
            index_offset += fv;
            total_triangles++;
        }

        // save light
        if (is_light && !current_light.tris.empty())
        {
            current_light.area = totalArea;
            current_light.normal = unit_vector(avg_normal);
            current_light.origin = current_light.tris[0].get_v0(); // TODO center
            lights.push_back(current_light);
        }
    }

    min_point = min_point - point3(AABB_EPS, AABB_EPS, AABB_EPS);
    max_point = max_point + point3(AABB_EPS, AABB_EPS, AABB_EPS);

    std::cout << "Total triangles: " << total_triangles << "\n";
    std::cout << "Total distinct lights: " << lights.size() << "\n";

    object obj(triangle_mesh, aabb(min_point, max_point), tmat, lights);

    for (triangle &tri : obj.mesh)
        tri.set_obj(&obj);

    return obj;
} */

object obj_to_mesh(const char *objpath, const char *mtlpath,
                   vec3 offset, material_preset preset, material_type mt,
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
    // FIX STRATEGY: Track separate light instances per material based on proximity
    // -------------------------------------------------------------
    std::map<int, int> mat_id_to_active_light_index;
    std::map<int, point3> mat_id_last_centroid;   // Tracks the geometric center of the last face
    const float WINDOW_DISTANCE_THRESHOLD = 5.0f; // If faces are further than 5 units apart, it's a new window pane

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

            if (mat_id >= 0 && mat_id < materials.size())
            {
                const auto &m = materials[mat_id];

                mat.ambient = color(m.ambient[0], m.ambient[1], m.ambient[2]);
                mat.diffuse = color(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
                mat.specular = color(m.specular[0], m.specular[1], m.specular[2]);
                mat.spec_exponent = m.shininess;

                if (m.emission[0] > 0.001f || m.emission[1] > 0.001f || m.emission[2] > 0.001f)
                {
                    is_face_emissive = true;
                    face_emission = color(m.emission[0], m.emission[1], m.emission[2]);

                    // We need to parse vertices temporarily here to check distance proximity
                    tinyobj::index_t idx0 = shape.mesh.indices[index_offset + 0];
                    tinyobj::index_t idx1 = shape.mesh.indices[index_offset + 1];
                    tinyobj::index_t idx2 = shape.mesh.indices[index_offset + 2];

                    // Approximate face centroid in world space
                    point3 p0(attrib.vertices[3 * idx0.vertex_index + 0] + offset.x(), attrib.vertices[3 * idx0.vertex_index + 1] + offset.y(), attrib.vertices[3 * idx0.vertex_index + 2] + offset.z());
                    point3 p1(attrib.vertices[3 * idx1.vertex_index + 0] + offset.x(), attrib.vertices[3 * idx1.vertex_index + 1] + offset.y(), attrib.vertices[3 * idx1.vertex_index + 2] + offset.z());
                    point3 p2(attrib.vertices[3 * idx2.vertex_index + 0] + offset.x(), attrib.vertices[3 * idx2.vertex_index + 1] + offset.y(), attrib.vertices[3 * idx2.vertex_index + 2] + offset.z());
                    point3 current_face_centroid = (p0 + p1 + p2) / 3.0f;

                    bool create_new_light_instance = false;

                    // If this is the first time seeing this material, definitely make a light
                    if (mat_id_to_active_light_index.find(mat_id) == mat_id_to_active_light_index.end())
                    {
                        create_new_light_instance = true;
                    }
                    else
                    {
                        // Check how far this face is from the last face of the same material
                        point3 last_centroid = mat_id_last_centroid[mat_id];
                        float distance = (current_face_centroid - last_centroid).length();

                        // If the distance is significant, we hopped to a completely distinct window pane!
                        if (distance > WINDOW_DISTANCE_THRESHOLD)
                        {
                            create_new_light_instance = true;
                        }
                    }

                    if (create_new_light_instance)
                    {
                        area_light new_light;
                        new_light.light_id = (int)lights.size();
                        new_light.emission = face_emission;

                        // Create a descriptive name showing instance mapping count
                        new_light.name = m.name + "_instance_" + std::to_string(new_light.light_id);

                        mat_id_to_active_light_index[mat_id] = new_light.light_id;
                        lights.push_back(new_light);

                        std::cout << "[SUCCESS] Created Unique Window Instance: " << new_light.name << "\n";
                    }

                    current_light_index = mat_id_to_active_light_index[mat_id];
                    mat_id_last_centroid[mat_id] = current_face_centroid; // Update reference coordinates

                    mat.emission = face_emission;
                    mat.light_id = current_light_index;
                }

                // (Keep your existing textures & mirror assignments here)
                if (!m.diffuse_texname.empty())
                {
                    mat.diffuse_texname = m.diffuse_texname;
                    mat.has_diffuse_tex = true;
                    std::string full_path = "./objects/" + m.diffuse_texname;
                    mat.diffuse_tex_id = tex_manager.load_texture(full_path);
                    if (mat.diffuse_tex_id == -1)
                        mat.has_diffuse_tex = false;
                }

                // MATERIALS (TODO: Subsurface Scattering, Fresnel Effect)
                //  ----------------------------------------------------------------
                //  1) MIRROR / PERFECT SPECULAR DETECTION
                //  ----------------------------------------------------------------
                if (m.illum == 5 || (m.name == "mirror") || (m.specular[0] > 0.99f && m.shininess > 500.0f))
                {
                    mat.is_delta_specular = true;
                    mat.specular_prob = 1.0f; // diffuse_prob = 0
                                              // Grab the mirror color tint (usually white: 0.95, 0.95, 0.95)
                    mat.specular[0] = m.specular[0], mat.specular[1] = m.specular[1], mat.specular[2] = m.specular[2];
                    mat.roughness = 0.0f; // Perfect reflection
                }

                // ----------------------------------------------------------------
                // 2) GLASS / TRANSPARENT DIELECTRIC DETECTION
                // ----------------------------------------------------------------
                bool is_transparent = (m.illum == 7) || (m.name == "transparent") || m.transmittance[0] > 0.0f;
                if (is_transparent)
                {
                    // mat.is_dielectric = true; // Mark it internally as a dielectric
                    mat.transparency = 1.0f;
                    // Fallback safely if IOR/Ni was missing or set to 0 in the file
                    mat.ior = (m.ior > 0.0f) ? m.ior : 1.5f;
                    // Grab the transmission color (Tf). If Tf is not provided,
                    // fall back to the diffuse color (Kd) as the glass color.
                    /* if (m.transmittance.length() > 0.0f)
                    {
                        mat.transmission_color = m.transmittance;
                    }
                    else
                    {
                        mat.transmission_color = m.diffuse;
                    } */
                    // Glass has a split personality: it reflects OR refracts depending on Schlick's Fresnel approximation.
                    mat.specular[0] = m.specular[0], mat.specular[1] = m.specular[1], mat.specular[2] = m.specular[2];
                    mat.roughness = 0.0f;
                }
            }

            // -------------------------------------------------------------
            // Vertex, Normal, and UV Parsing (Your exact code blocks remain identical)
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

            triangle tri(v[0], v[1], v[2], n[0], n[1], n[2], uv[0], uv[1], uv[2], mat, mt, nullptr, tri_area);

            if (is_face_emissive)
            {
                // Push directly to our active distinct light instance structure
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
                lights[i].tris_cdf.push_back(running_area_sum); // Raw un-normalized sum
                accumulated_normal += tri.n0 * tri.area;
                accumulated_center += (tri.v0 + tri.v1 + tri.v2) / 3.0f;
            }

            lights[i].area = running_area_sum; // total
            lights[i].normal = unit_vector(accumulated_normal);

            // Calculate the true geometric center of this specific window instance!
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
