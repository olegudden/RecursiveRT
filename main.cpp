#include "hpp/vec3.hpp" // use glm and homogeneous coordinates?
#include "hpp/triangle.hpp"
#include "hpp/camera.hpp"
#include "hpp/mesh_loader.hpp"
#include "hpp/light_source.hpp"
#include "hpp/material.hpp"
#include "hpp/bvh.hpp"
#include "hpp/ray.hpp"

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"

#include <vector>
#include <chrono>

std::vector<object> world;

// int ray:: = 0;
// int64_t triangle::triangleTests = 0;

int main()
{
    // runtime check
    auto start = std::chrono::system_clock::now();

    glm::mat4 tformLight = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.3f, 0.0f));
    tformLight = glm::scale(tformLight, glm::vec3(0.01f, 0.01f, 0.01f));

    glm::mat4 tformSphere = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -0.25f));
    tformSphere = glm::scale(tformSphere, glm::vec3(0.15f, 0.15f, 0.15f));

    glm::mat4 tformCube = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.75f));
    tformCube = glm::scale(tformCube, glm::vec3(0.25f, 0.25f, 0.25f));

    tformCube = glm::rotate(tformCube, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 tformCornell = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, 0.8f));
    tformCornell = glm::rotate(tformCornell, glm::radians(-20.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 tform = glm::mat4(1.0f); // identity matrix

    // add OBJs to world
    // world.push_back(obj_to_mesh("./objects/CornellBox-Empty-RG-BigLight.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformCornell, color(0.0, 0.0, 0.0)));
    world.push_back(obj_to_mesh("./objects/CornellBox-Empty-RG.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformCornell, color(0.0, 0.0, 0.0)));

    // world.push_back(obj_to_mesh("./objects/testSphere.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformLight, color(1, 1, 1)));
    world.push_back(obj_to_mesh("./objects/cube.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformCube, color(1, 1, 1)));
    // world.push_back(obj_to_mesh("./objects/suzanne.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformSphere, color(0, 0, 1)));

    // BVH construction
    std::vector<triangle> all_triangles;
    std::vector<area_light> area_lights;

    for (object obj : world)
    {
        all_triangles.insert(all_triangles.end(), obj.mesh.begin(), obj.mesh.end());
        // merge light triangles
        if (!obj.lights.empty())
        {
            area_light merged = merge_light_triangles(obj.lights);
            area_lights.push_back(merged);
        }
    }
    bvh bvh_tree;

    auto build_start = std::chrono::system_clock::now(); // bvh build runtime check

    bvh_tree.root = bvh_tree.build_bvh_binned_sah(all_triangles, 0); // choose BVH contruction strategy

    auto build_end = std::chrono::system_clock::now();
    bvh_tree.stats.build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();

    bvh_tree.stats.collect_bvh_stats(bvh_tree.root.get(), 0, bvh_tree.stats);

    // setup point light sources
    /* std::vector<light_source> lights;
    light_source light1(point3(0, 1.3, 0), color(1, 1, 1), 1.0f);
    lights.push_back(light1);

    light_source light2(point3(0, 1.5, 2), color(1, 1, 1), 1.0f);
    lights.push_back(light2); */

    // setup area light source
    /* area_light al;
    al.origin = point3(-0.24, 1.98, 0.16);
    al.e1 = vec3(0.47, 0, 0);  // (0.23 - -0.24) in X
    al.e2 = vec3(0, 0, -0.38); // (-0.22 - 0.16) in Z
    al.normal = vec3(0, -1, 0);
    al.emission = color(25, 25, 25); // strong light (adjust) */

    // setup camera
    camera cam;
    cam.adjustable = true; // choose between basic or adjustable camera
    // cam.aspect_ratio = 16.0 / 9.0; // ratio width-height
    cam.aspect_ratio = 1.0;
    cam.image_width = 400; // width of image in pixels

    // Rendering
    auto render_start = std::chrono::system_clock::now(); // bvh traversal runtime check

    aabb::total_tests = 0;

    cam.old_render(world, area_lights, bvh_tree);

    std::cout << "Total AABB tests: " << aabb::total_tests << "\n";

    auto render_end = std::chrono::system_clock::now();
    bvh_tree.stats.traverse_time_ms = std::chrono::duration<double, std::milli>(render_end - render_start).count();
    bvh_tree.stats.printStats();
    // std::cout << "Primary rays: " << cam.primary_ray_counter << '\n';
    //  std::cout << "Total rays: " << ray::ray_counter << '\n';
    // std::cout << "Total triangle tests: " << triangle::triangleTests << '\n';

    // runtime check
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total runtime (in ms): " << elapsed << '\n';

    // PPM Viewer
    int working = system(".\\ppm-viewer\\ppm.exe image.ppm");
    if (working != 0)
    {
        std::cerr << "Failed to launch PPM viewer.\n";
    }
    return 0;
}