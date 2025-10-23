#include "vec3.h" // use glm and homogeneous coordinates?
#include "triangle.h"
#include "camera.h"
#include "mesh_loader.h"
#include "light_source.h"
#include "material.h"
#include "bvh.h"
#include "ray.h"

#include "..\glm\glm.hpp"
#include "..\glm\gtc\matrix_transform.hpp"

#include <vector>
#include <chrono>

std::vector<object> world;
int ray::ray_counter = 0;
int64_t triangle::triangleTests = 0;

int main()
{
    // runtime check
    auto start = std::chrono::system_clock::now();

    glm::mat4 tformDragon = glm::translate(glm::mat4(1.0f), glm::vec3(1.5f, 0.85f, 2.0f)); // rotate
    tformDragon = glm::rotate(tformDragon, glm::radians(80.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 tformBistro = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, -4.0f, 2.0f));
    tformBistro = glm::rotate(tformBistro, glm::radians(50.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 tformSponza = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 1.5f));
    tformSponza = glm::rotate(tformSponza, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 tformMori = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.75f));
    tformMori = glm::rotate(tformMori, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 tformBike = glm::translate(glm::mat4(1.0f), glm::vec3(-0.25f, 0.25f, 0.0f)); // z=1
    tformBike = glm::rotate(tformBike, glm::radians(65.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    /* glm::mat4 tform3 = glm::translate(glm::mat4(1.0f), glm::vec3(6.0f, -5.0f, 0.0f)); // rotate + scale
    tform3 = glm::rotate(tform3, glm::radians(45.0f), glm::vec3(1.0f, 0.0f, 1.0f));
    tform3 = glm::scale(tform3, glm::vec3(1.0f, 0.5f, 2.0f)); */
    glm::mat4 tform = glm::mat4(1.0f); // identity matrix

    // add OBJs to world
    // world.push_back(obj_to_mesh("./objects/sponza.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformSponza, color(0.2, 0.3, 0.8)));
    // world.push_back(obj_to_mesh("./objects/bistroWine.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformBistro, color(0.2, 0.3, 0.8)));
    // world.push_back(obj_to_mesh("./objects/bunny.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformBistro, color(0.2, 0.3, 0.8)));
    // world.push_back(obj_to_mesh("./objects/moriKnob.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformMori, color(0.2, 0.3, 0.8)));
    // world.push_back(obj_to_mesh("./objects/dragon.obj", "./objects/", vec3(0, 0, -2), material_type::diffuse, tformDragon, color(0.2, 0.3, 0.8)));
    // world.push_back(obj_to_mesh("./objects/roadbike.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformBike, color(0.2, 0.3, 0.8)));

    world.push_back(obj_to_mesh("../objects/compareIco8.obj", "../objects/", vec3(0, 0.75, 0), material_type::transparent, tform, color(0.2, 0.1, 0.5)));
    world.push_back(obj_to_mesh("../objects/suzanne.obj", "../objects/", vec3(0, 0.5, -8), material_type::diffuse, tform, color(0.5, 0.2, 0.1)));
    world.push_back(obj_to_mesh("../objects/chess.obj", "../objects/", vec3(0, -2, -7.5), material_type::diffuse, tform, color(0.2, 0.4, 0.1)));

    // BVH construction
    std::vector<triangle> all_triangles;
    for (object obj : world)
        all_triangles.insert(all_triangles.end(), obj.mesh.begin(), obj.mesh.end());
    bvh bvh_tree;

    auto build_start = std::chrono::system_clock::now(); // bvh build runtime check

    bvh_tree.root = bvh_tree.build_bvh_cycle_axis_equal(all_triangles, 0); // choose BVH contruction strategy

    auto build_end = std::chrono::system_clock::now();
    bvh_tree.stats.build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();

    bvh_tree.stats.collect_bvh_stats(bvh_tree.root.get(), 0, bvh_tree.stats);

    // setup light sources
    std::vector<light_source> lights;

    light_source light1(point3(0, 0, -2), color(1, 1, 1), 0.5f);
    lights.push_back(light1);

    light_source light2(point3(-1, 2.5, -1), color(1, 1, 1), 0.75f);
    lights.push_back(light2);

    /* light_source light3(point3(-2, 2, 3), color(1, 1, 1), 1.0f);
    lights.push_back(light3); */

    // setup camera
    camera cam;
    cam.adjustable = true;         // choose between basic or adjustable camera
    cam.aspect_ratio = 16.0 / 9.0; // ratio width-height
    cam.image_width = 500;         // width of image in pixels

    // Rendering
    auto render_start = std::chrono::system_clock::now(); // bvh traversal runtime check

    aabb::total_tests = 0;

    cam.render(world, lights, bvh_tree);

    std::cout << "Total AABB tests: " << aabb::total_tests << "\n";

    auto render_end = std::chrono::system_clock::now();
    bvh_tree.stats.traverse_time_ms = std::chrono::duration<double, std::milli>(render_end - render_start).count();
    bvh_tree.stats.printStats();
    // std::cout << "Primary rays: " << cam.primary_ray_counter << '\n';
    //  std::cout << "Total rays: " << ray::ray_counter << '\n';
    std::cout << "Total triangle tests: " << triangle::triangleTests << '\n';

    // runtime check
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total runtime (in ms): " << elapsed << '\n';

    // PPM Viewer
    int working = system("..\\ppm-viewer\\ppm.exe image.ppm");
    if (working != 0)
    {
        std::cerr << "Failed to launch PPM viewer.\n";
    }
    return 0;
}