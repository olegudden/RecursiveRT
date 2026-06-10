#include "hpp/camera.hpp"
#include "hpp/mesh_loader.hpp"
#include "hpp/bvh.hpp"
#include "hpp/texture.hpp"
#include "../glm/gtc/matrix_transform.hpp"

#include <chrono>

std::vector<object> world;

// int ray::ray_counter = 0;
// int64_t triangle::triangleTests = 0;

int main()
{
    // RUNTIME 1
    auto start = std::chrono::system_clock::now();

    // Transformations TODO: modelview transform instead of geometry
    glm::mat4 tformLight = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.3f, 0.0f));
    tformLight = glm::scale(tformLight, glm::vec3(0.01f, 0.01f, 0.01f));

    glm::mat4 tformSphere = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.9f)); // 0.0,0.4,0.3, scale 0.3
    tformSphere = glm::scale(tformSphere, glm::vec3(0.4f, 0.4f, 0.4f));

    glm::mat4 tformZylinder = glm::translate(glm::mat4(1.0f), glm::vec3(0.55f, 0.1f, 0.0f)); // 0.0,0.4,0.3, scale 0.3
    tformZylinder = glm::rotate(tformZylinder, glm::radians(50.0f), glm::vec3(0.0f, 0.0f, 0.1f));
    tformZylinder = glm::scale(tformZylinder, glm::vec3(0.01f, 0.4f, 0.01f));

    glm::mat4 tformCone = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.85f, 0.0f)); // 0.0,0.4,0.3, scale 0.3
    tformCone = glm::scale(tformCone, glm::vec3(0.25f, 0.2f, 0.2f));

    glm::mat4 tformCube = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.5f));    // x=0, y=-0.2, z=0.5
    tformCube = glm::scale(tformCube, glm::vec3(0.35f, 0.35f, 0.35f));                     // 0.25
    tformCube = glm::rotate(tformCube, glm::radians(20.0f), glm::vec3(-1.0f, 0.0f, 0.0f)); // x:20
    tformCube = glm::rotate(tformCube, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));  // y:45

    glm::mat4 tformCornell = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, 0.8f));      // 0,-0.5,0.8
    tformCornell = glm::rotate(tformCornell, glm::radians(-20.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // x = -20
    // tformCornell = glm::rotate(tformCornell, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 tformCornellSphere = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.8f)); // 0,-0.5,0.8
    tformCornellSphere = glm::rotate(tformCornellSphere, glm::radians(-20.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 tform = glm::mat4(1.0f); // identity matrix

    glm::mat4 tformTeapot = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.85f)); // (-0.5,-0.2f,1.0f)
    tformTeapot = glm::scale(tformTeapot, glm::vec3(0.008f, 0.008f, 0.008f));

    glm::mat4 tformSuzanne = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.1f, -0.2f)); // y=0.2,z=-0.3
    tformSuzanne = glm::scale(tformSuzanne, glm::vec3(0.1f, 0.1f, 0.1f));
    tformSuzanne = glm::rotate(tformSuzanne, glm::radians(-25.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 tformDragon = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.25f));
    tformDragon = glm::scale(tformDragon, glm::vec3(1.5f, 1.5f, 1.5f));
    tformDragon = glm::rotate(tformDragon, glm::radians(65.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    tformDragon = glm::rotate(tformDragon, glm::radians(20.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    glm::mat4 tformBike = glm::translate(glm::mat4(1.0f), glm::vec3(-0.15f, -0.25f, 0.5f));
    tformBike = glm::scale(tformBike, glm::vec3(1.0f, 1.0f, 1.0f));
    tformBike = glm::rotate(tformBike, glm::radians(60.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 tformTest = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0.2, 0));
    tformTest = glm::scale(tformTest, glm::vec3(3.5f, 3.5f, 3.5f));
    // tformTest = glm::rotate(tformTest, glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 0.0f));

    glm::mat4 tformCrack = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.7f, 0.25f)); // 0,-0.5,0.8

    glm::mat4 tformLiving = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, -0.5f, 3.0f));
    tformLiving = glm::rotate(tformLiving, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 tformSibenik = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 10.0f, -12.0f)); // y 12, z -12
    tformSibenik = glm::rotate(tformSibenik, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    TextureManager tex_manager;

    // add OBJs to world
    world.push_back(obj_to_mesh("./objects/CornellBox-Sphere.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformCornellSphere, color(0.0, 0.0, 0.0), tex_manager));
    // world.push_back(obj_to_mesh("./objects/suzanne.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformSuzanne, color(0.0, 0.0, 0.0), tex_manager));
    // world.push_back(obj_to_mesh("./objects/door_crack.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformLiving, color(0.0, 0.0, 0.0)));
    // world.push_back(obj_to_mesh("./objects/sibenik.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformSibenik, color(0.0, 0.0, 0.0), tex_manager));
    // world.push_back(obj_to_mesh("./objects/living_room.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformLiving, color(0.0, 0.0, 0.0), tex_manager));

    // world.push_back(obj_to_mesh("./objects/discoball_perfect.obj", "./objects/", vec3(0, 0, 0), material_preset::plastic4, material_type::diffuse, tformSphere, color(1, 1, 1)));
    // world.push_back(obj_to_mesh("./objects/testSphere.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformLight, color(1, 1, 1))); // LIGHTBULB

    // world.push_back(obj_to_mesh("./objects/icosphere.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformSphere, color(1, 1, 1), tex_manager));
    // world.push_back(obj_to_mesh("./objects/crate.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformCube, color(1, 1, 1), tex_manager));
    // world.push_back(obj_to_mesh("./objects/icosphere.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformSphere, color(1, 1, 1), tex_manager));
    // world.push_back(obj_to_mesh("./objects/head.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformTest, color(1, 1, 1)));

    // BVH construction
    std::vector<triangle> all_triangles;
    std::vector<area_light> area_lights;

    // AREA LIGHTS
    for (object &obj : world) // HERE
    {
        all_triangles.insert(all_triangles.end(), obj.mesh.begin(), obj.mesh.end());
        for (const area_light &L : obj.lights)
        {
            area_lights.push_back(L);
        }
    }

    // BVH (TODO: parallel building)
    bvh bvh_tree;
    auto build_start = std::chrono::system_clock::now();             // bvh build runtime check
    bvh_tree.root = bvh_tree.build_bvh_binned_sah(all_triangles, 0); // choose BVH contruction strategy
    auto build_end = std::chrono::system_clock::now();
    bvh_tree.stats.build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    bvh_tree.stats.collect_bvh_stats(bvh_tree.root.get(), 0, bvh_tree.stats);

    // 1. Setup the global data structure
    VoxelGrid guiding_grid;
    guiding_grid.initialize(world[0].box.min_point() - point3(1e-04f, 1e-04f, 1e-04f), world[0].box.max_point() + point3(1e-04f, 1e-04f, 1e-04f), 16, 16, 16);

    // setup point light sources
    std::vector<light_source> lights;
    /* light_source light1(point3(0, 1.3, 2), color(1, 1, 1), 1.0f);
    lights.push_back(light1);

    light_source light2(point3(0, 8, -2), color(1, 1, 1), 1.0f);
    lights.push_back(light2);*/

    // setup area light source
    /* area_light al;
    al.origin = point3(-0.24, 1.98, 0.16);
    al.e1 = vec3(0.47, 0, 0);  // (0.23 - -0.24) in X
    al.e2 = vec3(0, 0, -0.38); // (-0.22 - 0.16) in Z
    al.normal = vec3(0, -1, 0);
    al.emission = color(25, 25, 25); // strong light (adjust) */

    // CAMERA
    camera cam;
    cam.adjustable = true; // choose between basic or adjustable camera
    // cam.aspect_ratio = 16.0 / 9.0; // ratio width-height
    cam.aspect_ratio = 1.0;
    cam.image_width = 400; // width of image in pixels

    // RENDER
    auto render_start = std::chrono::system_clock::now(); // bvh traversal runtime check

    aabb::total_tests = 0;

    cam.render(world, area_lights, bvh_tree, guiding_grid, tex_manager);

    std::cout << "Total AABB tests: " << aabb::total_tests << "\n";

    auto render_end = std::chrono::system_clock::now();
    bvh_tree.stats.traverse_time_ms = std::chrono::duration<double, std::milli>(render_end - render_start).count();
    bvh_tree.stats.printStats();
    // std::cout << "Primary rays: " << cam.primary_ray_counter << '\n';
    // std::cout << "Total rays: " << ray::ray_counter << '\n';
    // std::cout << "Total triangle tests: " << triangle::triangleTests << '\n';

    std::cout << "direct: " << cam.directHits << '\n';
    std::cout << "indirect: " << cam.indirectHits << '\n';
    std::cout << "misses: " << cam.misses << '\n';
    std::cout << "timeout: " << cam.timedOut << '\n';

    // RUNTIME 2
    auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration<double>(end - start).count();
    std::cout << "Total runtime (in s): " << elapsed << '\n';

    // PPM Viewer
    int working = system(".\\ppm-viewer\\ppm.exe image.ppm");
    if (working != 0)
    {
        std::cerr << "Failed to launch PPM viewer.\n";
    }
    return 0;
}
