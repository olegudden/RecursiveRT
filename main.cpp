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
    // RUNTIME 1
    auto start = std::chrono::system_clock::now();

    // Transformations TODO: modelview transform instead of geometry
    glm::mat4 tformLight = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.3f, 0.0f));
    tformLight = glm::scale(tformLight, glm::vec3(0.01f, 0.01f, 0.01f));

    glm::mat4 tformSphere = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.4f, 0.0f)); // 0.0,0.4,0.3, scale 0.3
    tformSphere = glm::scale(tformSphere, glm::vec3(0.4f, 0.4f, 0.4f));

    glm::mat4 tformZylinder = glm::translate(glm::mat4(1.0f), glm::vec3(0.55f, 0.1f, 0.0f)); // 0.0,0.4,0.3, scale 0.3
    tformZylinder = glm::rotate(tformZylinder, glm::radians(50.0f), glm::vec3(0.0f, 0.0f, 0.1f));
    tformZylinder = glm::scale(tformZylinder, glm::vec3(0.01f, 0.4f, 0.01f));

    glm::mat4 tformZylinder2 = glm::translate(glm::mat4(1.0f), glm::vec3(-0.55f, 0.1f, 0.0f)); // 0.0,0.4,0.3, scale 0.3
    tformZylinder2 = glm::rotate(tformZylinder2, glm::radians(-50.0f), glm::vec3(0.0f, 0.0f, 0.1f));
    tformZylinder2 = glm::scale(tformZylinder2, glm::vec3(0.01f, 0.4f, 0.01f));

    glm::mat4 tformCone = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.85f, 0.0f)); // 0.0,0.4,0.3, scale 0.3
    tformCone = glm::scale(tformCone, glm::vec3(0.25f, 0.2f, 0.2f));

    glm::mat4 tformSphere1 = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0.75f, 0.0f));
    tformSphere1 = glm::scale(tformSphere1, glm::vec3(0.25f, 0.25f, 0.25f));
    glm::mat4 tformSphere2 = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0.0f, 0.2f)); // 0.5, ,0.2
    tformSphere2 = glm::scale(tformSphere2, glm::vec3(0.25f, 0.25f, 0.25f));
    glm::mat4 tformSphere3 = glm::translate(glm::mat4(1.0f), glm::vec3(-0.5f, 0.75f, 0.0f));
    tformSphere3 = glm::scale(tformSphere3, glm::vec3(0.25f, 0.25f, 0.25f));
    glm::mat4 tformSphere4 = glm::translate(glm::mat4(1.0f), glm::vec3(-0.5f, 0.0f, 0.2f)); // 0.2
    tformSphere4 = glm::scale(tformSphere4, glm::vec3(0.25f, 0.25f, 0.25f));

    glm::mat4 tformCube = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.2f, 0.5f));   // x=0, y=0.5, z=0.75
    tformCube = glm::scale(tformCube, glm::vec3(0.35f, 0.35f, 0.35f));                     // 0.25
    tformCube = glm::rotate(tformCube, glm::radians(20.0f), glm::vec3(-1.0f, 0.0f, 0.0f)); // x:45
    tformCube = glm::rotate(tformCube, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));  // y:45

    glm::mat4 tformCornell = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, 0.8f));      // 0,-0.5,0.8
    tformCornell = glm::rotate(tformCornell, glm::radians(-20.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // x = -20
    // tformCornell = glm::rotate(tformCornell, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 tform = glm::mat4(1.0f); // identity matrix

    glm::mat4 tformTeapot = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.85f)); // (-0.5,-0.2f,1.0f)
    tformTeapot = glm::scale(tformTeapot, glm::vec3(0.008f, 0.008f, 0.008f));

    glm::mat4 tformCorridor = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, -0.5f, -0.7f));

    glm::mat4 tformPixel = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 5.5f, 5.0f));

    glm::mat4 tformRR = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, 2.5f)); // z = 1.5 or 6.5
    tformRR = glm::rotate(tformRR, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 tformSuzanne = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.4f, 0.4f)); // y=0.2,z=-0.3
    tformSuzanne = glm::scale(tformSuzanne, glm::vec3(0.1f, 0.1f, 0.1f));
    tformSuzanne = glm::rotate(tformSuzanne, glm::radians(-25.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 tformDragon = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.25f));
    tformDragon = glm::scale(tformDragon, glm::vec3(1.5f, 1.5f, 1.5f));
    tformDragon = glm::rotate(tformDragon, glm::radians(65.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    tformDragon = glm::rotate(tformDragon, glm::radians(20.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    glm::mat4 tformBike = glm::translate(glm::mat4(1.0f), glm::vec3(-0.15f, -0.25f, 0.5f));
    tformBike = glm::scale(tformBike, glm::vec3(1.0f, 1.0f, 1.0f));
    tformBike = glm::rotate(tformBike, glm::radians(60.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    // add OBJs to world
    // world.push_back(obj_to_mesh("./objects/pixel_sampling_test.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tform, color(0.0, 0.0, 0.0)));
    // world.push_back(obj_to_mesh("./objects/CornellBox-Empty-RGdiscoLight.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformCornell, color(0.0, 0.0, 0.0)));
    world.push_back(obj_to_mesh("./objects/CornellBox-Empty-RGLight.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformCornell, color(0.0, 0.0, 0.0)));
    // world.push_back(obj_to_mesh("./objects/zigzag.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformRR, color(0.0, 0.0, 0.0)));

    // world.push_back(obj_to_mesh("./objects/roadbike.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformBike, color(0.5, 0.5, 1)));
    // world.push_back(obj_to_mesh("./objects/discoball_perfect.obj", "./objects/", vec3(0, 0, 0), material_preset::plastic4, material_type::diffuse, tformSphere, color(1, 1, 1)));
    // world.push_back(obj_to_mesh("./objects/testSphere.obj", "./objects/", vec3(0, 0, 0), material_type::diffuse, tformLight, color(1, 1, 1))); // LIGHTBULB

    // world.push_back(obj_to_mesh("./objects/teapot.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformTeapot, color(0.8, 0.2, 0.5)));
    world.push_back(obj_to_mesh("./objects/icosphere.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformSphere, color(1, 1, 1)));
    /* world.push_back(obj_to_mesh("./objects/zylinder.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformZylinder, color(1, 1, 1)));
    world.push_back(obj_to_mesh("./objects/zylinder.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformZylinder2, color(1, 1, 1)));
    world.push_back(obj_to_mesh("./objects/cone.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformCone, color(1, 1, 1))); */
    //  world.push_back(obj_to_mesh("./objects/suzanne.obj", "./objects/", vec3(0, 0, 0), material_preset::diffuse, material_type::diffuse, tformSuzanne, color(0.8, 0.2, 0.5)));
    //  world.push_back(obj_to_mesh("./objects/cube.obj", "./objects/", vec3(0, 0, 0), material_preset::glass, material_type::diffuse, tformCube, color(1, 1, 1)));
    /* world.push_back(obj_to_mesh("./objects/icosphere.obj", "./objects/", vec3(0, 0, 0), material_preset::plastic1, material_type::specular, tformSphere1, color(1, 1, 0)));
    world.push_back(obj_to_mesh("./objects/icosphere.obj", "./objects/", vec3(0, 0, 0), material_preset::plastic2, material_type::diffuse, tformSphere2, color(1, 0, 1)));
    world.push_back(obj_to_mesh("./objects/icosphere.obj", "./objects/", vec3(0, 0, 0), material_preset::plastic3, material_type::diffuse, tformSphere3, color(0, 1, 1)));
    world.push_back(obj_to_mesh("./objects/icosphere.obj", "./objects/", vec3(0, 0, 0), material_preset::plastic4, material_type::diffuse, tformSphere4, color(1, 1, 1))); */

    // BVH construction
    std::vector<triangle>
        all_triangles;
    std::vector<area_light> area_lights;

    // AREA LIGHTS
    for (object &obj : world) // HERE
    {
        all_triangles.insert(all_triangles.end(), obj.mesh.begin(), obj.mesh.end());
        if (!obj.lights.empty())
        {
            area_light merged = merge_light_triangles(obj.lights);
            area_lights.push_back(merged);
        }

        /*if (!obj.lights.empty())
        {
            auto groups = cluster_light_triangles(obj.lights);

            for (auto &group : groups)
            {
                area_light merged = merge_light_triangles(group);

                merged.light_id = area_lights.size();
                area_lights.push_back(merged);

                // Assign merged light_id to all triangles in this group
                for (auto &tri_light : group)
                {
                    for (triangle &t : tri_light.tris)
                        t.mat.light_id = merged.light_id;
                }
            }
        } */
    }

    compute_lightweights(area_lights);
    for (area_light l : area_lights)
        std::cout << l.selection_weight << std::endl;

    // BVH
    bvh bvh_tree;
    auto build_start = std::chrono::system_clock::now();             // bvh build runtime check
    bvh_tree.root = bvh_tree.build_bvh_binned_sah(all_triangles, 0); // choose BVH contruction strategy
    auto build_end = std::chrono::system_clock::now();
    bvh_tree.stats.build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    bvh_tree.stats.collect_bvh_stats(bvh_tree.root.get(), 0, bvh_tree.stats);

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

    cam.render(world, area_lights, bvh_tree);

    std::cout << "Total AABB tests: " << aabb::total_tests << "\n";

    auto render_end = std::chrono::system_clock::now();
    bvh_tree.stats.traverse_time_ms = std::chrono::duration<double, std::milli>(render_end - render_start).count();
    bvh_tree.stats.printStats();
    // std::cout << "Primary rays: " << cam.primary_ray_counter << '\n';
    // std::cout << "Total rays: " << ray::ray_counter << '\n';
    // std::cout << "Total triangle tests: " << triangle::triangleTests << '\n';

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
