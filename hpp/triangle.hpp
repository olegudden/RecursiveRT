#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "hittable.hpp"
#include "vec3.hpp"
#include "material.hpp"

// Forward declaration to avoid circular include
class object;

// If you need glm in the .cpp, include it there instead of here

class triangle : public hittable
{
public:
    static int64_t triangleTests;

    triangle(const point3 &v0, const point3 &v1, const point3 &v2);
    triangle(const point3 &v0, const point3 &v1, const point3 &v2,
             const vec3 &n0, const vec3 &n1, const vec3 &n2,
             const material &mat, material_type mt, object *obj = nullptr);

    bool hit(const ray &r, float t_min, float t_max, hit_info &hi) const override;
    point3 centroid() const;

    point3 get_v0();
    point3 get_v1();
    point3 get_v2();

    material_type get_material_type() const;
    void set_obj(object *o);

private:
    point3 v0, v1, v2;
    vec3 n0, n1, n2;
    material mat;
    material_type mt;
    object *obj_ptr;
};

#endif
