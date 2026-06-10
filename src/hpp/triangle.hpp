#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "hittable.hpp"

// Forward declaration to avoid circular include
class object;

class triangle : public hittable
{
public:
    point3 v0, v1, v2;
    vec3 n0, n1, n2;
    material mat;
    material_type mt;
    object *obj_ptr;

    vec3 uv0, uv1, uv2;
    float area;

    triangle(const point3 &v0, const point3 &v1, const point3 &v2);

    /* triangle(const point3 &v0, const point3 &v1, const point3 &v2,
             const vec3 &n0, const vec3 &n1, const vec3 &n2,
             const material &mat, material_preset mt, object *obj = nullptr); // material_type */

    triangle(const point3 &a, const point3 &b, const point3 &c,
             const vec3 &n0, const vec3 &n1, const vec3 &n2,
             const vec3 &uv0, const vec3 &uv1, const vec3 &uv2,
             const material &mat, const material_type &mt,
             object *parent, const float area);

    bool hit(const ray &r, float t_min, float t_max, hit_info &hi) const override;

    point3 centroid() const;

private:
};

#endif
