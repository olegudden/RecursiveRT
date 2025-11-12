#include "../hpp/triangle.hpp"
#include <cmath>

// int64_t triangle::triangleTests = 0;

triangle::triangle(const point3 &v0, const point3 &v1, const point3 &v2)
    : v0(v0), v1(v1), v2(v2) {}

triangle::triangle(const point3 &v0, const point3 &v1, const point3 &v2,
                   const vec3 &n0, const vec3 &n1, const vec3 &n2,
                   const material &mat, material_type mt, object *obj)
    : v0(v0), v1(v1), v2(v2), n0(n0), n1(n1), n2(n2),
      mat(mat), mt(mt), obj_ptr(obj) {}

bool triangle::hit(const ray &r, float t_min, float t_max, hit_info &hi) const
{
    // triangleTests++;

    // Möller–Trumbore intersection algorithm
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;

    float eps = 1e-10f;

    vec3 pvec = cross(r.direction(), edge2);
    float det = dot(edge1, pvec);

    if (fabs(det) < eps)
        return false;

    float invDet = 1.0f / det;
    vec3 tvec = r.origin() - v0;

    float u = dot(tvec, pvec) * invDet;
    if (u < -eps || u > 1.0f + eps)
        return false;

    vec3 qvec = cross(tvec, edge1);
    float v = dot(r.direction(), qvec) * invDet;
    if (v < -eps || u + v > 1.0f + eps)
        return false;

    float t = dot(edge2, qvec) * invDet;
    if (t <= t_min + eps || t >= t_max - eps)
        return false;

    hi.t = t;
    hi.normal = unit_vector((1 - u - v) * n0 + u * n1 + v * n2);
    hi.p = r.at(t);
    hi.mat = this->mat;
    hi.mt = this->mt;

    return true;
}

point3 triangle::centroid() const
{
    return (v0 + v1 + v2) / 3.0f;
}

point3 triangle::get_v0() { return v0; }
point3 triangle::get_v1() { return v1; }
point3 triangle::get_v2() { return v2; }

material_type triangle::get_material_type() const { return mt; }

void triangle::set_obj(object *o)
{
    obj_ptr = o;
}
