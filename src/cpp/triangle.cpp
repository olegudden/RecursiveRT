#include "../hpp/triangle.hpp"
#include <iostream>

triangle::triangle(const point3 &v0, const point3 &v1, const point3 &v2)
    : v0(v0), v1(v1), v2(v2) {}

triangle::triangle(const point3 &a, const point3 &b, const point3 &c,
                   const vec3 &n0, const vec3 &n1, const vec3 &n2,
                   const vec3 &uv0, const vec3 &uv1, const vec3 &uv2,
                   const material &mat, const material_type &mt,
                   object *parent, const float area) : v0(a), v1(b), v2(c), n0(n0), n1(n1), n2(n2), uv0(uv0), uv1(uv1), uv2(uv2),
                                                       mat(mat), mt(mt), obj_ptr(parent), area(area) {}

/* bool triangle::hit(const ray &r, float t_min, float t_max, hit_info &hi) const
{
    // triangleTests++;

    // Möller–Trumbore intersection algorithm
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;

    float eps = 1e-15f;

    vec3 pvec = cross(r.direction(), edge2);
    float det = dot(edge1, pvec);

    if (fabs(det) < eps)
        return false;

    float invDet = 1.0f / det;
    vec3 tvec = r.origin() - v0;

    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) //-+eps
        return false;

    vec3 qvec = cross(tvec, edge1);
    float v = dot(r.direction(), qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) //-+eps
        return false;

    float t = dot(edge2, qvec) * invDet;
    if (t <= t_min || t >= t_max) //+-eps
        return false;

    hi.t = t;
    hi.normal = unit_vector((1 - u - v) * n0 + u * n1 + v * n2);
    hi.geometric_normal = unit_vector(cross(edge1, edge2));
    hi.p = r.at(t);
    hi.mat = this->mat;
    hi.mt = this->mt;

    return true;
} */

bool triangle::hit(const ray &r, float t_min, float t_max, hit_info &hi) const
{
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;

    // float eps = 1e-8f * (edge1.length() * edge2.length());
    float eps = 1e-10f;

    // Begin calculating determinant
    vec3 pvec = cross(r.direction(), edge2);
    float det = dot(edge1, pvec);

    // Reject nearly parallel rays
    if (fabs(det) < eps)
        return false;

    float invDet = 1.0f / det;

    // Calculate distance from v0 to ray origin
    vec3 tvec = r.origin() - v0;

    // Calculate u parameter and test bounds
    float u = dot(tvec, pvec) * invDet;
    if (u < -eps || u > 1.0f + eps)
        return false;

    // Prepare to test v parameter
    vec3 qvec = cross(tvec, edge1);

    float v = dot(r.direction(), qvec) * invDet;
    if (v < -eps || u + v > 1.0f + eps)
        return false;

    // Calculate t to intersection point
    float t = dot(edge2, qvec) * invDet;

    if (t <= t_min || t >= t_max)
        return false;

    if (t < 1e-6f)
    {
        std::cerr << "tiny_t self-hit t=" << t << " origin=" << r.origin() << "\n";
        return false; // for DEBUG only
    }

    // compute interpolated shading normal using original floats (still fine)
    hi.t = t;

    // also store shading normal (interpolated or face normal)
    float w = (1.0f - u - v);
    hi.shading_normal = unit_vector(w * n0 + u * n1 + v * n2);
    hi.normal = unit_vector(cross(edge1, edge2));

    hi.p = r.at(t);
    hi.mat = this->mat;
    hi.mt = this->mt;

    // textures
    hi.u = w * uv0.x() + u * uv1.x() + v * uv2.x();
    hi.v = w * uv0.y() + u * uv1.y() + v * uv2.y();

    return true;
}

point3 triangle::centroid() const
{
    return (v0 + v1 + v2) / 3.0f;
}

/* point3 triangle::get_v0() { return v0; }
point3 triangle::get_v1() { return v1; }
point3 triangle::get_v2() { return v2; }

point3 triangle::get_n0() { return n0; }

material_type triangle::get_material_type() const
{
    return mt;
}

void triangle::set_obj(object *o)
{
    obj_ptr = o;
}
*/