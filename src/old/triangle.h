#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hittable.hpp"
#include "vec3.hpp"
#include "material.hpp"
// #include "object.h"

#include "..\glm\glm.hpp"
#include "..\glm\gtc\matrix_transform.hpp"

class object;

class triangle : public hittable
{
public:
    static int64_t triangleTests;
    // point3 v[3];
    triangle(const point3 &v0, const point3 &v1, const point3 &v2) : v0(v0), v1(v1), v2(v2) {}

    triangle(const point3 &v0, const point3 &v1, const point3 &v2, const vec3 &n0, const vec3 &n1, const vec3 &n2, const material &mat, material_type mt, object *obj = nullptr)
        : v0(v0), v1(v1), v2(v2), n0(n0), n1(n1), n2(n2), mat(mat), mt(mt), obj_ptr(obj) {}

    // bool hit(const ray &r, float t_min, float t_max, hit_info &hi); // TRANSFORMATIONS

    bool hit(const ray &r, float t_min, float t_max, hit_info &hi) const override
    {
        triangleTests++;

        // Möller–Trumbore intersection algorithm
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;

        // float eps = 1e-8f * (edge1.length() * edge2.length());
        float eps = 1e-10f;

        // Begin calculating determinant
        vec3 pvec = cross(r.direction(), edge2); // order of cross product produces correct sign implicitly (-D)
        float det = dot(edge1, pvec);

        // 1) no intersection with plane (ray (nearly) parallel to triangle)
        if (fabs(det) < eps)
            return false;

        float invDet = 1.0f / det;

        // 2) ray-plane intersection outside triangle (using barycentric coordinates u and v)
        // Calculate distance from v0 to ray origin
        vec3 tvec = r.origin() - v0;

        // Calculate u parameter and test bounds
        float u = dot(tvec, pvec) * invDet; // D_1 of Matrix (b|edge20|dir) -> u = D_1/D (Cramer's Rule)
        if (u < -eps || u > 1.0f + eps)
            return false;

        // Prepare to test v parameter
        vec3 qvec = cross(tvec, edge1);

        float v = dot(r.direction(), qvec) * invDet; // D_2 of Matrix (edge10|b|dir) -> v = D_2/D (Cramer's Rule)
        if (v < -eps || u + v > 1.0f + eps)
            return false;

        // 3) ray-plane intersection inside triangle (calculate intersection point)
        float t = dot(edge2, qvec) * invDet; // D_3 of Matrix (edge10|edge20|b) -> t = D_3/D (Cramer's Rule)
        if (t <= t_min + eps || t >= t_max - eps)
            return false;

        // Hit info
        hi.t = t;
        hi.normal = unit_vector((1 - u - v) * n0 + u * n1 + v * n2); // compute intersection normal using barycentric coordinates
        hi.p = r.at(t);
        hi.mat = this->mat;
        hi.mt = this->mt;

        return true;
    }

    point3
    centroid() const // for SAH
    {
        return (v0 + v1 + v2) / 3.0f;
    }

    point3 get_v0() { return v0; }
    point3 get_v1() { return v1; }
    point3 get_v2() { return v2; }

    material_type get_material_type() const { return mt; }

    void set_obj(object *o) { obj_ptr = o; }

private:
    point3 v0, v1, v2;
    vec3 n0, n1, n2;
    material mat;
    material_type mt;
    object *obj_ptr;
};

#endif