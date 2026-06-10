#ifndef AABB_HPP
#define AABB_HPP

#include "triangle.hpp"
#include <vector>

class aabb : public hittable
{
public:
    inline static long long total_tests = 0;

    aabb();
    aabb(const point3 &min_xyz, const point3 &max_xyz);
    aabb(std::vector<triangle> mesh);

    bool hit(const ray &r, float t_min, float t_max, hit_info &hi) const override;
    float surface_area() const;
    static aabb from_triangle(triangle &tri);
    const point3 &min_point() const;
    const point3 &max_point() const;
    static aabb surrounding_box(const aabb &box0, const aabb &box1);
    static float intersection_volume(const aabb &a, const aabb &b);

private:
    point3 min_xyz;
    point3 max_xyz;
};

#endif
