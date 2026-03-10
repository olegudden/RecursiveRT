#include "../hpp/aabb.hpp"
#include <cfloat>
#include <cmath>
#include <algorithm>

aabb::aabb() : min_xyz(point3(0, 0, 0)), max_xyz(point3(0, 0, 0)) {}

aabb::aabb(const point3 &min_xyz, const point3 &max_xyz)
    : min_xyz(min_xyz), max_xyz(max_xyz) {}

aabb::aabb(std::vector<triangle> mesh)
{
    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

    for (auto &tri : mesh)
    {
        for (const point3 &v : {tri.get_v0(), tri.get_v1(), tri.get_v2()})
        {
            min_x = std::min(min_x, v.x());
            min_y = std::min(min_y, v.y());
            min_z = std::min(min_z, v.z());
            max_x = std::max(max_x, v.x());
            max_y = std::max(max_y, v.y());
            max_z = std::max(max_z, v.z());
        }
    }
    min_xyz = point3(min_x, min_y, min_z);
    max_xyz = point3(max_x, max_y, max_z);
}

/* bool aabb::hit(const ray &r, float t_min, float t_max, hit_info &hi) const
{
    // total_tests++;
    const float EPS_DIR = 1e-8f;
    float t_enter = t_min;
    float t_exit = t_max;

    for (int i = 0; i < 3; ++i)
    {
        float origin = r.origin()[i];
        float dir = r.direction()[i];
        if (fabs(dir) < EPS_DIR)
        {
            if (origin < min_xyz[i] || origin > max_xyz[i])
                return false;
            continue;
        }
        float inv = 1.0f / dir;
        float t1 = (min_xyz[i] - origin) * inv;
        float t2 = (max_xyz[i] - origin) * inv;
        if (t1 > t2)
            std::swap(t1, t2);
        t_enter = std::max(t_enter, t1);
        t_exit = std::min(t_exit, t2);
        if (t_enter > t_exit)
            return false;
    }
    hi.t = t_enter;
    hi.p = r.at(hi.t);
    return true;
} */

bool aabb::hit(const ray &r, float t_min, float t_max, hit_info &) const
{
    const float EPS_DIR = 1e-8f;
    float t_enter = t_min;
    float t_exit = t_max;

    for (int i = 0; i < 3; ++i)
    {
        float origin = r.origin()[i];
        float dir = r.direction()[i];

        if (fabs(dir) < EPS_DIR)
        {
            if (origin < min_xyz[i] || origin > max_xyz[i])
                return false;
            continue;
        }

        float inv = 1.0f / dir;
        float t1 = (min_xyz[i] - origin) * inv;
        float t2 = (max_xyz[i] - origin) * inv;
        if (t1 > t2)
            std::swap(t1, t2);

        t_enter = std::max(t_enter, t1);
        t_exit = std::min(t_exit, t2);

        if (t_enter > t_exit)
            return false;
    }
    return true;
}

float aabb::surface_area() const
{
    vec3 d = max_xyz - min_xyz;
    return 2.0f * (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
}

aabb aabb::from_triangle(triangle &tri)
{
    point3 v0 = tri.get_v0(), v1 = tri.get_v1(), v2 = tri.get_v2();
    vec3 small(fmin(fmin(v0.x(), v1.x()), v2.x()),
               fmin(fmin(v0.y(), v1.y()), v2.y()),
               fmin(fmin(v0.z(), v1.z()), v2.z()));
    vec3 big(fmax(fmax(v0.x(), v1.x()), v2.x()),
             fmax(fmax(v0.y(), v1.y()), v2.y()),
             fmax(fmax(v0.z(), v1.z()), v2.z()));
    return aabb(small, big);
}

const point3 &aabb::min_point() const { return min_xyz; }
const point3 &aabb::max_point() const { return max_xyz; }

aabb aabb::surrounding_box(const aabb &box0, const aabb &box1)
{
    vec3 small(fmin(box0.min_xyz[0], box1.min_xyz[0]),
               fmin(box0.min_xyz[1], box1.min_xyz[1]),
               fmin(box0.min_xyz[2], box1.min_xyz[2]));
    vec3 big(fmax(box0.max_xyz[0], box1.max_xyz[0]),
             fmax(box0.max_xyz[1], box1.max_xyz[1]),
             fmax(box0.max_xyz[2], box1.max_xyz[2]));
    return aabb(small, big);
}

float aabb::intersection_volume(const aabb &a, const aabb &b)
{
    float x_overlap = std::max(0.0f, std::min(a.max_xyz[0], b.max_xyz[0]) - std::max(a.min_xyz[0], b.min_xyz[0]));
    float y_overlap = std::max(0.0f, std::min(a.max_xyz[1], b.max_xyz[1]) - std::max(a.min_xyz[1], b.min_xyz[1]));
    float z_overlap = std::max(0.0f, std::min(a.max_xyz[2], b.max_xyz[2]) - std::max(a.min_xyz[2], b.min_xyz[2]));
    return x_overlap * y_overlap * z_overlap;
}
