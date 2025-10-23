#ifndef AABB_H
#define AABB_H

#include "hittable.h"
#include "vec3.h"
#include "triangle.h"

#include <vector>
#include <cfloat>

class aabb : public hittable
{
public:
    // static counter shared across all AABBs
    inline static long long total_tests = 0;

    aabb() : min_xyz(point3(0, 0, 0)), max_xyz(point3(0, 0, 0)) {}                             // default constructor
    aabb(const point3 &min_xyz, const point3 &max_xyz) : min_xyz(min_xyz), max_xyz(max_xyz) {} // create bounding box
    aabb(std::vector<triangle> mesh)
    {
        // Compute min/max over all triangle vertices
        float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
        float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

        for (triangle tri : mesh)
        {
            for (point3 v : {tri.get_v0(), tri.get_v1(), tri.get_v2()})
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

    bool hit(const ray &r, float t_min, float t_max, hit_info &hi) const override
    {
        // ray - box intersection(slap - test)
        total_tests++;

        /* float inv_x_dir = 1 / r.direction()[0];
        float inv_y_dir = 1 / r.direction()[1];
        float inv_z_dir = 1 / r.direction()[2];

        float tx1 = (min_xyz[0] - r.origin()[0]) * inv_x_dir;
        float tx2 = (max_xyz[0] - r.origin()[0]) * inv_x_dir;

        float ty1 = (min_xyz[1] - r.origin()[1]) * inv_y_dir;
        float ty2 = (max_xyz[1] - r.origin()[1]) * inv_y_dir;

        float tz1 = (min_xyz[2] - r.origin()[2]) * inv_z_dir;
        float tz2 = (max_xyz[2] - r.origin()[2]) * inv_z_dir;

        float tmin = std::max(std::max(std::min(tx1, tx2), std::min(ty1, ty2)), std::min(tz1, tz2));
        float tmax = std::min(std::min(std::max(tx1, tx2), std::max(ty1, ty2)), std::max(tz1, tz2));

        // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
        if (tmax < 0)
        {
            hi.t = tmax;
            hi.p = r.at(hi.t);
            return false;
        }

        // if tmin > tmax, ray doesn't intersect AABB
        if (tmin > tmax)
        {
            hi.t = tmax;
            return false;
        }

        hi.t = tmin;
        hi.p = r.at(hi.t);
        return true; */

        /* // Small epsilon to detect near-parallel directions
        const float EPS_DIR = 1e-5f;

        // Start with the interval provided by the caller
        float t_enter = t_min;
        float t_exit = t_max;

        // Loop over x=0, y=1, z=2
        for (int i = 0; i < 3; ++i)
        {
            float origin = r.origin()[i];
            float dir = r.direction()[i];

            if (std::fabs(dir) < EPS_DIR)
            {
                // Ray is (nearly) parallel to slab. If origin not within slab -> miss.
                if (origin < min_xyz[i] || origin > max_xyz[i])
                    return false;
                // otherwise this axis imposes no restriction
                continue;
            }

            // Compute intersection t-values with the two planes for this axis
            float inv = (std::fabs(dir) > EPS_DIR) ? 1.0f / dir : 0.0f;
            float t1 = (min_xyz[i] - origin) * inv;
            float t2 = (max_xyz[i] - origin) * inv;

            // Ensure t1 <= t2
            if (t1 > t2)
                std::swap(t1, t2);

            // Tighten global interval
            t_enter = std::max(t_enter, t1);
            t_exit = std::min(t_exit, t2);

            // Early rejection
            if (t_enter > t_exit)
                return false;
        }

        // Sanity checks: ensure finite results and that intersection lies within caller's interval
        if (!std::isfinite(t_enter) || !std::isfinite(t_exit))
            return false;
        if (t_enter < t_min || t_enter > t_max)
            return false;

        // Success: fill hit_info with the entry hit
        hi.t = t_enter;
        hi.p = r.at(hi.t);
        return true; */

        const float EPS_DIR = 1e-8f; // you can lower this
        float t_enter = t_min;
        float t_exit = t_max;

        for (int i = 0; i < 3; ++i)
        {
            float origin = r.origin()[i];
            float dir = r.direction()[i];
            float absdir = std::fabs(dir);

            if (absdir < EPS_DIR)
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
    }

    float surface_area() const // for SAH
    {
        vec3 d = max_xyz - min_xyz;
        return 2.0f * (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
    }

    static aabb from_triangle(triangle &tri)
    {
        point3 v0 = tri.get_v0();
        point3 v1 = tri.get_v1();
        point3 v2 = tri.get_v2();

        vec3 small(fmin(fmin(v0.x(), v1.x()), v2.x()),
                   fmin(fmin(v0.y(), v1.y()), v2.y()),
                   fmin(fmin(v0.z(), v1.z()), v2.z()));

        vec3 big(fmax(fmax(v0.x(), v1.x()), v2.x()),
                 fmax(fmax(v0.y(), v1.y()), v2.y()),
                 fmax(fmax(v0.z(), v1.z()), v2.z()));

        return aabb(small, big);
    }

    const point3 &min_point() const { return min_xyz; }
    const point3 &max_point() const { return max_xyz; }
    /* float minX()
    {
        return min_xyz[0];
    }
    float minY()
    {
        return min_xyz[1];
    }
    float minZ()
    {
        return min_xyz[2];
    }

    float maxX()
    {
        return max_xyz[0];
    }
    float maxY()
    {
        return max_xyz[1];
    }
    float maxZ()
    {
        return max_xyz[2];
    } */

    static aabb surrounding_box(const aabb &box0, const aabb &box1) // merges two aabb (for SAH)
    {
        vec3 small(fmin(box0.min_xyz[0], box1.min_xyz[0]),
                   fmin(box0.min_xyz[1], box1.min_xyz[1]),
                   fmin(box0.min_xyz[2], box1.min_xyz[2]));

        vec3 big(fmax(box0.max_xyz[0], box1.max_xyz[0]),
                 fmax(box0.max_xyz[1], box1.max_xyz[1]),
                 fmax(box0.max_xyz[2], box1.max_xyz[2]));

        return aabb(small, big);
    }

    static float intersection_volume(const aabb &a, const aabb &b)
    {
        float x_overlap = std::max(0.0f, std::min(a.max_xyz[0], b.max_xyz[0]) - std::max(a.min_xyz[0], b.min_xyz[0]));
        float y_overlap = std::max(0.0f, std::min(a.max_xyz[1], b.max_xyz[1]) - std::max(a.min_xyz[1], b.min_xyz[1]));
        float z_overlap = std::max(0.0f, std::min(a.max_xyz[2], b.max_xyz[2]) - std::max(a.min_xyz[2], b.min_xyz[2]));
        return x_overlap * y_overlap * z_overlap;
    }

private:
    point3 min_xyz;
    point3 max_xyz;
};

#endif