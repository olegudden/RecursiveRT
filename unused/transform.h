/* #ifndef TRANSFROM_H
#define TRANSFROM_H

#include "vec3.h"
#include "vec4.h"
#include "mat4.h"

class tranform
{
public:
    vec3 transform_point(const mat4 &m, const vec3 &p)
    {
        vec4 v = m * vec4(p, 1.0f);
        return vec3(v.x / v.w, v.y / v.w, v.z / v.w);
    }

    vec3 transform_direction(const mat4 &m, const vec3 &d)
    {
        vec4 v = m * vec4(d, 0.0f);
        return vec3(v.x, v.y, v.z); // no division by w
    }
};

#endif */