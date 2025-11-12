#ifndef LIGHT_SOURCE_H
#define LIGHT_SOURCE_H

#include "vec3.hpp"
#include "color.hpp"

class light_source
{
public:
    point3 pos = point3(0, 0, 0);
    color col = color(1, 1, 1);
    float intensity = 0.0f;
    light_source(point3 pos, color col, float intensity) : pos(pos), col(col), intensity(intensity) {}
    light_source() = default;

private:
};
#endif