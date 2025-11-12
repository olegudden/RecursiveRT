#ifndef COLOR_HPP
#define COLOR_HPP

#include "vec3.hpp"
#include <iostream>
#include <algorithm>

using color = vec3;

// Function declarations
color clamp_color(const color &c);
vec3 reinhard_tonemap(const vec3 &c);
color reinhard_extended(const color &c, float white_point = 4.0f);
color exposure_tonemap(const color &c, float exposure = 1.0f);
color aces_tonemap(const color &c);
vec3 gamma_correct(const vec3 &c, float gamma = 2.2f);
void std_write_color(std::ostream &out, const color &pixel_color);
void write_color(std::ostream &out, const color &pixel_color);

#endif // COLOR_HPP
