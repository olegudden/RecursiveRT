#include "../hpp/color.hpp"
#include <iostream>

color clamp_color(const color &c)
{
    return color(
        std::clamp(c.x(), 0.0f, 1.0f),
        std::clamp(c.y(), 0.0f, 1.0f),
        std::clamp(c.z(), 0.0f, 1.0f));
}

vec3 reinhard_tonemap(const vec3 &c)
{
    return color(
        c.x() / (1.0f + c.x()),
        c.y() / (1.0f + c.y()),
        c.z() / (1.0f + c.z()));
}

color reinhard_extended(const color &c, float white_point)
{
    return color(
        c.x() * (1.0f + c.x() / (white_point * white_point)) / (1.0f + c.x()),
        c.y() * (1.0f + c.y() / (white_point * white_point)) / (1.0f + c.y()),
        c.z() * (1.0f + c.z() / (white_point * white_point)) / (1.0f + c.z()));
}

color exposure_tonemap(const color &c, float exposure)
{
    return color(
        1.0f - std::exp(-c.x() * exposure),
        1.0f - std::exp(-c.y() * exposure),
        1.0f - std::exp(-c.z() * exposure));
}

color aces_tonemap(const color &c)
{
    color a = c * 0.6f; // tweak this scale
    return (a * (2.51f * a + color(0.03f, 0.03f, 0.03f))) /
           (a * (2.43f * a + color(0.59f, 0.59f, 0.59f)) + color(0.14f, 0.14f, 0.14f));
}

vec3 gamma_correct(const vec3 &c, float gamma)
{
    return color(
        std::pow(c.x(), 1.0f / gamma),
        std::pow(c.y(), 1.0f / gamma),
        std::pow(c.z(), 1.0f / gamma));
}

void std_write_color(std::ostream &out, const color &pixel_color)
{
    int rbyte = static_cast<int>(255.999 * pixel_color.x());
    int gbyte = static_cast<int>(255.999 * pixel_color.y());
    int bbyte = static_cast<int>(255.999 * pixel_color.z());

    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

void write_color(std::ostream &out, const color &pixel_color)
{
    color mapped = reinhard_tonemap(pixel_color);
    // color mapped = exposure_tonemap(pixel_color, 0.7f);
    // mapped = gamma_correct(mapped);

    int rbyte = static_cast<int>(255.999 * mapped.x());
    int gbyte = static_cast<int>(255.999 * mapped.y());
    int bbyte = static_cast<int>(255.999 * mapped.z());

    if (rbyte < 0 || gbyte < 0 || bbyte < 0)
        std::cout << "ERROR PIXEL" << std::endl;
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

float luminance(const vec3 &c)
{
    return 0.2126f * c.x() + 0.7152f * c.y() + 0.0722f * c.z();
}

color sample_texture(unsigned char *data, int width, int height, float u, float v)
{
    if (!data)
        return color(1, 0, 1); // Bright pink for "Missing Texture"

    // 1. Clamp or Repeat the coordinates
    // This ensures u,v stay between 0 and 1
    u = u - floor(u);
    v = v - floor(v);

    // 2. Map 0..1 to 0..width/height
    int i = static_cast<int>(u * width);
    int j = static_cast<int>((1.0f - v) * height); // Flip V: many formats are top-to-bottom

    // 3. Stay within bounds
    i = std::max(0, std::min(i, width - 1));
    j = std::max(0, std::min(j, height - 1));

    // 4. Read the RGB bytes and convert to 0..1 float color
    int index = 3 * (j * width + i);
    float r = data[index + 0] / 255.0f;
    float g = data[index + 1] / 255.0f;
    float b = data[index + 2] / 255.0f;

    return color(r, g, b);
}
