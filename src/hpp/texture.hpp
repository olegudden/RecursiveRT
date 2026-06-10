#ifndef TEX_HPP
#define TEX_HPP

#include "stb_image.h"
#include "color.hpp"
#include "hittable.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>

struct checker_texture
{
    color color1;
    color color2;
    float scale;

    checker_texture(color c1, color c2, float s)
        : color1(c1), color2(c2), scale(s) {}

    color sample(float u, float v) const
    {
        // 1. Scale the UVs up to create grid cells
        int u_cell = static_cast<int>(std::floor(u * scale));
        int v_cell = static_cast<int>(std::floor(v * scale));

        // 2. Alternate colors based on whether the cell coordinates sum to an even/odd number
        if ((u_cell + v_cell) % 2 == 0)
        {
            return color1; // e.g., Black
        }
        else
        {
            return color2; // e.g., White
        }
    }
};

/*struct wood_texture
{
    color dark_wood;
    color light_wood;
    float ring_frequency;

    wood_texture(color dark, color light, float freq)
        : dark_wood(dark), light_wood(light), ring_frequency(freq) {}

    color sample(float u, float v) const
    {
        // 1. Shift coordinates so (0,0) is in the center of the texture
        float u_centered = u - 0.5f;
        float v_centered = v - 0.5f;

        // 2. Calculate the distance from the center
        // float dist = std::sqrt(u_centered * u_centered + v_centered * v_centered); // wood
        float dist = u_centered + v_centered; // marble

        // 3. Create an oscillating wave based on the distance
        // Adding 1.0 and multiplying by 0.5 normalizes the sine wave from [-1, 1] to [0, 1]
        float wave = 0.5f * (std::sin(dist * ring_frequency) + 1.0f);

        // 4. Linearly interpolate (LERP) between the two wood colors
        return lerp(dark_wood, light_wood, wave);
    }

private:
// Simple helper function for linear interpolation
color lerp(const color &a, const color &b, float t) const
{
    return a * (1.0f - t) + b * t;
}
};*/

// 1. A clean container for an individual loaded image
struct TextureImage
{
    unsigned char *data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 3;
    std::string filename;
};

class TextureManager
{
public:
    // Holds the actual pixel data allocated on the heap
    std::vector<TextureImage> textures;

    // Maps a filename (e.g., "kamen.png") to its index inside the 'textures' vector
    std::unordered_map<std::string, int> filename_to_id;

    ~TextureManager();

    int load_texture(const std::string &filepath);

    color sample_texture_from_cache(const TextureManager &tex_manager, int tex_id, float u, float v);

private:
};
#endif