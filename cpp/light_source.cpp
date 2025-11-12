#include "../hpp/light_source.hpp"

float rand_float()
{
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(generator);
}

point3 sample_area_light(const area_light &light)
{
    float s1 = rand_float();
    float s2 = rand_float();
    return light.origin + s1 * light.e1 + s2 * light.e2;
}

light_source::light_source()
    : pos(0, 0, 0), col(1, 1, 1), intensity(0.0f) {}

light_source::light_source(point3 pos, color col, float intensity)
    : pos(pos), col(col), intensity(intensity) {}
