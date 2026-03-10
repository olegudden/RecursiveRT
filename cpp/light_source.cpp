#include "../hpp/light_source.hpp"

float rand_float()
{
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(generator);
}

float random_float(float min, float max)
{
    return min + (max - min) * rand_float();
}
point3 sample_area_light(const area_light &light)
{
    float s1 = rand_float();
    float s2 = rand_float();
    return light.origin + s1 * light.e1 + s2 * light.e2;
}

std::vector<point3> uniform_sample_light(const area_light &L, int N)
{
    // uniform random sampling of area light
    std::vector<point3> samples;
    samples.reserve(N);

    for (int i = 0; i < N; ++i)
    {
        // uniform u, v in [0,1)
        double u = rand_float();
        double v = rand_float();

        // sample point on the light
        point3 p = L.origin + u * L.e1 + v * L.e2;
        samples.push_back(p);
    }

    return samples;
}

std::vector<point3> stratified_sample_light(const area_light &L, int N) // maybe try uniform
{
    // stratified sampling of area light
    std::vector<point3> samples;
    samples.reserve(N * N);

    for (int py = 0; py < N; ++py)
    {
        for (int px = 0; px < N; ++px)
        {
            // stratified u, v in [0,1]
            double u = (px + rand_float()) / N;
            double v = (py + rand_float()) / N;

            // sample point on the light
            point3 p = L.origin + u * L.e1 + v * L.e2;
            samples.push_back(p);
        }
    }
    return samples;
}

bool visible(const point3 &p, const point3 &light_point, const bvh &bvh_tree)
{
    vec3 dir = light_point - p;
    float dist = dir.length();
    dir = dir / dist;

    ray shadow_ray(p + dir * 1e-4f, dir);
    hit_info temp;
    if (bvh_tree.hit(shadow_ray, 1e-4f, dist - 1e-4f, temp))
    {
        if (temp.mat.is_emissive())
            return true;
        return false;
    }
    return true;
}

void compute_lightweights(std::vector<area_light> &lights)
{
    float total = 0;
    for (auto &L : lights)
    {
        L.selection_weight = L.emission.length() * L.area;
        total += L.selection_weight;
    }
    for (auto &L : lights)
    {
        L.selection_weight /= total;
    }
}

area_light sample_multiple_AL(const std::vector<area_light> &lights)
{
    float u = rand_float();
    int chosen = 0;
    float accum = 0.0f;

    for (int i = 0; i < lights.size(); i++)
    {
        accum += lights[i].selection_weight;
        if (u < accum)
        {
            chosen = i;
            break;
        }
    }
    return lights[chosen];
}

light_source::light_source()
    : pos(0, 0, 0), col(1, 1, 1), intensity(0.0f) {}

light_source::light_source(point3 pos, color col, float intensity)
    : pos(pos), col(col), intensity(intensity) {}
