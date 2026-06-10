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

point3 sample(vec3 &tri_normal, const area_light &light, float &out_pdf)
{
    if (light.tris.empty())
    {
        out_pdf = 0.0f;
        return point3(0, 0, 0);
    }

    // importance sample triangle by area
    // Roll a number scaled to the total area of this light source
    float r_tri = rand_float() * light.area;
    // std::lower_bound still works flawlessly because the raw areas are strictly increasing!
    auto it = std::lower_bound(light.tris_cdf.begin(), light.tris_cdf.end(), r_tri);
    int idx = std::distance(light.tris_cdf.begin(), it);
    if (idx >= light.tris.size())
        idx = light.tris.size() - 1;
    const triangle &tri = light.tris[idx];

    // random
    // const triangle &tri = light.tris[rand() % light.tris.size()];

    // pick random point on triangle (barycentric coordinates)
    float r1 = rand_float();
    float r2 = rand_float();
    float sqrt_r1 = std::sqrt(r1);
    float u = 1.0f - sqrt_r1;
    float v = r2 * sqrt_r1;

    float w = 1.0f - u - v;
    tri_normal = unit_vector(w * tri.n0 + u * tri.n1 + v * tri.n2);
    out_pdf = 1.0f / light.area;

    return tri.v0 + u * (tri.v1 - tri.v0) + v * (tri.v2 - tri.v0);
}

point3 sample_stratified(vec3 &tri_normal, const area_light &light, float &out_pdf, float r1, float r2)
{
    if (light.tris.empty())
    {
        out_pdf = 0.0f;
        return point3(0, 0, 0);
    }

    // We still use a raw random variable to select WHICHEVER triangle inside the window instance
    float r_tri = rand_float() * light.area;

    auto it = std::lower_bound(light.tris_cdf.begin(), light.tris_cdf.end(), r_tri);
    int idx = std::distance(light.tris_cdf.begin(), it);

    if (idx >= light.tris.size())
        idx = light.tris.size() - 1;
    const triangle &tri = light.tris[idx];

    // Map our incoming stratified grid positions uniformly across the triangle area
    float sqrt_r1 = std::sqrt(r1);
    float u = 1.0f - sqrt_r1;
    float v = r2 * sqrt_r1;

    float w = 1.0f - u - v;
    tri_normal = unit_vector(w * tri.n0 + u * tri.n1 + v * tri.n2);
    out_pdf = 1.0f / light.area;

    return tri.v0 + u * (tri.v1 - tri.v0) + v * (tri.v2 - tri.v0);
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

void compute_global_light_weights(std::vector<area_light> &lights)
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

std::vector<float> compute_local_light_weights(const point3 &hit_p, const vec3 &surface_normal, const std::vector<area_light> &lights)
{
    std::vector<float> local_weights(lights.size(), 0.0f);
    float total = 0.0f;

    for (int i = 0; i < lights.size(); i++)
    {
        // distance
        vec3 Lvec = lights[i].origin - hit_p;
        float dist_squared = Lvec.length_squared();
        // cosine angles
        vec3 wo = unit_vector(Lvec);
        float cos_s = dot(surface_normal, wo);
        float cos_l = dot(lights[i].normal, -wo);
        if (cos_s < 0.0f || cos_l < 0.0f)
            continue;
        // probability = power * solid angle
        float power = luminance(lights[i].emission);
        // float power = lights[i].emission.length();
        float solid_angle = (lights[i].area * cos_l) / dist_squared; // * cos_l) / dist_squared
        local_weights[i] = power * solid_angle;
        total += local_weights[i];
    }
    // normalization
    if (total > 0.0f)
    {
        for (int i = 0; i < lights.size(); i++)
        {
            local_weights[i] /= total;
        }
    }
    else
    {
        // Fallback if everything is in shadow/backfaced: pick uniformly
        float uniform_w = 1.0f / static_cast<float>(lights.size());
        std::fill(local_weights.begin(), local_weights.end(), uniform_w);
    }
    return local_weights;
}

const area_light &sample_multiple_AL(const std::vector<area_light> &lights, const std::vector<float> &local_probs, float &chosen_prob, int &chosen_id)
{
    float r = rand_float();
    float accum = 0.0f;
    for (int i = 0; i < lights.size(); i++)
    {
        accum += local_probs[i];
        if (r <= accum)
        {
            chosen_prob = local_probs[i];
            chosen_id = i;
            return lights[i];
        }
    }
    chosen_prob = local_probs.back();
    chosen_id = lights.size() - 1;
    return lights.back();
}

light_source::light_source()
    : pos(0, 0, 0), col(1, 1, 1), intensity(0.0f) {}

light_source::light_source(point3 pos, color col, float intensity)
    : pos(pos), col(col), intensity(intensity) {}
