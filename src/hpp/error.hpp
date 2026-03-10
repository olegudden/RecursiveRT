#include <vector>
#include "vec3.hpp"
#include "color.hpp"

struct PixelStats
{
    int count = 0;
    color mean = color(0, 0, 0);
    color M2 = color(0, 0, 0);
};

inline void add_sample(PixelStats &s, const color &x)
{
    s.count++;
    color delta = x - s.mean;
    s.mean += delta * (1.0f / s.count);
    color delta2 = x - s.mean;
    s.M2 += delta * delta2;
}

inline color variance(const PixelStats &s)
{
    if (s.count < 2)
        return color(0, 0, 0);
    return s.M2 * (1.0f / (s.count - 1));
}

class ErrorAnalyzer
{
public:
    ErrorAnalyzer(int width, int height);
    ErrorAnalyzer();

    void add_pixel_sample(int x, int y, const color &c);
    color get_variance(int x, int y) const;
    float compute_global_noise() const;

    // error metrics
    float mse(const std::vector<color> &A, const std::vector<color> &B) const;
    float psnr(const std::vector<color> &A, const std::vector<color> &B) const;
    float ssim(const std::vector<color> &A, const std::vector<color> &B) const;

    // helper
    std::vector<color> load_ppm(const std::string &filename);

    const std::vector<PixelStats> &get_stats() const { return stats; }

private:
    int width, height;
    std::vector<PixelStats> stats;
};
