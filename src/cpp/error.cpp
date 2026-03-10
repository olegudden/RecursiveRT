#include "../hpp/error.hpp"
#include <fstream>
#include <cmath>

ErrorAnalyzer::ErrorAnalyzer(int w, int h)
    : width(w), height(h), stats(w * h) {}

ErrorAnalyzer::ErrorAnalyzer() {}

void ErrorAnalyzer::add_pixel_sample(int x, int y, const color &c)
{
    add_sample(stats[y * width + x], c);
}

color ErrorAnalyzer::get_variance(int x, int y) const
{
    return variance(stats[y * width + x]);
}

float ErrorAnalyzer::compute_global_noise() const
{
    double sum = 0.0;
    for (const auto &s : stats)
    {
        color v = variance(s);
        sum += (v.x() + v.y() + v.z()) / 3.0;
    }
    return float(sum / stats.size());
}

// -------------------------
// MSE / PSNR / SSIM
// -------------------------
float ErrorAnalyzer::mse(const std::vector<color> &A,
                         const std::vector<color> &B) const
{
    double s = 0.0;
    for (int i = 0; i < A.size(); i++)
    {
        vec3 d = A[i] - B[i];
        s += (d.x() * d.x() + d.y() * d.y() + d.z() * d.z()) / 3.0;
    }
    return float(s / A.size());
}

float ErrorAnalyzer::psnr(const std::vector<color> &A,
                          const std::vector<color> &B) const
{
    float MSE = mse(A, B);
    if (MSE == 0.0f)
        return 100.0f;
    return 10.f * std::log10((255.0f * 255.0f) / MSE); // max = 1.0f
}

float ErrorAnalyzer::ssim(const std::vector<color> &A,
                          const std::vector<color> &B) const
{
    double meanA = 0, meanB = 0;
    double varA = 0, varB = 0, covAB = 0;

    int N = A.size();

    // means
    for (int i = 0; i < N; i++)
    {
        float a = (A[i].x() + A[i].y() + A[i].z()) / 3;
        float b = (B[i].x() + B[i].y() + B[i].z()) / 3;
        meanA += a;
        meanB += b;
    }
    meanA /= N;
    meanB /= N;

    // variances
    for (int i = 0; i < N; i++)
    {
        float a = (A[i].x() + A[i].y() + A[i].z()) / 3;
        float b = (B[i].x() + B[i].y() + B[i].z()) / 3;
        varA += (a - meanA) * (a - meanA);
        varB += (b - meanB) * (b - meanB);
        covAB += (a - meanA) * (b - meanB);
    }
    varA /= N;
    varB /= N;
    covAB /= N;

    const float C1 = 1e-4f, C2 = 9e-4f;

    return ((2 * meanA * meanB + C1) * (2 * covAB + C2)) /
           ((meanA * meanA + meanB * meanB + C1) * (varA + varB + C2));
}

std::vector<color> ErrorAnalyzer::load_ppm(const std::string &filename)
{
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("Cannot open file");

    std::string magic;
    int width, height, maxval;
    in >> magic >> width >> height >> maxval;

    std::vector<color> img(width * height);

    for (int i = 0; i < width * height; i++)
    {
        int r, g, b;
        in >> r >> g >> b;
        img[i] = color(r / float(maxval),
                       g / float(maxval),
                       b / float(maxval));
    }

    return img;
}
