#include "../hpp/vec3.hpp"

// Constructors
vec3::vec3() : e{0, 0, 0} {}
vec3::vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

// Accessors
float vec3::x() const { return e[0]; }
float vec3::y() const { return e[1]; }
float vec3::z() const { return e[2]; }

// Operators
vec3 vec3::operator-() const { return vec3(-e[0], -e[1], -e[2]); }
float vec3::operator[](int i) const { return e[i]; }
float &vec3::operator[](int i) { return e[i]; }

vec3 &vec3::operator+=(const vec3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

vec3 &vec3::operator*=(float t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

vec3 &vec3::operator/=(float t)
{
    return *this *= 1 / t;
}

// Length functions
float vec3::length() const { return std::sqrt(length_squared()); }
float vec3::length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

// Utility functions
std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

vec3 operator*(float t, const vec3 &v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

vec3 operator*(const vec3 &v, float t)
{
    return t * v;
}

vec3 operator/(const vec3 &v, float t)
{
    return (1 / t) * v;
}

vec3 operator/(const vec3 &u, const vec3 &v)
{
    return vec3(u.x() / v.x(), u.y() / v.y(), u.z() / v.z());
}

float dot(const vec3 &u, const vec3 &v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

vec3 unit_vector(const vec3 &v)
{
    return v / v.length();
}

vec3 min_vec(const vec3 &a, const vec3 &b)
{
    return vec3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
}

vec3 max_vec(const vec3 &a, const vec3 &b)
{
    return vec3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));
}

glm::vec3 to_glm(const vec3 &v)
{
    return glm::vec3(v.x(), v.y(), v.z());
}

vec3 from_glm(const glm::vec3 &v)
{
    return vec3(v.x, v.y, v.z);
}

vec3 perpendicular(const vec3 &v)
{
    // pick the smallest component to avoid near-zero cross products
    if (fabs(v.x()) < fabs(v.y()))
        return unit_vector(cross(v, vec3(1, 0, 0)));
    else
        return unit_vector(cross(v, vec3(0, 1, 0)));
}

// Builds an orthonormal basis (T, B, N) from a unit-length normal N
// T = tangent, B = bitangent
void make_orthonormal_basis(const vec3 &n, vec3 &t, vec3 &b)
{
    if (n.z() < -0.9999999f)
    {
        t = vec3(0, -1, 0);
        b = vec3(-1, 0, 0);
        return;
    }

    float a = 1.0f / (1.0f + n.z());
    float b2 = -n.x() * n.y() * a;
    t = vec3(1.0f - n.x() * n.x() * a, b2, -n.x());
    b = vec3(b2, 1.0f - n.y() * n.y() * a, -n.y());
}
