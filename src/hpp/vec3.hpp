#ifndef VEC3_H
#define VEC3_H

#include "../../glm/glm.hpp"
#include <ostream>

class vec3
{
public:
    float e[3];

    vec3();
    vec3(float e0, float e1, float e2);

    float x() const;
    float y() const;
    float z() const;

    vec3 operator-() const;
    float operator[](int i) const;
    float &operator[](int i);

    vec3 &operator+=(const vec3 &v);
    vec3 &operator*=(float t);
    vec3 &operator/=(float t);

    float length() const;
    float length_squared() const;
};

// Alias
using point3 = vec3;

// Function declarations
std::ostream &operator<<(std::ostream &out, const vec3 &v);
vec3 operator+(const vec3 &u, const vec3 &v);
vec3 operator-(const vec3 &u, const vec3 &v);
vec3 operator*(const vec3 &u, const vec3 &v);
vec3 operator*(float t, const vec3 &v);
vec3 operator*(const vec3 &v, float t);
vec3 operator/(const vec3 &v, float t);
vec3 operator/(const vec3 &u, const vec3 &v);

float dot(const vec3 &u, const vec3 &v);
vec3 cross(const vec3 &u, const vec3 &v);
vec3 unit_vector(const vec3 &v);
vec3 min_vec(const vec3 &a, const vec3 &b);
vec3 max_vec(const vec3 &a, const vec3 &b);

// GLM helpers
glm::vec3 to_glm(const vec3 &v);
vec3 from_glm(const glm::vec3 &v);

vec3 perpendicular(const vec3 &v);

void make_orthonormal_basis(const vec3 &N, vec3 &T, vec3 &B);

#endif // VEC3_H
