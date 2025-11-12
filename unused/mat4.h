/* #ifndef MAT4_H
#define MAT4_H

#include "vec4.h"

class mat4
{
public:
    float m[4][4];

    mat4() { identity(); }

    void identity()
    {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    mat4 identity_matrix()
    {
        mat4 r;
        r.identity();
        return r;
    }

    mat4 translate(const vec3 &t)
    {
        mat4 r = identity_matrix();
        r.m[0][3] = t.x();
        r.m[1][3] = t.y();
        r.m[2][3] = t.z();
        return r;
    }

    mat4 scale(const vec3 &s)
    {
        mat4 r = {};
        r.m[0][0] = s.x();
        r.m[1][1] = s.y();
        r.m[2][2] = s.z();
        r.m[3][3] = 1.0f;
        return r;
    }

    mat4 rotate_x(float angle_rad)
    {
        float c = cos(angle_rad), s = sin(angle_rad);
        mat4 r = identity_matrix();
        r.m[1][1] = c;
        r.m[1][2] = -s;
        r.m[2][1] = s;
        r.m[2][2] = c;
        return r;
    }

    mat4 rotate_y(float angle_rad)
    {
        float c = cos(angle_rad), s = sin(angle_rad);
        mat4 r = identity_matrix();
        r.m[0][0] = c;
        r.m[0][2] = s;
        r.m[2][0] = -s;
        r.m[2][2] = c;
        return r;
    }

    mat4 rotate_z(float angle_rad)
    {
        float c = cos(angle_rad), s = sin(angle_rad);
        mat4 r = identity_matrix();
        r.m[0][0] = c;
        r.m[0][1] = -s;
        r.m[1][0] = s;
        r.m[1][1] = c;
        return r;
    }

    // Matrix multiplication
    mat4 operator*(const mat4 &b) const
    {
        mat4 r = {};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    r.m[i][j] += m[i][k] * b.m[k][j];
        return r;
    }

    // Matrix * vec4
    vec4 operator*(const vec4 &v) const
    {
        vec4 r;
        r[0] = m[0][0] * v.x() + m[0][1] * v.y() + m[0][2] * v.z() + m[0][3] * v.w();
        r[1] = m[1][0] * v.x() + m[1][1] * v.y() + m[1][2] * v.z() + m[1][3] * v.w();
        r[2] = m[2][0] * v.x() + m[2][1] * v.y() + m[2][2] * v.z() + m[2][3] * v.w();
        r[3] = m[3][0] * v.x() + m[3][1] * v.y() + m[3][2] * v.z() + m[3][3] * v.w();
        return r;
    }
};
#endif */