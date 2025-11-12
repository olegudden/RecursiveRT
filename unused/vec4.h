/* #ifndef VEC4_H
#define VEC4_H

class vec4
{
public:
    float e[4];
    vec4() : e{0, 0, 0, 0} {}
    vec4(float e0, float e1, float e2, float e3) : e{e0, e1, e2, e3} {}
    float x() const { return e[0]; }
    float y() const { return e[1]; }
    float z() const { return e[2]; }
    float w() const { return e[3]; }

    float operator[](int i) const { return e[i]; }
    float &operator[](int i) { return e[i]; }

    vec3 xyz() const { return vec3(x(), y(), z()); } // conversion to vec3
};
#endif */