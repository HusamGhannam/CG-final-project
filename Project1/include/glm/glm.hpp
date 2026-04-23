#pragma once

#include <cmath>

namespace glm {

struct vec3 {
    float x, y, z;
    vec3() : x(0), y(0), z(0) {}
    vec3(float a, float b, float c) : x(a), y(b), z(c) {}
};

struct mat4 {
    // Column-major storage to match OpenGL/GLM
    float data[16];
    mat4() {
        for (int i = 0; i < 16; ++i) data[i] = 0.0f;
    }
    explicit mat4(float diag) {
        for (int i = 0; i < 16; ++i) data[i] = 0.0f;
        data[0] = diag; data[5] = diag; data[10] = diag; data[15] = diag;
    }
};

inline float radians(float deg) { return deg * 3.14159265358979323846f / 180.0f; }

// Multiply column-major matrices: result = a * b
inline mat4 operator*(const mat4& a, const mat4& b) {
    mat4 r;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                // a(row,i) is a.data[i*4 + row]; b(i,col) is b.data[col*4 + i]
                sum += a.data[i*4 + row] * b.data[col*4 + i];
            }
            r.data[col*4 + row] = sum;
        }
    }
    return r;
}

} // namespace glm
