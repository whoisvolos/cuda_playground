#include "math.cuh"

math::triangle_t::triangle_t(float3& a, float3& b, float3& c) : a(a), b(b), c(c) { }

math::triangle_t::triangle_t(float3* arr) : a(*arr), b(*(arr + 1)), c(*(arr + 2)) { }

float math::triangle_t::area() const {
    return length(cross(b - a, c - a)) / 2;
}

float3 math::triangle_t::center() const {
    return (a + b + c) / 3;
}