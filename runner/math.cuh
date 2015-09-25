#include <cuda_runtime.h>
#include <helper_math.h>

namespace math {
    
    struct triangle_t {
        float3 a;
        float3 b;
        float3 c;

        triangle_t(float3& a, float3& b, float3& c);
        triangle_t(float3* arr);

        __host__ __device__ float area() const;
        __host__ __device__ float3 center() const;
    };

    struct ray_t {
        float3 origin;
        float3 direction;
    };
}