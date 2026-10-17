
typedef unsigned int uint;

struct bp_int3;
struct bp_float2 {
    float x, y;
    bp_float2() = default;
    bp_float2(float a, float b) : x(a), y(b) {}
};
struct bp_float3 {
    float x, y, z;
    bp_float3() = default;
    explicit bp_float3(float s) : x(s), y(s), z(s) {}
    bp_float3(float a, float b, float c) : x(a), y(b), z(c) {}
    explicit bp_float3(const bp_int3& v);
};
struct bp_int3 {
    int x, y, z;
    bp_int3() = default;
    bp_int3(int a, int b, int c) : x(a), y(b), z(c) {}
    explicit bp_int3(const bp_float3& v) : x(int(v.x)), y(int(v.y)), z(int(v.z)) {}
};
inline bp_float3::bp_float3(const bp_int3& v) : x(float(v.x)), y(float(v.y)), z(float(v.z)) {}
struct bp_uint2 {
    uint x, y;
    bp_uint2() = default;
    bp_uint2(uint a, uint b) : x(a), y(b) {}
};
struct bp_uint4 {
    uint x, y, z, w;
    bp_uint4() = default;
    bp_uint4(uint a, uint b, uint c, uint d) : x(a), y(b), z(c), w(d) {}
};

inline bp_float3 operator+(bp_float3 a, bp_float3 b) { return bp_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline bp_float3 operator-(bp_float3 a, bp_float3 b) { return bp_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline bp_float3 operator-(bp_float3 a, float s) { return bp_float3(a.x - s, a.y - s, a.z - s); }
inline bp_float3 operator-(bp_float3 a) { return bp_float3(-a.x, -a.y, -a.z); }
inline bp_float3 operator*(bp_float3 a, float s) { return bp_float3(a.x * s, a.y * s, a.z * s); }
inline bp_float3 operator*(float s, bp_float3 a) { return bp_float3(s * a.x, s * a.y, s * a.z); }
inline bp_float3 operator/(bp_float3 a, bp_float3 b) { return bp_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
inline bp_float3 operator/(bp_float3 a, float s) { return bp_float3(a.x / s, a.y / s, a.z / s); }
inline bp_float3& operator+=(bp_float3& a, bp_float3 b) { a.x += b.x, a.y += b.y, a.z += b.z; return a; }
inline bp_float3& operator-=(bp_float3& a, bp_float3 b) { a.x -= b.x, a.y -= b.y, a.z -= b.z; return a; }
inline float dot(bp_float3 a, bp_float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(bp_float3 a) { return sqrtf(dot(a, a)); }
inline bp_float3 floor(bp_float3 a) { return bp_float3(floorf(a.x), floorf(a.y), floorf(a.z)); }
inline bp_float3 min(bp_float3 a, bp_float3 b) { return bp_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
inline bp_float3 max(bp_float3 a, bp_float3 b) { return bp_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
inline bp_int3 operator+(bp_int3 a, int s) { return bp_int3(a.x + s, a.y + s, a.z + s); }
inline bp_int3 min(bp_int3 a, bp_int3 b) { return bp_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
inline bp_uint2& operator+=(bp_uint2& a, bp_uint2 b) { a.x += b.x, a.y += b.y; return a; }
inline float clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
