// Fused PatchmatchNet kernels (exp 011, max path). Batch 1. Layouts:
//   features NHWC fp16: ref [HW, C], src [S, HW, C];  depth hypotheses [D, HW] fp32;  costs [D, HW] fp32.
//   Neighbour coordinates [N, HW] fp32 are grid_sample-unnormalized (align_corners=False, border-clipped) pixel coords.
// MLP weights (1x1x1 Conv3d + folded BN): w0 [16, G], b0 [16], w1 [8, 16], b1 [8], w2 [8], b2 [1] -> packed fp32.
#include <cuda_fp16.h>
#include <stdint.h>

#define MLP_SIZE(G) (16 * (G) + 16 + 128 + 8 + 8 + 1)

template <int G>
__device__ __forceinline__ float mlp(const float* __restrict__ w, const float* sim) {
    const float* w0 = w;
    const float* b0 = w0 + 16 * G;
    const float* w1 = b0 + 16;
    const float* b1 = w1 + 128;
    const float* w2 = b1 + 8;
    const float* b2 = w2 + 8;
    float h0[16];
#pragma unroll
    for (int o = 0; o < 16; o++) {
        float a = b0[o];
#pragma unroll
        for (int i = 0; i < G; i++) a = fmaf(w0[o * G + i], sim[i], a);
        h0[o] = fmaxf(a, 0.f);
    }
    float out = b2[0];
#pragma unroll
    for (int o = 0; o < 8; o++) {
        float a = b1[o];
#pragma unroll
        for (int i = 0; i < 16; i++) a = fmaf(w1[o * 16 + i], h0[i], a);
        out = fmaf(w2[o], fmaxf(a, 0.f), out);
    }
    return out;
}

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

// dot of 8 fp16 channels (one uint4) with 8 fp32 ref values
__device__ __forceinline__ float dot8(uint4 v, const float* r) {
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    float a = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        float2 f = __half22float2(h[k]);
        a = fmaf(f.x, r[2 * k], a);
        a = fmaf(f.y, r[2 * k + 1], a);
    }
    return a;
}

// per-group correlation of a bilinear sample (zeros padding) with the reference vector
template <int C, int G>
__device__ __forceinline__ void corr_sample(const __half* __restrict__ f, int H, int W, float x, float y,
                                            const float* ref, float* sim) {
    constexpr int CG = C / G;           // channels per group (4 or 8)
    constexpr int NV = C / 8;           // uint4 vectors per pixel
#pragma unroll
    for (int g = 0; g < G; g++) sim[g] = 0.f;
    float x0f = floorf(x), y0f = floorf(y);
    int x0 = (int)x0f, y0 = (int)y0f;
    float ax = x - x0f, ay = y - y0f;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        int xx = x0 + (c & 1), yy = y0 + (c >> 1);
        float wgt = ((c & 1) ? ax : 1.f - ax) * ((c >> 1) ? ay : 1.f - ay);
        if (xx < 0 || yy < 0 || xx >= W || yy >= H || wgt == 0.f) continue;
        const uint4* p = reinterpret_cast<const uint4*>(f + ((size_t)yy * W + xx) * C);
#pragma unroll
        for (int v = 0; v < NV; v++) {
            uint4 q = __ldg(p + v);
            if (CG == 8) {
                sim[v] = fmaf(wgt, dot8(q, ref + 8 * v), sim[v]);
            } else {  // CG == 4: one uint4 spans two groups
                const __half2* h = reinterpret_cast<const __half2*>(&q);
                float2 f0 = __half22float2(h[0]), f1 = __half22float2(h[1]);
                float2 f2 = __half22float2(h[2]), f3 = __half22float2(h[3]);
                const float* r = ref + 8 * v;
                float s0 = f0.x * r[0] + f0.y * r[1] + f1.x * r[2] + f1.y * r[3];
                float s1 = f2.x * r[4] + f2.y * r[5] + f3.x * r[6] + f3.y * r[7];
                sim[2 * v] = fmaf(wgt, s0, sim[2 * v]);
                sim[2 * v + 1] = fmaf(wgt, s1, sim[2 * v + 1]);
            }
        }
    }
#pragma unroll
    for (int g = 0; g < G; g++) sim[g] *= (1.f / CG);
}

// projected source pixel (x, y) for ref pixel (px, py) at depth d. proj: [S, 12] = rot (row-major 3x3), trans (3)
__device__ __forceinline__ void project(const float* __restrict__ P, float px, float py, float d, int H, int W,
                                        float& x, float& y) {
    // same op order as upstream: (rot @ xyz) * depth + trans
    float rx = P[0] * px + P[1] * py + P[2];
    float ry = P[3] * px + P[4] * py + P[5];
    float rz = P[6] * px + P[7] * py + P[8];
    float X = rx * d + P[9], Y = ry * d + P[10], Z = rz * d + P[11];
    if (Z <= 1e-3f) { X = (float)W; Y = (float)H; Z = 1.f; }
    x = X / Z;
    y = Y / Z;
}

// view weights: vw[s, p] = max_d sigmoid(pixelwise_net(sim_s(p, d))). grid: (ceil(HW/128), D, S). vw must be zeroed.
template <int C, int G>
__global__ void __launch_bounds__(128) view_weight_k(const __half* __restrict__ ref, const __half* __restrict__ src,
                                                     const float* __restrict__ proj, const float* __restrict__ depth,
                                                     const float* __restrict__ wpw, float* __restrict__ vw,
                                                     int H, int W) {
    int HW = H * W;
    int p = blockIdx.x * 128 + threadIdx.x;
    int d = blockIdx.y, s = blockIdx.z;
    if (p >= HW) return;
    float r[C];
    const uint4* rp = reinterpret_cast<const uint4*>(ref + (size_t)p * C);
#pragma unroll
    for (int v = 0; v < C / 8; v++) {
        uint4 q = __ldg(rp + v);
        const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
        for (int k = 0; k < 4; k++) { float2 f = __half22float2(h[k]); r[8 * v + 2 * k] = f.x; r[8 * v + 2 * k + 1] = f.y; }
    }
    float x, y, sim[G];
    project(proj + 12 * s, (float)(p % W), (float)(p / W), depth[(size_t)d * HW + p], H, W, x, y);
    corr_sample<C, G>(src + (size_t)s * HW * C, H, W, x, y, r, sim);
    float o = sigm(mlp<G>(wpw, sim));
    atomicMax(reinterpret_cast<int*>(vw + (size_t)s * HW + p), __float_as_int(o));
}

// matching cost: cost[d, p] = similarity_net( sum_s vw_s * sim_s / (1e-5 + sum_s vw_s) ). grid: (ceil(HW/128), D)
template <int C, int G>
__global__ void __launch_bounds__(128) cost_k(const __half* __restrict__ ref, const __half* __restrict__ src,
                                              const float* __restrict__ proj, const float* __restrict__ depth,
                                              const float* __restrict__ vw, const float* __restrict__ wsim,
                                              float* __restrict__ cost, int S, int H, int W) {
    int HW = H * W;
    int p = blockIdx.x * 128 + threadIdx.x;
    int d = blockIdx.y;
    if (p >= HW) return;
    float r[C];
    const uint4* rp = reinterpret_cast<const uint4*>(ref + (size_t)p * C);
#pragma unroll
    for (int v = 0; v < C / 8; v++) {
        uint4 q = __ldg(rp + v);
        const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
        for (int k = 0; k < 4; k++) { float2 f = __half22float2(h[k]); r[8 * v + 2 * k] = f.x; r[8 * v + 2 * k + 1] = f.y; }
    }
    float dep = depth[(size_t)d * HW + p];
    float px = (float)(p % W), py = (float)(p / W);
    float acc[G], sim[G];
#pragma unroll
    for (int g = 0; g < G; g++) acc[g] = 0.f;
    float wsum = 1e-5f;
    for (int s = 0; s < S; s++) {
        float x, y;
        project(proj + 12 * s, px, py, dep, H, W, x, y);
        corr_sample<C, G>(src + (size_t)s * HW * C, H, W, x, y, r, sim);
        float w = vw[(size_t)s * HW + p];
#pragma unroll
        for (int g = 0; g < G; g++) acc[g] = fmaf(sim[g], w, acc[g]);
        wsum += w;
    }
#pragma unroll
    for (int g = 0; g < G; g++) acc[g] = acc[g] / wsum;
    cost[(size_t)d * HW + p] = mlp<G>(wsim, acc);
}

__device__ __forceinline__ void bil_coef(float x, float y, int W, int& i00, int& dx, int& dy, float& ax, float& ay) {
    // border-clipped coords: x in [0, W-1], y in [0, H-1]
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    ax = x - x0;
    ay = y - y0;
    dx = (ax > 0.f) ? 1 : 0;   // x0 + 1 exists whenever its weight is non-zero
    dy = (ay > 0.f) ? W : 0;
    i00 = y0 * W + x0;
}

__device__ __forceinline__ float bil(const float* __restrict__ m, int i00, int dx, int dy, float ax, float ay) {
    float v00 = m[i00], v01 = m[i00 + dx], v10 = m[i00 + dy], v11 = m[i00 + dy + dx];
    return (v00 * (1.f - ax) + v01 * ax) * (1.f - ay) + (v10 * (1.f - ax) + v11 * ax) * ay;
}

// adaptive spatial cost aggregation + softmax + depth regression (+ photometric confidence).
// nx, ny: [NB, HW] neighbour coords; fw: [NB, HW] feature weights. One thread per pixel.
template <int NB>
__global__ void __launch_bounds__(128) aggregate_k(const float* __restrict__ cost, const float* __restrict__ depth,
                                                   const float* __restrict__ nx, const float* __restrict__ ny,
                                                   const float* __restrict__ fw, float* __restrict__ out_depth,
                                                   float* __restrict__ out_conf, float inv_max, float inv_range,
                                                   float inv_scale, int D, int H, int W, int inverse) {
    int HW = H * W;
    int p = blockIdx.x * 128 + threadIdx.x;
    if (p >= HW) return;
    int i00[NB], dx[NB], dy[NB];
    float ax[NB], ay[NB], fwn[NB];
#pragma unroll
    for (int n = 0; n < NB; n++) {
        bil_coef(nx[(size_t)n * HW + p], ny[(size_t)n * HW + p], W, i00[n], dx[n], dy[n], ax[n], ay[n]);
        fwn[n] = fw[(size_t)n * HW + p];
    }
    float m = -INFINITY, ssum = 0.f, acc = 0.f;
    float probs_raw[64];   // only used for the confidence (D <= 64)
    for (int d = 0; d < D; d++) {
        const float* dm = depth + (size_t)d * HW;
        const float* cm = cost + (size_t)d * HW;
        float xc = (1.f / dm[p] - inv_max) / inv_range;
        float wts[NB], wsum = 0.f;
        float c = 0.f;
#pragma unroll
        for (int n = 0; n < NB; n++) {
            // normalized inverse depth at the 4 corners, then bilinear (as grid_sample of the normalized map)
            float v00 = (1.f / dm[i00[n]] - inv_max) / inv_range, v01 = (1.f / dm[i00[n] + dx[n]] - inv_max) / inv_range;
            float v10 = (1.f / dm[i00[n] + dy[n]] - inv_max) / inv_range, v11 = (1.f / dm[i00[n] + dy[n] + dx[n]] - inv_max) / inv_range;
            float x1 = (v00 * (1.f - ax[n]) + v01 * ax[n]) * (1.f - ay[n]) + (v10 * (1.f - ax[n]) + v11 * ax[n]) * ay[n];
            float t = fminf(fmaxf(fabsf(x1 - xc) * inv_scale, 0.f), 4.f);
            float w = (1.f / (1.f + expf(-(4.f - 2.f * t)))) * fwn[n];
            wts[n] = w;
            wsum += w;
        }
#pragma unroll
        for (int n = 0; n < NB; n++) c += bil(cm, i00[n], dx[n], dy[n], ax[n], ay[n]) * (wts[n] / wsum);
        if (out_conf) probs_raw[d & 63] = c;
        float val = inverse ? (float)d : dm[p];
        if (c > m) {
            float sc = expf(m - c);
            ssum = ssum * sc + 1.f;
            acc = acc * sc + val;
            m = c;
        } else {
            float e = expf(c - m);
            ssum += e;
            acc = fmaf(e, val, acc);
        }
    }
    float e = acc / ssum;
    if (inverse) {
        float imin = 1.f / depth[(size_t)(D - 1) * HW + p];
        float imax = 1.f / depth[p];
        out_depth[p] = 1.f / (imax + e / (D - 1) * (imin - imax));
    } else {
        out_depth[p] = e;
    }
    if (out_conf) {
        // confidence = sum of probabilities at depth_index-1 .. depth_index+2, depth_index = long(sum_d d * p_d)
        float idx = 0.f;
        for (int d = 0; d < D; d++) idx += (float)d * (expf(probs_raw[d] - m) / ssum);
        int k = min(max((int)idx, 0), D - 1);
        float cf = 0.f;
        for (int j = k - 1; j <= k + 2; j++)
            if (j >= 0 && j < D) cf += expf(probs_raw[j] - m) / ssum;
        out_conf[p] = cf;
    }
}

// feature weights: fw[n, p] = sigmoid(net(groupcorr(bilinear_border(ref, nb_n(p)), ref(p)))). grid (ceil(HW/128), NB)
template <int C, int G>
__global__ void __launch_bounds__(128) feat_weight_k(const __half* __restrict__ ref, const float* __restrict__ nx,
                                                     const float* __restrict__ ny, const float* __restrict__ wfw,
                                                     float* __restrict__ fw, int H, int W) {
    int HW = H * W;
    int p = blockIdx.x * 128 + threadIdx.x;
    int n = blockIdx.y;
    if (p >= HW) return;
    float r[C];
    const uint4* rp = reinterpret_cast<const uint4*>(ref + (size_t)p * C);
#pragma unroll
    for (int v = 0; v < C / 8; v++) {
        uint4 q = __ldg(rp + v);
        const __half2* h = reinterpret_cast<const __half2*>(&q);
#pragma unroll
        for (int k = 0; k < 4; k++) { float2 f = __half22float2(h[k]); r[8 * v + 2 * k] = f.x; r[8 * v + 2 * k + 1] = f.y; }
    }
    float sim[G];
    // border-clipped coords are always in range -> zero-padding sampler equals border sampler here
    corr_sample<C, G>(ref, H, W, nx[(size_t)n * HW + p], ny[(size_t)n * HW + p], r, sim);
    fw[(size_t)n * HW + p] = sigm(mlp<G>(wfw, sim));
}

// hypotheses: base samples (perturbation around prev depth, or given random samples) + propagation, sorted ascending.
// prev: [HW] (perturb mode) ; base: [Db, HW] (given mode, prev == nullptr). nx, ny: [P, HW]. out: [Db + P, HW]
template <int MAXD>
__global__ void __launch_bounds__(128) hypo_k(const float* __restrict__ prev, const float* __restrict__ base,
                                              const float* __restrict__ nx, const float* __restrict__ ny,
                                              float* __restrict__ out, int Db, int P, float inv_min, float inv_max,
                                              float inv_interval, int H, int W) {
    int HW = H * W;
    int p = blockIdx.x * 128 + threadIdx.x;
    if (p >= HW) return;
    float v[MAXD];
    int n = 0;
    if (prev) {
        float id = 1.f / prev[p];
        for (int k = -Db / 2; k < Db / 2; k++) {
            float s = id + inv_interval * (float)k;
            s = fminf(fmaxf(s, inv_max), inv_min);
            v[n++] = 1.f / s;
        }
    } else {
        for (int k = 0; k < Db; k++) v[n++] = base[(size_t)k * HW + p];
    }
    if (P > 0) {
        // propagation samples the middle base hypothesis map. In perturb mode that map is a function of prev:
        // 1 / clamp(1 / prev) -> sample it at 4 corners.
        for (int q = 0; q < P; q++) {
            int i00, dx, dy;
            float ax, ay;
            bil_coef(nx[(size_t)q * HW + p], ny[(size_t)q * HW + p], W, i00, dx, dy, ax, ay);
            float c[4];
            int idx[4] = {i00, i00 + dx, i00 + dy, i00 + dy + dx};
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float s;
                if (prev) {
                    s = 1.f / fminf(fmaxf(1.f / prev[idx[j]], inv_max), inv_min);
                } else {
                    s = base[(size_t)(Db / 2) * HW + idx[j]];
                }
                c[j] = s;
            }
            v[n++] = (c[0] * (1.f - ax) + c[1] * ax) * (1.f - ay) + (c[2] * (1.f - ax) + c[3] * ax) * ay;
        }
    }
    // insertion sort (n <= MAXD)
    for (int i = 1; i < n; i++) {
        float key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; j--; }
        v[j + 1] = key;
    }
    for (int i = 0; i < n; i++) out[(size_t)i * HW + p] = v[i];
}

// ---------------------------------------------------------------- launchers (pybind, torch tensors)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CHK(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be a contiguous CUDA tensor")
static inline dim3 g1(int HW, int y = 1, int z = 1) { return dim3((HW + 127) / 128, y, z); }
#define HP(t) reinterpret_cast<const __half*>((t).data_ptr<at::Half>())
#define FP(t) (t).data_ptr<float>()

// ref [HW,C] half, src [S,HW,C] half, proj [S,12], depth [D,HW], w (mlp), vw [S,HW] (zeroed)
void view_weight(torch::Tensor ref, torch::Tensor src, torch::Tensor proj, torch::Tensor depth, torch::Tensor w,
                 torch::Tensor vw, int64_t H, int64_t W, int64_t G) {
    CHK(ref); CHK(src); CHK(proj); CHK(depth); CHK(w); CHK(vw);
    int C = ref.size(-1), D = depth.size(0), S = src.size(0), HW = H * W;
    auto st = at::cuda::getCurrentCUDAStream();
#define VW(CC, GG) if (C == CC && G == GG) { view_weight_k<CC, GG><<<g1(HW, D, S), 128, 0, st>>>(HP(ref), HP(src), FP(proj), FP(depth), FP(w), FP(vw), H, W); return; }
    VW(16, 4) VW(32, 8) VW(64, 8)
    TORCH_CHECK(false, "view_weight: unsupported C/G");
}

void cost(torch::Tensor ref, torch::Tensor src, torch::Tensor proj, torch::Tensor depth, torch::Tensor vw,
          torch::Tensor w, torch::Tensor out, int64_t H, int64_t W, int64_t G) {
    CHK(ref); CHK(src); CHK(proj); CHK(depth); CHK(w); CHK(vw); CHK(out);
    int C = ref.size(-1), D = depth.size(0), S = src.size(0), HW = H * W;
    auto st = at::cuda::getCurrentCUDAStream();
#define CO(CC, GG) if (C == CC && G == GG) { cost_k<CC, GG><<<g1(HW, D), 128, 0, st>>>(HP(ref), HP(src), FP(proj), FP(depth), FP(vw), FP(w), FP(out), S, H, W); return; }
    CO(16, 4) CO(32, 8) CO(64, 8)
    TORCH_CHECK(false, "cost: unsupported C/G");
}

void feat_weight(torch::Tensor ref, torch::Tensor nx, torch::Tensor ny, torch::Tensor w, torch::Tensor fw,
                 int64_t H, int64_t W, int64_t G) {
    CHK(ref); CHK(nx); CHK(ny); CHK(w); CHK(fw);
    int C = ref.size(-1), NB = nx.size(0), HW = H * W;
    auto st = at::cuda::getCurrentCUDAStream();
#define FW(CC, GG) if (C == CC && G == GG) { feat_weight_k<CC, GG><<<g1(HW, NB), 128, 0, st>>>(HP(ref), FP(nx), FP(ny), FP(w), FP(fw), H, W); return; }
    FW(16, 4) FW(32, 8) FW(64, 8)
    TORCH_CHECK(false, "feat_weight: unsupported C/G");
}

void aggregate(torch::Tensor cost, torch::Tensor depth, torch::Tensor nx, torch::Tensor ny, torch::Tensor fw,
               torch::Tensor out_depth, c10::optional<torch::Tensor> out_conf, double inv_max, double inv_range,
               double inv_scale, int64_t H, int64_t W, bool inverse) {
    CHK(cost); CHK(depth); CHK(nx); CHK(ny); CHK(fw); CHK(out_depth);
    int D = depth.size(0), HW = H * W;
    TORCH_CHECK(nx.size(0) == 9 && D <= 64, "aggregate: 9 neighbours, D <= 64");
    float* oc = out_conf.has_value() ? FP(out_conf.value()) : nullptr;
    aggregate_k<9><<<g1(HW), 128, 0, at::cuda::getCurrentCUDAStream()>>>(FP(cost), FP(depth), FP(nx), FP(ny), FP(fw),
        FP(out_depth), oc, (float)inv_max, (float)inv_range, (float)inv_scale, D, H, W, inverse ? 1 : 0);
}

void hypo(c10::optional<torch::Tensor> prev, c10::optional<torch::Tensor> base, torch::Tensor nx, torch::Tensor ny,
          torch::Tensor out, int64_t Db, int64_t P, double inv_min, double inv_max, double inv_interval, int64_t H, int64_t W) {
    int HW = H * W;
    TORCH_CHECK(Db + P <= 64, "hypo: at most 64 hypotheses");
    const float* pp = prev.has_value() ? FP(prev.value()) : nullptr;
    const float* bp = base.has_value() ? FP(base.value()) : nullptr;
    hypo_k<64><<<g1(HW), 128, 0, at::cuda::getCurrentCUDAStream()>>>(pp, bp, P ? FP(nx) : nullptr, P ? FP(ny) : nullptr,
        FP(out), Db, P, (float)inv_min, (float)inv_max, (float)inv_interval, H, W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("view_weight", &view_weight);
    m.def("cost", &cost);
    m.def("feat_weight", &feat_weight);
    m.def("aggregate", &aggregate);
    m.def("hypo", &hypo);
}
