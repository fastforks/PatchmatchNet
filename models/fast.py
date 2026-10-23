"""Fast PatchmatchNet inference (exp 011). Same weights, same outputs as models.net.PatchmatchNet (eval mode).

FastPatchmatchNet(state_dict, mode, **patchmatch_args)(images [1,N,3,H,W] in [0,1], intrinsics [1,N,3,3],
extrinsics [1,N,4,4], depth_min [1], depth_max [1]) -> (depth [1,H,W], photometric confidence [1,H,W]).
Batch 1 (the per-view unit of MVS inference).

mode "friendly": plain PyTorch (ONNX-exportable ops): BN folded into convs, fp16 channels-last CNNs (feature net for all
  views in one batch, refinement with an fp32 depth path), sources warped and correlated in fp32 per view, 1x1x1 Conv3d
  nets as 1x1 Conv2d, no score/view-weight copies that are not used.
mode "max": friendly CNNs + fused CUDA kernels (models/pmn_kernels.cu): warp + group correlation + view weighting +
  similarity net in one pass per (pixel, hypothesis), never materializing warped features; adaptive aggregation +
  depth weights + softmax + regression + confidence in one pass; hypothesis perturbation + propagation + sort in one.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from .net import PatchmatchNet

_HERE = Path(__file__).resolve().parent
_EXT = None


def load_ext():
    """Prebuilt extension named by the source hash (built once by build_ext(); no compile at runtime if shipped)."""
    global _EXT
    if _EXT is not None:
        return _EXT
    src = _HERE / "pmn_kernels.cu"
    tag = hashlib.sha1(src.read_bytes()).hexdigest()[:12]
    name = f"pmn_kernels_{tag}"
    d = _HERE / "prebuilt" / name
    so = sorted(d.glob(f"{name}*.so")) if d.exists() else []
    if not so:
        build_ext()
        so = sorted(d.glob(f"{name}*.so"))
    spec = importlib.util.spec_from_file_location(name, so[0])
    _EXT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_EXT)
    return _EXT


def build_ext():
    from torch.utils.cpp_extension import load
    src = _HERE / "pmn_kernels.cu"
    tag = hashlib.sha1(src.read_bytes()).hexdigest()[:12]
    name = f"pmn_kernels_{tag}"
    d = _HERE / "prebuilt" / name
    d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0;12.0+PTX")
    load(name=name, sources=[str(src)], build_directory=str(d), extra_cuda_cflags=["-O3", "--use_fast_math"],
         verbose=False)


def _fold(conv_w, bn, conv_b=None, transposed=False):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = bn.bias - bn.running_mean * s
    if conv_b is not None:
        b = b + conv_b * s
    if transposed:
        return conv_w * s.view(1, -1, 1, 1), b
    return conv_w * s.view(-1, *([1] * (conv_w.dim() - 1))), b


def _mlp_pack(m, last):
    """ConvBnReLU3D(G,16) -> ConvBnReLU3D(16,8) -> Conv3d(8,1): packed fp32 [w0,b0,w1,b1,w2,b2] + 2D conv weights."""
    w0, b0 = _fold(m.conv0.conv.weight.flatten(1), m.conv0.bn)
    w1, b1 = _fold(m.conv1.conv.weight.flatten(1), m.conv1.bn)
    w2, b2 = last.weight.flatten(), last.bias.flatten()
    packed = torch.cat([w0.flatten(), b0, w1.flatten(), b1, w2, b2]).float().contiguous()
    convs = [(w0[:, :, None, None].float(), b0.float()), (w1[:, :, None, None].float(), b1.float()),
             (w2.view(1, 8, 1, 1).float(), b2.float())]
    return packed, convs


def _mlp2d(x, convs):
    """x [B, G, D, H, W] -> [B, D, H, W] through the 1x1x1 nets as 1x1 Conv2d over a [B, G, D*H, W] view."""
    B, G, D, H, W = x.shape
    y = x.reshape(B, G, D * H, W)
    y = F.relu(F.conv2d(y, *convs[0]))
    y = F.relu(F.conv2d(y, *convs[1]))
    y = F.conv2d(y, *convs[2])
    return y.view(B, D, H, W)


class FastPatchmatchNet(torch.nn.Module):
    def __init__(self, state_dict, mode="max", device="cuda", hacc=False, **args):
        super().__init__()
        self.mode = mode
        self.hacc = hacc
        ref = PatchmatchNet(**args)
        ref.load_state_dict(state_dict)
        ref.eval()
        self.args = args
        self.dev = torch.device(device)
        hdt = torch.float16
        with torch.no_grad():
            f = ref.feature
            self.feat = []
            for name, stride, pad in [("conv0", 1, 1), ("conv1", 1, 1), ("conv2", 2, 2), ("conv3", 1, 1), ("conv4", 1, 1),
                                      ("conv5", 2, 2), ("conv6", 1, 1), ("conv7", 1, 1), ("conv8", 2, 2), ("conv9", 1, 1),
                                      ("conv10", 1, 1)]:
                c = getattr(f, name)
                w, b = _fold(c.conv.weight, c.bn)
                self.feat.append((self._w(w), b.to(device, hdt), stride, pad))
            # FPN rewritten with linearity (1x1 convs and bilinear x2 upsampling commute; bilinear weights sum to 1, so
            # biases pass through): all output heads at the lowest channel count, upsampling 48 / 16 instead of 64 channels
            o1, o2, o3 = (getattr(f, f"output{i}").weight.double()[:, :, 0, 0] for i in (1, 2, 3))
            i1w, i1b = f.inner1.weight.double()[:, :, 0, 0], f.inner1.bias.double()
            i2w, i2b = f.inner2.weight.double()[:, :, 0, 0], f.inner2.bias.double()
            self.c_out2 = o2.shape[0]
            self.heads = self._w(torch.cat([o1, o2, o3])[:, :, None, None].float())
            self.c7conv = (self._w(torch.cat([o2 @ i1w, o3 @ i1w])[:, :, None, None].float()),
                           torch.cat([o2 @ i1b, o3 @ i1b]).to(device, hdt))
            self.c4conv = (self._w((o3 @ i2w)[:, :, None, None].float()), (o3 @ i2b).to(device, hdt))
            self.c_out1 = o1.shape[0]
            r = ref.upsample_net
            self.r_conv0 = _fold(r.conv0.conv.weight, r.conv0.bn)
            self.r_conv1 = _fold(r.conv1.conv.weight, r.conv1.bn)
            self.r_conv2 = _fold(r.conv2.conv.weight, r.conv2.bn)
            self.r_deconv = _fold(r.deconv.weight, r.bn, transposed=True)
            self.r_conv3 = _fold(r.conv3.conv.weight, r.conv3.bn)
            self.r_res = r.res.weight.detach()
            for k in ["r_conv0", "r_conv1", "r_conv2", "r_deconv", "r_conv3"]:
                w, b = getattr(self, k)
                setattr(self, k, (self._w(w), b.to(device, hdt)))
            self.r_res = self._w(self.r_res)
            self.pm = []
            for i in (1, 2, 3):
                p = getattr(ref, f"patchmatch_{i}")
                e = p.evaluation
                st = dict(
                    G=p.G, dil=p.dilation, iters=p.patchmatch_iteration, scale=p.patchmatch_interval_scale,
                    nsample=p.depth_initialization.patchmatch_num_sample, P=p.propagate_neighbors,
                    NE=p.evaluate_neighbors, stage=i,
                    propa=(self._w(p.propa_conv.weight), p.propa_conv.bias.to(device, hdt)),
                    evalc=(self._w(p.eval_conv.weight), p.eval_conv.bias.to(device, hdt)),
                    grid_fn=p.get_grid, grid_type=p.grid_type,
                )
                st["pw"], st["pw2d"] = _mlp_pack(e.pixel_wise_net, e.pixel_wise_net.conv2)
                st["sim"], st["sim2d"] = _mlp_pack(e.similarity_net, e.similarity_net.similarity)
                st["fw"], st["fw2d"] = _mlp_pack(p.feature_weight_net, p.feature_weight_net.similarity)
                for k in ("pw", "sim", "fw"):
                    st[k] = st[k].to(device)
                    st[k + "2d"] = [(w.to(device), b.to(device)) for w, b in st[k + "2d"]]
                self.pm.append(st)
        if mode == "max":
            load_ext()
            if hacc:
                import fastbb.cudnn_conv as cc
                self._cc = cc

    def _w(self, w):
        w = w.detach().to(self.dev, torch.float16)
        return w.contiguous(memory_format=torch.channels_last) if w.dim() == 4 else w

    def _conv(self, x, w, b, stride=1, pad=0, dil=1):
        if self.hacc and dil == 1 and w.shape[0] % 8 == 0 and x.shape[1] % 8 == 0:
            return self._cc.conv_half_acc(x, w, b, (pad, pad), (stride, stride))
        return F.conv2d(x, w, b, stride, pad, dil)

    # ------------------------------------------------------------------ CNNs (fp16, channels-last)
    def _cbr(self, x, w, b, stride=1, pad=0):
        """conv + bias + relu; max: one fused cuDNN kernel (bias/relu epilogue), friendly: plain ops."""
        if self.mode == "max" and not self.hacc:
            return torch.cudnn_convolution_relu(x, w, b, (stride, stride), (pad, pad), (1, 1), 1)
        return F.relu(self._conv(x, w, b, stride, pad))

    def features(self, x):
        cl = torch.channels_last
        for i, (w, b, s, p) in enumerate(self.feat):
            x = self._cbr(x, w, b, s, p)
            if i == 1:
                c1 = x
            elif i == 4:
                c4 = x
            elif i == 7:
                c7 = x
        h = F.conv2d(x, self.heads)
        a = self.c_out1
        t = F.interpolate(h[:, a:], scale_factor=2.0, mode="bilinear", align_corners=False) + F.conv2d(c7, *self.c7conv)
        b = self.c_out2
        o1 = F.interpolate(t[:, b:], scale_factor=2.0, mode="bilinear", align_corners=False) + F.conv2d(c4, *self.c4conv)
        return {3: h[:, :a].contiguous(memory_format=cl), 2: t[:, :b].contiguous(memory_format=cl),
                1: o1.contiguous(memory_format=cl)}

    def refine(self, img, depth0, dmin, dmax):
        depth = (depth0 - dmin.view(1, 1, 1, 1)) / (dmax - dmin).view(1, 1, 1, 1)
        c0 = self._cbr(img, *self.r_conv0, 1, 1)
        d = depth.half().contiguous(memory_format=torch.channels_last)
        d = F.relu(F.conv2d(d, *self.r_conv1, 1, 1))
        d = self._cbr(d, *self.r_conv2, 1, 1)
        d = F.relu(F.conv_transpose2d(d, self.r_deconv[0], self.r_deconv[1], 2, 1, 1))
        x = torch.cat((d, c0), 1)
        res = F.conv2d(self._cbr(x, *self.r_conv3, 1, 1), self.r_res, None, 1, 1)
        depth = F.interpolate(depth, scale_factor=2.0, mode="nearest") + res.float()
        return depth * (dmax - dmin).view(1, 1, 1, 1) + dmin.view(1, 1, 1, 1)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _proj(K, E, scale):
        """[S,12] rot|trans of src_proj @ inv(ref_proj) per source, exactly as upstream (fp32)."""
        Kl = K.clone()
        Kl[:, :, :2] *= scale
        proj = E.clone()
        proj[:, :, :3, :4] = torch.matmul(Kl, E[:, :, :3, :4])
        proj = proj[0]
        rel = torch.matmul(proj[1:], torch.inverse(proj[0:1]))
        return torch.cat([rel[:, :3, :3].reshape(-1, 9), rel[:, :3, 3]], 1).contiguous()

    @staticmethod
    def _unnorm(grid, h, w):
        """upstream grid [1, N*h, w, 2] -> grid_sample pixel coords (align_corners=False, border clip): nx, ny [N, h*w]"""
        g = grid.view(-1, h, w, 2)
        nx = (((g[..., 0] + 1) * w - 1) / 2).clamp(0, w - 1)
        ny = (((g[..., 1] + 1) * h - 1) / 2).clamp(0, h - 1)
        return nx.reshape(g.shape[0], h * w).contiguous(), ny.reshape(g.shape[0], h * w).contiguous()

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def _input(self, imgs):
        """[N,3,H,W] float -> fp16 channels-last (zero-padding to 8 channels for tensor cores: -1 %, +0.13 GB; not kept)."""
        return imgs.to(torch.float16).contiguous(memory_format=torch.channels_last)

    @torch.no_grad()
    def src_features(self, images):
        """scene-level feature cache: features of the source views (computed once per image when a scene is
        processed view by view; exact, the feature net is per-image)."""
        return self.features(self._input(images[0, 1:]))

    @torch.no_grad()
    def forward(self, images, intrinsics, extrinsics, depth_min, depth_max, src_feats=None):
        B, N, _, H, W = images.shape
        assert B == 1, "batch 1"
        assert H % 8 == 0 and W % 8 == 0, "H, W must be multiples of 8 (upstream resizes otherwise)"
        if src_feats is None:
            x = self._input(images[0])
            feats = self.features(x)
        else:
            x = self._input(images[0, :1])
            fr = self.features(x)
            feats = {k: torch.cat((fr[k], src_feats[k])).contiguous(memory_format=torch.channels_last) for k in fr}
        dmin, dmax = depth_min.float(), depth_max.float()
        K, E = intrinsics.float(), extrinsics.float()
        depth = None
        vw = None
        conf = None
        scale = 0.125
        for stage in (3, 2, 1):
            st = self.pm[stage - 1]
            proj = self._proj(K, E, scale)
            scale *= 2.0
            f = feats[stage]
            if self.mode == "max":
                depth, vw, conf = self._stage_max(st, f, proj, dmin, dmax, depth, vw, stage == 1)
            else:
                depth, vw, conf = self._stage_friendly(st, f, proj, dmin, dmax, depth, vw, stage == 1)
            if stage > 1:
                depth = F.interpolate(depth, scale_factor=2.0, mode="nearest")
                vw = F.interpolate(vw, scale_factor=2.0, mode="nearest")
        depth = self.refine(x[0:1], depth, dmin, dmax)
        conf = F.interpolate(conf, size=[H, W], mode="nearest")
        return depth[:, 0], conf[:, 0]

    def _offsets(self, st, ref, h, w):
        """propagation / evaluation grids as upstream (offset convs in fp16, grids in fp32)."""
        dev = ref.device
        pgrid = None
        if st["P"] > 0 and not (st["stage"] == 1 and st["iters"] == 1):
            off = F.conv2d(ref, *st["propa"], 1, st["dil"], st["dil"]).float().view(1, 2 * st["P"], h * w)
            pgrid = st["grid_fn"](st["grid_type"]["propagation"], 1, h, w, off, dev)
        off = F.conv2d(ref, *st["evalc"], 1, st["dil"], st["dil"]).float().view(1, 2 * st["NE"], h * w)
        egrid = st["grid_fn"](st["grid_type"]["evaluation"], 1, h, w, off, dev)
        return pgrid, egrid

    def _init(self, st, depth, dmin, dmax, h, w):
        """random init (stage 3, iteration 1) or local perturbation, as upstream DepthInitialization."""
        imin, imax = 1.0 / dmin, 1.0 / dmax
        if depth is None:
            n = 48
            s = torch.rand(size=(1, n, h, w), device=dmin.device) + torch.arange(0, n, 1, device=dmin.device).view(1, n, 1, 1)
            s = imax.view(1, 1, 1, 1) + s / n * (imin.view(1, 1, 1, 1) - imax.view(1, 1, 1, 1))
            return 1.0 / s
        k = st["nsample"]
        s = torch.arange(-k // 2, k // 2, 1, device=depth.device).view(1, k, 1, 1).float()
        s = 1.0 / depth + ((imin - imax) * st["scale"]).view(1, 1, 1, 1) * s
        return 1.0 / torch.clamp(s, min=imax[0], max=imin[0])

    # ------------------------------------------------------------------ friendly (plain PyTorch)
    def _stage_friendly(self, st, f, proj, dmin, dmax, depth, vw, last):
        G = st["G"]
        _, C, h, w = f.shape
        S = f.shape[0] - 1
        ref16 = f[0:1]
        ref = ref16.float()
        src = f[1:].float()
        pgrid, egrid = self._offsets(st, ref16, h, w)
        # feature weight [1, NE, h, w]
        NE = st["NE"]
        wt = F.grid_sample(ref, egrid, mode="bilinear", padding_mode="border", align_corners=False)
        wt = (wt.view(1, G, C // G, NE, h, w) * ref.view(1, G, C // G, 1, h, w)).mean(2)
        fweight = torch.sigmoid(_mlp2d(wt, st["fw2d"]))
        ys, xs = torch.meshgrid(torch.arange(h, device=f.device, dtype=torch.float32),
                                torch.arange(w, device=f.device, dtype=torch.float32), indexing="ij")
        xyz = torch.stack((xs.reshape(-1), ys.reshape(-1), torch.ones(h * w, device=f.device)))  # [3, hw]
        rot = proj[:, :9].view(S, 3, 3)
        trans = proj[:, 9:].view(S, 3, 1, 1)
        rxyz = torch.matmul(rot, xyz).unsqueeze(2)  # [S,3,1,hw]
        conf = None
        for it in range(1, st["iters"] + 1):
            inverse = st["stage"] == 1 and it == st["iters"]
            ds = self._init(st, depth, dmin, dmax, h, w)
            if pgrid is not None and not (st["stage"] == 1 and it == st["iters"]):
                D0 = ds.shape[1]
                prop = F.grid_sample(ds[:, D0 // 2:D0 // 2 + 1], pgrid, mode="bilinear", padding_mode="border",
                                     align_corners=False).view(1, st["P"], h, w)
                ds = torch.sort(torch.cat((ds, prop), 1), 1)[0]
            D = ds.shape[1]
            # depth weight * feature weight, normalized over neighbours: [1, D, NE, h, w]
            imin, imax = 1.0 / dmin, 1.0 / dmax
            xn = (1.0 / ds - imax.view(1, 1, 1, 1)) / (imin - imax).view(1, 1, 1, 1)
            x1 = F.grid_sample(xn, egrid, mode="bilinear", padding_mode="border", align_corners=False).view(1, D, NE, h, w)
            dw = torch.sigmoid(4.0 - 2.0 * (torch.abs(x1 - xn.unsqueeze(2)) / st["scale"]).clamp(0, 4))
            weight = dw * fweight.unsqueeze(1)
            weight = weight / weight.sum(2, keepdim=True)
            # warp + group correlation per source
            dflat = ds.view(1, 1, D, h * w)
            compute_vw = vw is None
            vws = []
            sim_sum = torch.zeros((1, G, D, h, w), device=f.device)
            wsum = torch.full((1, 1, 1, h, w), 1e-5, device=f.device)
            for s in range(S):
                p = rxyz[s:s + 1] * dflat + trans[s:s + 1]  # [1,3,D,hw]
                neg = p[:, 2:3] <= 1e-3
                px = torch.where(neg, torch.full_like(p[:, 0:1], float(w)), p[:, 0:1])
                py = torch.where(neg, torch.full_like(p[:, 1:2], float(h)), p[:, 1:2])
                pz = torch.where(neg, torch.ones_like(p[:, 2:3]), p[:, 2:3])
                gx = (px / pz) / ((w - 1) / 2) - 1
                gy = (py / pz) / ((h - 1) / 2) - 1
                grid = torch.stack((gx[:, 0], gy[:, 0]), 3).view(1, D * h, w, 2)
                with torch.backends.cudnn.flags(enabled=False):  # native kernel is 10-25 % faster than cuDNN's sampler
                    warped = F.grid_sample(src[s:s + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                wv, rv = warped.view(G, C // G, D, h * w), ref.view(G, C // G, 1, h * w)
                sim = wv[:, 0] * rv[:, 0]
                for k in range(1, C // G):
                    sim.addcmul_(wv[:, k], rv[:, k])
                sim = sim.mul_(G / C).view(1, G, D, h, w)
                if compute_vw:
                    v = torch.sigmoid(_mlp2d(sim, st["pw2d"])).amax(1, keepdim=True)  # [1,1,h,w]
                    vws.append(v)
                else:
                    v = vw[:, s:s + 1]
                sim_sum.addcmul_(sim, v.unsqueeze(1))
                wsum += v.unsqueeze(1)
            if compute_vw:
                vw = torch.cat(vws, 1)
            cost = _mlp2d(sim_sum / wsum, st["sim2d"])  # [1,D,h,w]
            cost = F.grid_sample(cost, egrid, mode="bilinear", padding_mode="border", align_corners=False).view(1, D, NE, h, w)
            score = torch.softmax((cost * weight).sum(2), 1)
            if inverse:
                idx = (torch.arange(D, device=f.device).view(1, D, 1, 1) * score).sum(1)
                imn = 1.0 / ds[:, -1]
                imx = 1.0 / ds[:, 0]
                depth = (1.0 / (imx + idx / (D - 1) * (imn - imx))).unsqueeze(1)
            else:
                depth = (ds * score).sum(1, keepdim=True)
            if last and it == st["iters"]:
                s4 = 4 * F.avg_pool3d(F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1).squeeze(1)
                di = (score * torch.arange(D, device=f.device, dtype=torch.float).view(1, D, 1, 1)).sum(1, keepdim=True)
                conf = torch.gather(s4, 1, di.long().clamp(0, D - 1))
        return depth, vw, conf

    # ------------------------------------------------------------------ max (fused CUDA kernels)
    def _stage_max(self, st, f, proj, dmin, dmax, depth, vw, last):
        ext = _EXT
        G = st["G"]
        _, C, h, w = f.shape
        S = f.shape[0] - 1
        hw = h * w
        nhwc = f.permute(0, 2, 3, 1)  # channels-last -> contiguous [N, h, w, C]
        if not nhwc.is_contiguous():
            nhwc = nhwc.contiguous()
        ref = nhwc[0]
        src = nhwc[1:]
        pgrid, egrid = self._offsets(st, f[0:1], h, w)
        enx, eny = self._unnorm(egrid, h, w)
        pnx = pny = None
        if pgrid is not None:
            pnx, pny = self._unnorm(pgrid, h, w)
        NE = st["NE"]
        fw = torch.empty((NE, hw), device=f.device)
        ext.feat_weight(ref, enx, eny, st["fw"], fw, h, w, G)
        imin, imax = float(1.0 / dmin), float(1.0 / dmax)
        conf = None
        out = torch.empty((1, 1, h, w), device=f.device)
        for it in range(1, st["iters"] + 1):
            inverse = st["stage"] == 1 and it == st["iters"]
            use_p = pgrid is not None and not (st["stage"] == 1 and it == st["iters"])
            P = st["P"] if use_p else 0
            if depth is None:
                base = self._init(st, None, dmin, dmax, h, w)[0].contiguous()
                Db = base.shape[0]
                if P == 0:
                    ds = base
                else:
                    ds = torch.empty((Db + P, hw), device=f.device)
                    ext.hypo(None, base, pnx, pny, ds, Db, P, imin, imax, 0.0, h, w)
            else:
                Db = st["nsample"]
                ds = torch.empty((Db + P, hw), device=f.device)
                ext.hypo(depth.reshape(hw).contiguous(), None, pnx if P else enx, pny if P else eny, ds, Db, P,
                         imin, imax, (imin - imax) * st["scale"], h, w)
            ds = ds.view(-1, hw)
            D = ds.shape[0]
            if vw is None:
                vw = torch.zeros((1, S, h, w), device=f.device)
                ext.view_weight(ref, src, proj, ds, st["pw"], vw, h, w, G)
            vwc = vw.contiguous()
            cost = torch.empty((D, hw), device=f.device)
            ext.cost(ref, src, proj, ds, vwc, st["sim"], cost, h, w, G)
            want_conf = last and it == st["iters"]
            cf = torch.empty((1, 1, h, w), device=f.device) if want_conf else None
            ext.aggregate(cost, ds, enx, eny, fw, out, cf, imax, imin - imax, 1.0 / st["scale"], h, w, inverse)
            depth = out.clone() if it < st["iters"] else out
            if want_conf:
                conf = cf
        return depth, vw, conf
