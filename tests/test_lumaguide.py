"""Mechanics tests for LumaGuide against the real ComfyUI code (CPU, fake denoiser).

Run from anywhere:   COMFYUI_PATH=/path/to/ComfyUI python tests/test_lumaguide.py
(the pack must be inside ComfyUI/custom_nodes/ComfyUI-LumaGuide)
"""
import os, sys
COMFY = os.environ.get("COMFYUI_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.argv = ["main.py", "--cpu"]
sys.path.insert(0, COMFY)
import comfy.options
comfy.options.enable_args_parsing()
import comfy.samplers, comfy.model_sampling, comfy.latent_formats, folder_paths
import importlib
LG = importlib.import_module("custom_nodes.ComfyUI-LumaGuide")
nodes_mod = importlib.import_module("custom_nodes.ComfyUI-LumaGuide.nodes")
core = importlib.import_module("custom_nodes.ComfyUI-LumaGuide.luma_core")
import math, json, os, types
import torch
import comfy.k_diffusion.sampling as kds
import comfy.model_sampling as cms
import comfy.latent_formats as clf

torch.manual_seed(0)
fails = []
def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + info if info else ""))
    if not cond:
        fails.append(name)

# ---- fake Krea2-like model: Wan21 latent format, Flux flow sampling --------
class MS(cms.ModelSamplingFlux, cms.CONST):
    pass
ms = MS()
lf = clf.Wan21()

class FakeBase:                       # stands in for BaseModel
    model_sampling = ms
    latent_format = lf
class FakeGuider:                     # stands in for CFGGuider (model_wrap)
    inner_model = FakeBase()

def gaussian_denoiser(x, sigma):
    """Exact posterior mean for x0 ~ N(0, I) under x_t = (1-s) x0 + s e."""
    s = sigma.view(-1, *([1] * (x.ndim - 1)))
    return (1 - s) / ((1 - s) ** 2 + s ** 2) * x

class FakeK:                          # stands in for KSamplerX0Inpaint
    inner_model = FakeGuider()
    calls = 0
    def __call__(self, x, sigma, **kw):
        FakeK.calls += 1
        return gaussian_denoiser(x, sigma)

def flux_sigmas(n):
    t = torch.linspace(1, 0, n + 1)
    return torch.stack([ms.sigma(tt) if tt > 0 else torch.tensor(0.0) for tt in t]).float()

# target: a spiral-ish pattern, 0..1, at "image" resolution 512x512
H = W = 512
yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
rr, th = (xx ** 2 + yy ** 2).sqrt(), torch.atan2(yy, xx)
target = (torch.sin(10 * rr + 3 * th) > 0).float()
img = target[None, :, :, None].repeat(1, 1, 1, 3)

Settings = LG.NODE_CLASS_MAPPINGS["LumaGuideSettings"]()
def make_guide(**kw):
    p = dict(strength=0.8, start_percent=0.0, end_percent=0.7, taper=0.3, match="relative", contrast=0.35,
             black_level=-0.6, white_level=0.6, tolerance=0.0, blur_schedule="coarse_to_fine", blur_start=8.0,
             blur_end=1.0, fit="crop", invert=False, map_pre_blur=0.0, max_delta=2.0, verbose=False)
    p.update(kw)
    return Settings.build(img, **p)[0]

def run_sampler(guide, sig, x, inner=0, renoise=0.0):
    fn = nodes_mod._make_sampler_function(guide, inner, renoise)
    return fn(FakeK(), x.clone(), sig, extra_args={"seed": 1}, disable=True)

def pattern_corr(z, blur=2.0):
    probe = core.LumaProbe.from_preview_factors(lf.latent_rgb_factors, lf.latent_rgb_factors_bias)
    zf = z[:, :, 0] if z.ndim == 5 else z
    L = core.gaussian_blur(core.probe_luma(zf.float(), probe), blur)
    T = core.gaussian_blur(core.resize_map(target[None], L.shape[-2], L.shape[-1]), blur)
    a, b = L.flatten() - L.mean(), T.flatten() - T.mean()
    return float(a @ b / (a.norm() * b.norm()))

sig = flux_sigmas(8)
x0 = torch.randn(1, 16, 1, 64, 64)           # 5D like Krea2, 512px image -> 64 latent

# 1. no guidance == comfy euler
g0 = make_guide(strength=0.0)
out_ours = run_sampler(g0, sig, x0)
out_euler = kds.sample_euler(FakeK(), x0.clone(), sig, disable=True)
check("sampler == euler when strength 0", torch.allclose(out_ours, out_euler, atol=1e-5),
      f"maxdiff={float((out_ours-out_euler).abs().max()):.2e}")

# 2. guidance raises pattern correlation
c_none = pattern_corr(out_euler)
g = make_guide()
out_g = run_sampler(g, sig, x0)
c_g = pattern_corr(out_g)
check("guided pattern corr >> unguided", c_g > 0.8 and c_none < 0.2, f"unguided={c_none:.3f} guided={c_g:.3f}")
check("guided output finite", torch.isfinite(out_g).all().item())
check("guided latent std reasonable", 0.5 < float(out_g.std()) < 2.0, f"std={float(out_g.std()):.3f}")

# 3. inner iters / renoise run and use extra NFEs
FakeK.calls = 0
run_sampler(g, sig, x0, inner=1)
check("inner_iters adds NFEs only on guided steps", FakeK.calls > 8, f"calls={FakeK.calls}")
out_r = run_sampler(g, sig, x0, renoise=0.3)
check("renoise runs, still follows pattern", pattern_corr(out_r) > 0.7, f"corr={pattern_corr(out_r):.3f}")

# 4. hard projection: strength 1 at one step makes probe luma (blurred) match target exactly-ish
g1 = make_guide(strength=1.0, taper=0.0, match="absolute", blur_schedule="fixed", blur_end=3.0)
D = torch.randn(1, 16, 1, 64, 64)
s_mid = 0.8
D2 = g1.apply(D, s_mid, ms, lf)
probe = g1.probe_for(lf)
L2 = core.gaussian_blur(core.probe_luma(D2[:, :, 0], probe), 3.0)
T2 = g1.cfg.black_level + (g1.cfg.white_level - g1.cfg.black_level) * core.gaussian_blur(
    g1.cfg.target_on_grid(1, 64, 64, "cpu"), 3.0)
L1 = core.gaussian_blur(core.probe_luma(D[:, :, 0], probe), 3.0)
e_before = float((L1 - T2).pow(2).mean().sqrt()); e_after = float((L2 - T2).pow(2).mean().sqrt())
check("strength=1 projection shrinks band-limited error", e_after < 0.25 * e_before,
      f"rms before={e_before:.3f} after={e_after:.3f}")

# 5. window: outside [end, start] nothing happens
D3 = g.apply(D, 0.05, ms, lf)
check("no change outside window", D3 is D)
s_end = g.window(ms)[1]
check("window end sigma sane (flux shift 1.15 maps 0.7 -> ~0.58)", 0.5 < s_end < 0.65, f"end sigma={s_end:.3f}")

# 6. post-cfg patch path through real comfy cfg_function
Patch = LG.NODE_CLASS_MAPPINGS["LumaGuideModelPatch"]()
class FakePatcher:
    def __init__(self): self.model_options = {}
    def clone(self):
        c = FakePatcher(); c.model_options = dict(self.model_options); return c
    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        import comfy.model_patcher as mp
        self.model_options = mp.set_model_options_post_cfg_function(self.model_options, fn, disable_cfg1_optimization)
pm = Patch.patch(FakePatcher(), g)[0]
xt = torch.randn(1, 16, 1, 64, 64)
sg = torch.tensor([0.7])
pred = gaussian_denoiser(xt, sg)
res = comfy.samplers.cfg_function(FakeBase(), pred, pred, 1.0, xt, sg, model_options=pm.model_options)
check("post-cfg patch modifies denoised", not torch.allclose(res, pred))
check("post-cfg == runtime.apply", torch.allclose(res, g.apply(pred, 0.7, ms, lf), atol=1e-6))

# 7. 4D latents (Flux-style) also work
lf4 = clf.Flux()
D4 = torch.randn(2, 16, 64, 64)
D4g = g1.apply(D4, 0.8, ms, lf4)
check("4D batch-2 latent works", D4g.shape == D4.shape and not torch.allclose(D4g, D4))

# 8. calibration fit recovers a known linear probe
w_true = torch.randn(16); b_true = 0.1
lat, lum = [], []
for i in range(6):
    z = core.gaussian_blur(torch.randn(16, 48, 48), 1.5) * 3
    L = torch.einsum("chw,c->hw", z, w_true) + b_true + 0.01 * torch.randn(48, 48)
    lat.append(z); lum.append(L)
pr, rep = core.fit_probe(lat, lum, fit_blur=1.0, ridge=0.0)
check("fit_probe recovers w", torch.allclose(pr.w, w_true, atol=0.02) and rep["r2"] > 0.99,
      f"r2={rep['r2']:.4f} maxerr={float((pr.w-w_true).abs().max()):.4f}")
check("w.a ~ R2 for OLS", abs(pr.wa() - rep["r2"]) < 0.02, f"w.a={pr.wa():.4f}")

# 9. Calibrate node end-to-end with a fake 5D video VAE + save/load roundtrip
class FakeVAE:
    def __init__(self):
        g_ = torch.Generator().manual_seed(3)
        self.M = torch.randn(16, 3, generator=g_)
    def encode(self, pix):                      # pix [1,H,W,3] -> raw latent [1,16,1,H/8,W/8]
        assert pix.shape[0] == 1, "must encode one image at a time"
        x = torch.nn.functional.avg_pool2d(pix.movedim(-1, 1), 8)          # [1,3,h,w]
        z = torch.einsum("ck,bkhw->bchw", self.M, x * 2 - 1)
        z = lf.process_out(z.unsqueeze(2))                                 # back to raw space
        return z
class FakeModelPatcher:
    def get_model_object(self, k):
        return lf
cal = LG.NODE_CLASS_MAPPINGS["LumaGuideCalibrateProbe"]()
cimgs = torch.rand(4, 256, 320, 3)
cimgs = torch.nn.functional.interpolate(cimgs.movedim(-1, 1), scale_factor=0.125).repeat_interleave(8, 2).repeat_interleave(8, 3).movedim(1, -1)
probe_c, diag, report = cal.calibrate(FakeModelPatcher(), FakeVAE(), 256, 1.0, 1e-3, True, "unit_test_probe", "", 32, images=cimgs)
print(report)
check("calibrate on linear fake VAE -> R2 high", probe_c.info["r2_fit"] > 0.98, f"r2={probe_c.info['r2_fit']:.4f}")
check("diagnostic image shape", diag.ndim == 4 and diag.shape[-1] == 3, str(tuple(diag.shape)))
loader = LG.NODE_CLASS_MAPPINGS["LumaGuideLoadProbe"]()
p2, _ = loader.load("unit_test_probe.json")
check("probe save/load roundtrip", torch.allclose(p2.w, probe_c.w) and torch.allclose(p2.a, probe_c.a))
os.remove(os.path.join(nodes_mod.PROBE_DIR, "unit_test_probe.json"))

# 10. calibrated probe used in guidance
gc = make_guide(probe=probe_c)
outc = run_sampler(gc, sig, x0)
check("guidance with calibrated probe runs", torch.isfinite(outc).all().item())

# 11. Check node
chk = LG.NODE_CLASS_MAPPINGS["LumaGuideCheck"]()
comp, rep = chk.check(img * 0.8 + 0.1, g, 16.0)
check("check node: perfect image corr ~1", "= 0.99" in rep or "= 1.00" in rep, rep)

# 12. hinge: tolerance reduces correction size
gt = make_guide(tolerance=0.2)
d_small = (gt.apply(D, 0.8, ms, lf) - D).abs().mean()
d_big = (g.apply(D, 0.8, ms, lf) - D).abs().mean()
check("tolerance shrinks correction", d_small < d_big, f"{float(d_small):.4f} < {float(d_big):.4f}")

# 13. schedule values
cfg = g.cfg
ws, we = g.window(ms)
for mode in ("coarse_to_fine", "spectral", "fixed"):
    cfg.blur_schedule = mode
    print(f"blur ({mode}) per step:", [round(core.blur_for_sigma(cfg, float(s), 64, 64, ws, we), 2) for s in sig[:-1]])
cfg.blur_schedule = "coarse_to_fine"
print("flux sigmas (8 steps):", [round(float(s), 3) for s in sig])
print("strength per step:", [round(g.strength_at(float(s), ms), 3) for s in sig[:-1]])

print("\nFAILURES:", fails if fails else "none")
