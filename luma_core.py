"""
LumaGuide core math.

Nothing in this file imports ComfyUI, so it can be unit-tested on its own.

---------------------------------------------------------------------------
The idea in one paragraph
---------------------------------------------------------------------------
QR Code Monster was, functionally, a learned "make the low-frequency
brightness of the image match this map" constraint. That constraint is a
LINEAR measurement of the image:

        y  =  A(x)  =  Blur( Luma( Decode(z) ) )

Flow models give us a clean-image estimate z0_hat at every step. If we can
measure A(z0_hat) cheaply and push z0_hat so that A(z0_hat) matches the
target, the sampler carries that correction forward and the transformer
fills in everything A cannot see (color, texture, semantics). This is the
DDNM "range/null-space" idea applied in latent space:

        z0_tilde = z0_hat + A_pinv( y - A(z0_hat) )

We avoid the VAE entirely by using a LINEAR PROBE:

        Luma(Decode(z))  ~=  w . z + b        (per latent pixel)

w is a C-vector (C = latent channels, 16 for Krea2's Wan2.1-style latent).
Because the probe is linear and per-pixel, Blur commutes with it, and the
"pseudo-inverse" is a single per-pixel direction:

        dz = a * r / (w . a)         where r = band-limited luma residual

 - If a == w this is the minimum-norm correction (pure DDNM).
 - If a is the "natural brightness direction" fitted from real encoded
   images (cov(z, L) / var(L)), the correction moves the latent the way
   latents actually move when real images get brighter. That stays closer
   to the data manifold and reduces the gray-haze look. (w . a) makes sure
   the probe sees exactly the requested change r either way.

---------------------------------------------------------------------------
Tensor conventions
---------------------------------------------------------------------------
 - Latents arrive as [B, C, H, W] or [B, C, T, H, W] (Krea2 uses the 5D
   Wan21 format with T == 1). They are in the SAMPLER's latent space, i.e.
   after latent_format.process_in(). The probe must be fitted in that same
   space. ComfyUI's preview factors already are.
 - Luma maps are handled as [N, H, W] float32, where N = B*T.
 - Luma units: about -1 (black) .. +1 (white), matching ComfyUI's
   Latent2RGB preview convention, which is also what the calibrated probe
   is fitted to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

# Rec.601 luma weights. The probe only needs to be self-consistent, and the
# target map is a user-drawn pattern anyway, so 601 vs 709 does not matter.
LUMA_RGB = (0.299, 0.587, 0.114)


# =============================================================================
# Probe
# =============================================================================

@dataclass
class LumaProbe:
    """Linear latent -> luma probe plus the direction used to apply corrections.

    w : [C]  luma = w . z + b
    a : [C]  direction a correction is applied along (see module docstring)
    b : float
    """
    w: torch.Tensor
    a: torch.Tensor
    b: float
    name: str = "unnamed"
    source: str = "unknown"          # "preview_factors" or "calibrated"
    info: dict = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return int(self.w.numel())

    def wa(self) -> float:
        """w . a. The correction is divided by this, so it must not be ~0."""
        return float((self.w * self.a).sum())

    # ---- (de)serialization so a calibration can be saved and reused ----
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "w": [float(v) for v in self.w.flatten()],
            "a": [float(v) for v in self.a.flatten()],
            "b": float(self.b),
            "info": self.info,
        }

    @staticmethod
    def from_dict(d: dict) -> "LumaProbe":
        return LumaProbe(
            w=torch.tensor(d["w"], dtype=torch.float32),
            a=torch.tensor(d["a"], dtype=torch.float32),
            b=float(d["b"]),
            name=d.get("name", "unnamed"),
            source=d.get("source", "calibrated"),
            info=d.get("info", {}),
        )

    @staticmethod
    def from_preview_factors(latent_rgb_factors, latent_rgb_factors_bias=None, name="preview") -> "LumaProbe":
        """Build a probe from ComfyUI's latent_format.latent_rgb_factors.

        Those factors are a [C][3] list: RGB = z @ factors + bias, applied to
        the sampler-space x0 (see latent_preview.Latent2RGBPreviewer). Luma is a
        fixed mix of RGB, so the luma probe is w = factors @ LUMA_RGB.

        The correction direction is a = w (minimum-norm). That is the weakest
        assumption we can make without calibration data.
        """
        fac = torch.tensor(latent_rgb_factors, dtype=torch.float32)      # [C, 3]
        lum = torch.tensor(LUMA_RGB, dtype=torch.float32)                # [3]
        w = fac @ lum                                                    # [C]
        b = 0.0
        if latent_rgb_factors_bias is not None:
            b = float(torch.tensor(latent_rgb_factors_bias, dtype=torch.float32) @ lum)
        return LumaProbe(w=w, a=w.clone(), b=b, name=name, source="preview_factors")


# =============================================================================
# Blur / low-pass
# =============================================================================

def _gauss_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gauss_blur_direct(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of [N, H, W]. Replicate padding (no reflect
    size limit, and edges don't drift toward black)."""
    k = _gauss_kernel1d(sigma, img.device, img.dtype)
    r = (k.numel() - 1) // 2
    x = img.unsqueeze(1)                                   # [N,1,H,W]
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x.squeeze(1)


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian low-pass of [N, H, W] with sigma in (latent) pixels.

    For large sigma (the early, very coarse steps can ask for 20+ latent px)
    a direct kernel gets huge, so we blur a downsampled copy and upsample
    again. That is the usual Gaussian-pyramid approximation and it is
    plenty accurate for a smooth target.
    """
    if sigma <= 0.05:
        return img
    if sigma <= 3.0:
        return _gauss_blur_direct(img, sigma)

    N, H, W = img.shape
    f = sigma / 1.5                                   # blur at ~1.5 px after downsampling
    h2, w2 = max(1, int(round(H / f))), max(1, int(round(W / f)))
    small = F.interpolate(img.unsqueeze(1), size=(h2, w2), mode="area")
    small = _gauss_blur_direct(small.squeeze(1), 1.5).unsqueeze(1)
    up = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    return up.squeeze(1)


# =============================================================================
# Target preparation
# =============================================================================

def image_to_luma01(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE [B, H, W, 3|4] in 0..1  ->  luma [B, H, W] in 0..1."""
    rgb = image[..., :3].float()
    lum = torch.tensor(LUMA_RGB, dtype=rgb.dtype, device=rgb.device)
    return (rgb * lum).sum(-1)


def resize_map(m: torch.Tensor, H: int, W: int, fit: str = "stretch") -> torch.Tensor:
    """Resize [N, h, w] to [N, H, W] with area averaging.

    fit = "stretch": ignore aspect ratio.
    fit = "crop":    center-crop the map to the latent aspect first, then resize.
    """
    if fit == "crop":
        _, h, w = m.shape
        target_ar = W / H
        if w / h > target_ar:                       # map too wide: crop width
            nw = max(1, int(round(h * target_ar)))
            x0 = (w - nw) // 2
            m = m[:, :, x0:x0 + nw]
        else:                                       # map too tall: crop height
            nh = max(1, int(round(w / target_ar)))
            y0 = (h - nh) // 2
            m = m[:, y0:y0 + nh, :]
    mode = "area" if (m.shape[-2] >= H and m.shape[-1] >= W) else "bilinear"
    kw = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(m.unsqueeze(1).float(), size=(H, W), mode=mode, **kw).squeeze(1)


# =============================================================================
# Guide configuration
# =============================================================================

@dataclass
class LumaGuideConfig:
    """Everything needed to compute a correction at a given sigma.

    target01:   [Nt, h, w] luma map in 0..1 (after invert/levels), full res.
                Resized lazily to the latent grid on first use.
    probe:      LumaProbe or None. None means "build from the model's preview
                factors at sampling time".
    """
    target01: torch.Tensor
    probe: Optional[LumaProbe] = None

    strength: float = 0.8            # 0..1 fraction of the residual removed per step (1 = hard projection)
    start_percent: float = 0.0       # same convention as ControlNet Apply (Advanced)
    end_percent: float = 0.7
    taper: float = 0.3               # last fraction of the window over which strength ramps to 0

    match: str = "relative"          # "relative": keep the image's own mean brightness, enforce the pattern
                                     # "absolute": enforce black/white levels literally
    contrast: float = 0.35           # relative mode: target swing, in luma units (-1..1 scale), per unit of map swing
    black_level: float = -0.6        # absolute mode: luma for map value 0
    white_level: float = 0.6         # absolute mode: luma for map value 1
    tolerance: float = 0.0           # hinge dead-band (luma units). Residuals smaller than this are ignored.

    blur_schedule: str = "coarse_to_fine"  # "coarse_to_fine" | "spectral" | "fixed"
    blur_start: float = 8.0          # latent px. Blur at the START of the window (coarsest scale enforced)
    blur_end: float = 1.0            # latent px. Blur at the END of the window (finest scale enforced)
    blur_max_frac: float = 0.25      # max blur as a fraction of the short latent side

    fit: str = "stretch"             # how the map is fitted to the latent aspect ratio
    deblur_iters: int = 2            # Van Cittert iterations so the BLURRED luma lands on target (see compute_correction)
    max_delta: float = 2.0           # clamp per-channel |dz| (sampler space is ~unit std). 0 disables.
    verbose: bool = False

    # caches (not user-facing)
    _target_cache: dict = field(default_factory=dict, repr=False)

    def target_on_grid(self, N: int, H: int, W: int, device) -> torch.Tensor:
        key = (N, H, W, str(device))
        t = self._target_cache.get(key)
        if t is None:
            m = resize_map(self.target01, H, W, self.fit).to(device)
            if m.shape[0] == 1 and N > 1:
                m = m.expand(N, H, W)
            elif m.shape[0] != N:
                # Batch mismatch: repeat/cycle the maps over the batch.
                idx = torch.arange(N, device=m.device) % m.shape[0]
                m = m[idx]
            t = m.contiguous()
            self._target_cache[key] = t
        return t


# =============================================================================
# Schedules
# =============================================================================

def blur_for_sigma(cfg: LumaGuideConfig, sigma: float, H: int, W: int,
                   s_start: float, s_end: float) -> float:
    """Blur width (latent px) to enforce at noise level sigma.

    Every mode is coarse-to-fine, because sampling is: low spatial
    frequencies are settled at high noise, fine ones only later. Enforcing a
    fine pattern while the model is still deciding the global layout only
    fights it, and enforcing only coarse blobs late leaves the pattern mushy.

    All modes go from blur_start (window start) to blur_end (window end), so
    the knobs mean the same thing whatever the model's sigma shift:

    "coarse_to_fine": log-linear in window progress. Predictable, the default.

    "spectral": follows the noise level itself. With x_t = (1-s) x0 + s e and
        a ~1/k^2 image spectrum, the smallest wavelength that still carries
        signal scales like s/(1-s). The blur is proportional to that,
        normalized to equal blur_end at the window end and clamped to
        blur_start. With shifted schedules (Flux/Krea2 shift = 1.15 is an
        exponential shift) most steps sit at high sigma, so spectral stays
        coarse longer than coarse_to_fine. That rule was derived for
        pixel-space statistics, so treat it as an experiment.

    "fixed": always blur_end.
    """
    bmax = max(cfg.blur_end, cfg.blur_max_frac * min(H, W))
    lo, hi = cfg.blur_end, max(cfg.blur_start, cfg.blur_end)
    if cfg.blur_schedule == "fixed":
        b = lo
    elif cfg.blur_schedule == "spectral" and sigma <= 1.0 and s_end < 1.0:
        ratio = sigma / max(1e-4, 1.0 - sigma)
        ratio_end = max(1e-4, s_end) / max(1e-4, 1.0 - s_end)
        b = min(hi, lo * ratio / ratio_end)
    else:
        # coarse_to_fine (also the fallback for sigma > 1, i.e. EPS models)
        s_hi = min(s_start, 1.0) if sigma <= 1.0 else s_start
        span = max(1e-8, s_hi - s_end)
        p = min(1.0, max(0.0, (s_hi - sigma) / span))
        b = math.exp(math.log(max(hi, 1e-3)) * (1 - p) + math.log(max(lo, 1e-3)) * p)
    return float(min(max(b, 0.0), bmax))


def strength_for_sigma(cfg: LumaGuideConfig, sigma: float, s_start: float, s_end: float) -> float:
    """Strength inside [s_end, s_start] (sigmas decrease during sampling), with
    a linear taper over the last `taper` fraction of the window. 0 outside."""
    if not (s_end <= sigma <= s_start):
        return 0.0
    span = max(1e-8, s_start - s_end)
    p = (s_start - sigma) / span                     # 0 at window start, 1 at window end
    k = 1.0
    if cfg.taper > 0 and p > 1.0 - cfg.taper:
        k = max(0.0, (1.0 - p) / cfg.taper)
    return cfg.strength * k


# =============================================================================
# The correction
# =============================================================================

def _flatten_latent(z: torch.Tensor):
    """[B,C,H,W] or [B,C,T,H,W] -> ([N,C,H,W], restore_fn)."""
    if z.ndim == 4:
        return z, (lambda t: t)
    if z.ndim == 5:
        B, C, T, H, W = z.shape
        zf = z.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        def restore(t):
            return t.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return zf, restore
    raise ValueError(f"LumaGuide: unsupported latent rank {z.ndim} (shape {tuple(z.shape)})")


def probe_luma(z: torch.Tensor, probe: LumaProbe) -> torch.Tensor:
    """[N,C,H,W] -> [N,H,W] probe luma."""
    w = probe.w.to(z.device, z.dtype)
    return torch.einsum("nchw,c->nhw", z, w) + probe.b


def compute_correction(z0: torch.Tensor, cfg: LumaGuideConfig, probe: LumaProbe,
                       sigma: float, lam: float, window: tuple = (1.0, 0.0),
                       weight: Optional[torch.Tensor] = None,
                       stats: Optional[dict] = None) -> torch.Tensor:
    """Return dz with the same shape as z0 (dz is zero where nothing applies).

    z0:     sampler-space x0 estimate, 4D or 5D.
    sigma:  current noise level (sets the blur width).
    lam:    already-scheduled strength for this step.
    window: (sigma_start, sigma_end) of the guide window, for the blur schedule.
    weight: optional [N?, H, W] 0..1 spatial weight (where to apply guidance).
    """
    if lam <= 0:
        return torch.zeros_like(z0)

    if probe.channels != z0.shape[1]:
        raise ValueError(
            f"LumaGuide: probe has {probe.channels} channels but latent has {z0.shape[1]}. "
            f"Wrong probe for this model?")

    orig_dtype = z0.dtype
    zf, restore = _flatten_latent(z0.float())
    N, C, H, W = zf.shape

    r_blur = blur_for_sigma(cfg, sigma, H, W, window[0], window[1])

    # Current low-frequency luma of the model's own x0 estimate.
    L = probe_luma(zf, probe)                              # [N,H,W]
    GL = gaussian_blur(L, r_blur)

    # Target at the same band.
    Y = cfg.target_on_grid(N, H, W, zf.device)             # [N,H,W] 0..1
    GY = gaussian_blur(Y, r_blur)

    if cfg.match == "absolute":
        tgt = cfg.black_level + (cfg.white_level - cfg.black_level) * GY
    else:
        # relative: the image keeps its own mean brightness (the model decides
        # if it is a night scene or a beach), only the PATTERN is enforced.
        # A map going 0 -> 1 asks for a luma swing of 2*contrast.
        mGL = GL.mean(dim=(-2, -1), keepdim=True)
        mGY = GY.mean(dim=(-2, -1), keepdim=True)
        tgt = mGL + 2.0 * cfg.contrast * (GY - mGY)

    res = tgt - GL

    # Hinge / dead-band: only push where the image is off by more than the
    # tolerance. This is what lets the model "explain" the pattern with its
    # own content instead of having it painted on. 0 = plain L2 projection.
    if cfg.tolerance > 0:
        res = torch.sign(res) * torch.clamp(res.abs() - cfg.tolerance, min=0.0)

    if weight is not None:
        wgt = resize_map(weight.float(), H, W, cfg.fit).to(zf.device)
        if wgt.shape[0] != N:
            idx = torch.arange(N, device=wgt.device) % wgt.shape[0]
            wgt = wgt[idx]
        res = res * wgt

    # `res` is the change we want in the BLURRED luma. Adding it as-is would
    # get blurred again when measured (G(res) != res), so the error would only
    # partly close. A few Van Cittert iterations find d with G(d) ~= res.
    # Kept small on purpose: more iterations act like sharpening/deconvolution
    # and start adding mid-frequency texture.
    if cfg.deblur_iters > 0 and r_blur > 0.05:
        target_change = res
        d = res
        for _ in range(cfg.deblur_iters):
            d = d + (target_change - gaussian_blur(d, r_blur))
        res = d

    res = res * lam

    # Apply along a, scaled so the probe sees exactly `res`.
    a = probe.a.to(zf.device, zf.dtype)
    wa = float((probe.w.to(a) * a).sum())
    if abs(wa) < 1e-6:
        raise ValueError("LumaGuide: probe w.a is ~0, cannot apply a correction. Re-calibrate the probe.")
    dz = res.unsqueeze(1) * (a / wa).view(1, C, 1, 1)      # [N,C,H,W]

    if cfg.max_delta > 0:
        dz = dz.clamp(-cfg.max_delta, cfg.max_delta)

    if stats is not None:
        stats["blur"] = r_blur
        stats["lam"] = lam
        stats["res_rms_before"] = float((tgt - GL).pow(2).mean().sqrt())
        stats["dz_rms"] = float(dz.pow(2).mean().sqrt())

    return restore(dz).to(orig_dtype)


# =============================================================================
# Runtime: binds a config to a model's sampling schedule and latent format
# =============================================================================

class GuideRuntime:
    """Shared by the sampler and the model-patch paths.

    Duck-typed on purpose (no ComfyUI imports): `model_sampling` must have
    percent_to_sigma(p); `latent_format` must have latent_rgb_factors and
    latent_rgb_factors_bias (only used when no calibrated probe was given).
    """

    def __init__(self, cfg: LumaGuideConfig, weight: Optional[torch.Tensor] = None):
        self.cfg = cfg
        self.weight = weight
        self._window_cache = {}
        self._probe_cache = {}

    def window(self, model_sampling):
        key = id(model_sampling)
        win = self._window_cache.get(key)
        if win is None:
            s_start = float(model_sampling.percent_to_sigma(self.cfg.start_percent))
            s_end = float(model_sampling.percent_to_sigma(self.cfg.end_percent))
            win = (s_start, s_end)
            self._window_cache[key] = win
        return win

    def probe_for(self, latent_format) -> LumaProbe:
        if self.cfg.probe is not None:
            return self.cfg.probe
        key = id(latent_format)
        p = self._probe_cache.get(key)
        if p is None:
            fac = getattr(latent_format, "latent_rgb_factors", None)
            if fac is None:
                raise ValueError(
                    "LumaGuide: this model's latent format has no preview RGB factors. "
                    "Connect a calibrated probe (LumaGuide Calibrate Probe).")
            p = LumaProbe.from_preview_factors(fac, getattr(latent_format, "latent_rgb_factors_bias", None),
                                               name=type(latent_format).__name__ + "_preview")
            self._probe_cache[key] = p
        return p

    def strength_at(self, sigma: float, model_sampling) -> float:
        s_start, s_end = self.window(model_sampling)
        return strength_for_sigma(self.cfg, sigma, s_start, s_end)

    def apply(self, z0: torch.Tensor, sigma: float, model_sampling, latent_format, tag: str = "") -> torch.Tensor:
        lam = self.strength_at(sigma, model_sampling)
        if lam <= 0:
            return z0
        probe = self.probe_for(latent_format)
        stats = {} if self.cfg.verbose else None
        dz = compute_correction(z0, self.cfg, probe, sigma, lam, self.window(model_sampling), self.weight, stats)
        if stats is not None:
            print(f"[LumaGuide{tag}] sigma={sigma:.4f} lam={stats['lam']:.3f} blur={stats['blur']:.2f}px "
                  f"luma_res_rms={stats['res_rms_before']:.4f} dz_rms={stats['dz_rms']:.4f}")
        return z0 + dz


# =============================================================================
# Calibration (fitting the probe). This only fits a 16-number linear map;
# the generator itself is untouched.
# =============================================================================

def _sample_pixels(z: torch.Tensor, L: torch.Tensor, max_px: int, gen: torch.Generator):
    """z [C,h,w], L [h,w] -> X [n,C], y [n] with at most max_px random pixels."""
    C = z.shape[0]
    X = z.reshape(C, -1).t()
    y = L.reshape(-1)
    if X.shape[0] > max_px:
        idx = torch.randperm(X.shape[0], generator=gen)[:max_px]
        X, y = X[idx], y[idx]
    return X, y


def fit_probe(latents, lumas, fit_blur: float = 1.0, ridge: float = 1e-3,
              max_px_per_image: int = 20000, seed: int = 0, name: str = "calibrated"):
    """Least-squares luma probe.

    latents: list of [C, h, w] sampler-space latents (after process_in)
    lumas:   list of [h, w] ground-truth luma in -1..1, already on the latent grid
    fit_blur: both sides are Gaussian-blurred by this many latent px before the
              fit. We only ever use the probe on blurred luma, so it should be
              accurate at low frequency. A linear map commutes with blur, so this
              stays consistent with how the probe is used.

    Returns (probe, report_dict).
    """
    gen = torch.Generator().manual_seed(seed)
    Xs, ys = [], []
    for z, L in zip(latents, lumas):
        z = z.float()
        L = L.float()
        if fit_blur > 0:
            z = gaussian_blur(z, fit_blur)                 # blur each channel as an "image"
            L = gaussian_blur(L.unsqueeze(0), fit_blur)[0]
        X, y = _sample_pixels(z, L, max_px_per_image, gen)
        Xs.append(X)
        ys.append(y)
    X = torch.cat(Xs).double()
    y = torch.cat(ys).double()
    n, C = X.shape

    mx = X.mean(0)
    my = y.mean()
    Xc = X - mx
    yc = y - my

    # Ridge-regularized normal equations. ridge is relative to the mean
    # feature variance so it is scale-free.
    XtX = Xc.t() @ Xc
    lam = ridge * float(torch.diagonal(XtX).mean())
    w = torch.linalg.solve(XtX + lam * torch.eye(C, dtype=X.dtype), Xc.t() @ yc)
    b = float(my - mx @ w)

    pred = Xc @ w + my
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float((yc ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Natural brightness direction: how latents move, on average, per unit of
    # luma in real images (regression of each channel on luma).
    a = (Xc.t() @ yc) / max(float(yc @ yc), 1e-12)

    probe = LumaProbe(w=w.float(), a=a.float(), b=b, name=name, source="calibrated",
                      info={"r2_fit": r2, "fit_blur": fit_blur, "ridge": ridge,
                            "n_pixels": n, "channels": C, "w_dot_a": float(w @ a)})
    return probe, {"r2": r2, "n": n, "w_dot_a": float(w @ a)}


def probe_r2(probe: LumaProbe, latents, lumas, blur: float) -> dict:
    """Evaluate a probe on data at a given blur. Returns raw R^2 (absolute
    accuracy, includes scale/offset errors) and corr^2 (pattern accuracy only)."""
    ps, ts = [], []
    for z, L in zip(latents, lumas):
        z = z.float().unsqueeze(0)
        Lp = probe_luma(z, probe)[0]
        Lt = L.float()
        if blur > 0:
            Lp = gaussian_blur(Lp.unsqueeze(0), blur)[0]
            Lt = gaussian_blur(Lt.unsqueeze(0), blur)[0]
        ps.append(Lp.flatten())
        ts.append(Lt.flatten())
    p = torch.cat(ps).double()
    t = torch.cat(ts).double()
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    pc, tc = p - p.mean(), t - t.mean()
    corr = float((pc @ tc) / max(float(pc.norm() * tc.norm()), 1e-12))
    return {"r2": r2, "corr2": corr * corr, "scale": float((pc @ tc) / max(float(pc @ pc), 1e-12))}
