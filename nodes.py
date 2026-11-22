"""
LumaGuide ComfyUI nodes.

Node overview (typical Krea2 graph):

    Load Image (luma map) --> [LumaGuide Settings] --LUMA_GUIDE--> [LumaGuide Sampler] --SAMPLER--> SamplerCustom(Advanced)
                                     ^ (optional) LUMA_PROBE
                                     |
               [LumaGuide Calibrate Probe] or [LumaGuide Load Probe]

    Alternative for plain KSampler users:
    MODEL --> [LumaGuide Model Patch] (uses the same LUMA_GUIDE) --> KSampler

    After decoding:
    IMAGE + LUMA_GUIDE --> [LumaGuide Check] -> how well the result matches the map

Why two ways to apply it:
  * LumaGuide Sampler is the exact version. It replaces the clean-image
    estimate and re-noises with the model's own noise estimate, so
    strength=1 really means "the probe luma matches the target on this
    step's trajectory". It also offers inner iterations and re-noising.
  * Model Patch uses ComfyUI's post-CFG hook, so it works with any sampler.
    Most samplers only move part of the way toward x0 per step, so the
    effective strength is lower and depends on the sampler and step count.
"""

import json
import math
import os

import torch
import torch.nn.functional as F

import comfy.model_sampling
import comfy.utils
import folder_paths  # noqa: F401  (imported so ComfyUI path setup is guaranteed)

from .luma_core import (
    GuideRuntime,
    LumaGuideConfig,
    LumaProbe,
    fit_probe,
    gaussian_blur,
    image_to_luma01,
    probe_luma,
    probe_r2,
    resize_map,
)

PROBE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes")
os.makedirs(PROBE_DIR, exist_ok=True)

CATEGORY = "LumaGuide"


def _list_probes():
    files = sorted(f for f in os.listdir(PROBE_DIR) if f.lower().endswith(".json"))
    return files if files else ["(none - run Calibrate first)"]


# =============================================================================
# Settings node: the luma map + every knob. Produces a LUMA_GUIDE object.
# =============================================================================

class LumaGuideSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_map": ("IMAGE", {"tooltip": "The hidden pattern. White = bright, black = dark. Any image; only its luma is used."}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.01,
                                       "tooltip": "Fraction of the low-frequency luma error removed per guided step. 1.0 = hard projection. >1 overshoots."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Stop guiding here so the model owns the fine detail. Like ControlNet end_percent."}),
                "taper": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Last fraction of the window over which strength fades to 0 (avoids a visible 'release' seam)."}),
                "match": (["relative", "absolute"], {"default": "relative",
                                                     "tooltip": "relative: the image keeps its own mean brightness; only the pattern is enforced. absolute: black/white levels are enforced literally."}),
                "contrast": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "relative mode: luma swing between map-black and map-white is 2*contrast (luma scale -1..1)."}),
                "black_level": ("FLOAT", {"default": -0.6, "min": -1.5, "max": 1.5, "step": 0.01}),
                "white_level": ("FLOAT", {"default": 0.6, "min": -1.5, "max": 1.5, "step": 0.01}),
                "tolerance": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Dead-band. Only push where the luma is off by more than this. Higher = more freedom for the model, softer illusion."}),
                "blur_schedule": (["coarse_to_fine", "spectral", "fixed"], {"default": "coarse_to_fine",
                                  "tooltip": "How the enforced scale moves from blur_start to blur_end across the window. coarse_to_fine: log-linear in progress. spectral: follows the noise level (stays coarse longer on shifted schedules). fixed: always blur_end."}),
                "blur_start": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 64.0, "step": 0.25,
                                         "tooltip": "Latent pixels (x8 image px for Krea2). Coarsest scale, enforced at the start of the window."}),
                "blur_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 32.0, "step": 0.05,
                                       "tooltip": "Latent pixels. Finest scale, enforced at the end of the window. Lower = crisper pattern, more 'stamped' risk."}),
                "fit": (["stretch", "crop"], {"default": "crop", "tooltip": "How the map is fitted if its aspect ratio differs from the latent."}),
                "invert": ("BOOLEAN", {"default": False}),
                "map_pre_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.5,
                                           "tooltip": "Image pixels. Softens hard-edged maps before use. QR Monster users often blurred their masks slightly."}),
                "max_delta": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                                        "tooltip": "Safety clamp on per-channel latent change per step (sampler latents have ~unit std). 0 = off."}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Print per-step residual stats to the console."}),
            },
            "optional": {
                "probe": ("LUMA_PROBE", {"tooltip": "Calibrated probe. If not connected, the model's built-in preview RGB factors are used."}),
                "weight_mask": ("MASK", {"tooltip": "Optional 0..1 mask: where the guide applies."}),
            },
        }

    RETURN_TYPES = ("LUMA_GUIDE", "IMAGE")
    RETURN_NAMES = ("guide", "processed_map")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, luma_map, strength, start_percent, end_percent, taper, match, contrast,
              black_level, white_level, tolerance, blur_schedule, blur_start, blur_end, fit,
              invert, map_pre_blur, max_delta, verbose, probe=None, weight_mask=None):
        if end_percent <= start_percent:
            raise ValueError("LumaGuide: end_percent must be greater than start_percent.")

        m = image_to_luma01(luma_map)                       # [B,H,W] 0..1
        if invert:
            m = 1.0 - m
        if map_pre_blur > 0:
            m = gaussian_blur(m, map_pre_blur)
        m = m.clamp(0, 1).cpu()

        cfg = LumaGuideConfig(
            target01=m, probe=probe,
            strength=strength, start_percent=start_percent, end_percent=end_percent, taper=taper,
            match=match, contrast=contrast, black_level=black_level, white_level=white_level,
            tolerance=tolerance, blur_schedule=blur_schedule, blur_start=blur_start,
            blur_end=blur_end, fit=fit, max_delta=max_delta, verbose=verbose,
        )
        weight = weight_mask.cpu() if weight_mask is not None else None
        guide = GuideRuntime(cfg, weight)

        preview = m.unsqueeze(-1).repeat(1, 1, 1, 3)
        return (guide, preview)


# =============================================================================
# Exact sampler
# =============================================================================

def _make_sampler_function(guide: GuideRuntime, inner_iters: int, renoise: float):
    """Build a k-diffusion style sampler function for comfy.samplers.KSAMPLER.

    Flow models (CONST, which includes Krea2):
        x_t = (1 - s) * x0 + s * eps
      Each step:
        D      = model(x, s)                    clean estimate (after CFG)
        eps    = (x - (1 - s) * D) / s          model's implied noise
        D'     = D + correction(D)              luma projection
        x_next = (1 - s_next) * D' + s_next * eps

      With no correction this is exactly Euler (same trajectory as
      ComfyUI's "euler"). With a correction it is the DDNM / PnP-Flow
      "fix x0, keep the noise, move on" update.

    Other models (EPS / V-pred, k-diffusion VE space):
        x = D + s * eps,  eps = (x - D) / s,  x_next = D' + s_next * eps
      Also identical to Euler when the correction is zero.
    """
    try:
        from comfy.utils import model_trange as trange    # newer ComfyUI
    except ImportError:
        from tqdm.auto import trange

    @torch.no_grad()
    def luma_guided_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        base = model.inner_model.inner_model               # BaseModel
        ms = base.model_sampling
        lf = base.latent_format
        is_flow = isinstance(ms, comfy.model_sampling.CONST)

        s_in = x.new_ones([x.shape[0]])
        gen = torch.Generator(device="cpu").manual_seed(int(extra_args.get("seed", 0)) + 7919)

        def implied_eps(x_, D_, s_):
            if is_flow:
                return (x_ - (1.0 - s_) * D_) / s_
            return (x_ - D_) / s_

        def combine(D_, eps_, s_):
            if is_flow:
                return (1.0 - s_) * D_ + s_ * eps_
            return D_ + s_ * eps_

        for i in trange(len(sigmas) - 1, disable=disable):
            s = sigmas[i]
            s_next = sigmas[i + 1]
            sf = float(s)

            D = model(x, s * s_in, **extra_args)
            eps = implied_eps(x, D, s)
            D_g = guide.apply(D, sf, ms, lf, tag=f" step {i}")
            active = D_g is not D

            # Optional inner iterations: put the corrected estimate back at the
            # SAME noise level, let the model re-predict, correct again. Each one
            # costs a full model evaluation. Useful for few-step models where only
            # 2-4 steps fall inside the guide window.
            if active:
                for k in range(inner_iters):
                    x_re = combine(D_g, eps, s)
                    D = model(x_re, s * s_in, **extra_args)
                    eps = implied_eps(x_re, D, s)
                    D_g = guide.apply(D, sf, ms, lf, tag=f" step {i}.{k + 1}")

            if callback is not None:
                callback({"x": x, "i": i, "sigma": s, "sigma_hat": s, "denoised": D_g})

            if float(s_next) == 0.0:
                x = D_g
                continue

            # Optional partial re-noising while guiding: blend in fresh noise.
            # This gives the model a chance to "absorb" the correction instead of
            # inheriting a noise estimate that was computed for the uncorrected
            # image (PnP-Flow re-noises fully, i.e. renoise=1).
            if active and renoise > 0:
                fresh = torch.randn(eps.shape, generator=gen, dtype=torch.float32).to(eps.device, eps.dtype)
                eps = math.sqrt(1.0 - renoise) * eps + math.sqrt(renoise) * fresh

            x = combine(D_g, eps, s_next)
        return x

    return luma_guided_sampler


class LumaGuideSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide": ("LUMA_GUIDE",),
                "inner_iters": ("INT", {"default": 0, "min": 0, "max": 8,
                                        "tooltip": "Extra model evaluations per guided step (re-predict at the same sigma after correcting). 0 = plain."}),
                "renoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Fraction of the noise replaced with fresh noise on guided steps. 0 = deterministic. Try 0.2-0.5 if the pattern looks pasted on."}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = CATEGORY

    def get_sampler(self, guide, inner_iters, renoise):
        import comfy.samplers
        fn = _make_sampler_function(guide, inner_iters, renoise)
        return (comfy.samplers.KSAMPLER(fn),)


# =============================================================================
# Model patch (any sampler, approximate)
# =============================================================================

class LumaGuideModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "guide": ("LUMA_GUIDE",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, guide):
        m = model.clone()

        def post_cfg(args):
            D = args["denoised"]
            base = args["model"]                           # BaseModel
            sigma = float(args["sigma"].flatten()[0])
            return guide.apply(D, sigma, base.model_sampling, base.latent_format, tag=" patch")

        m.set_model_sampler_post_cfg_function(post_cfg)
        return (m,)


# =============================================================================
# Calibration
# =============================================================================

def _load_folder_images(folder, max_images):
    import numpy as np
    from PIL import Image, ImageOps
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))[:max_images]
    out = []
    for f in files:
        im = ImageOps.exif_transpose(Image.open(os.path.join(folder, f))).convert("RGB")
        out.append(torch.from_numpy(np.asarray(im).astype("float32") / 255.0).unsqueeze(0))
    return out


def _prep_image(img, short_side, multiple=16):
    """[1,H,W,3] -> resized so the short side ~= short_side, dims multiple of `multiple`."""
    _, H, W, _ = img.shape
    s = short_side / min(H, W)
    nh = max(multiple, int(round(H * s / multiple)) * multiple)
    nw = max(multiple, int(round(W * s / multiple)) * multiple)
    x = img.movedim(-1, 1)
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    return x.movedim(1, -1).clamp(0, 1)


def _to_nchw(z):
    """VAE latent [1,C,H,W] or [1,C,T,H,W] -> [C,H,W] (first frame)."""
    if z.ndim == 5:
        z = z[:, :, 0]
    return z[0]


class LumaGuideCalibrateProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Needed for its latent format (normalization). Not run."}),
                "vae": ("VAE",),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64,
                                       "tooltip": "Short side for calibration encodes."}),
                "fit_blur": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.25,
                                       "tooltip": "Latent px. Fit the probe on blurred data, since it is only used on blurred luma."}),
                "ridge": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0005}),
                "augment": ("BOOLEAN", {"default": True,
                                        "tooltip": "Also encode darker/brighter gamma variants of each image. Adds brightness variety at equal content."}),
                "save_name": ("STRING", {"default": "krea2_luma_probe"}),
                "image_folder": ("STRING", {"default": "", "tooltip": "Optional folder of calibration images (used together with or instead of the images input)."}),
                "max_images": ("INT", {"default": 32, "min": 1, "max": 512}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Batch of diverse COLOR images (photos, illustrations). 8-32 is plenty."}),
            },
        }

    RETURN_TYPES = ("LUMA_PROBE", "IMAGE", "STRING")
    RETURN_NAMES = ("probe", "diagnostic", "report")
    FUNCTION = "calibrate"
    CATEGORY = CATEGORY

    def calibrate(self, model, vae, resolution, fit_blur, ridge, augment, save_name, image_folder,
                  max_images, images=None):
        latent_format = model.get_model_object("latent_format")

        imgs = []
        if images is not None:
            imgs += [images[i:i + 1] for i in range(min(images.shape[0], max_images))]
        if image_folder.strip():
            imgs += _load_folder_images(image_folder.strip(), max_images)
        if not imgs:
            raise ValueError("LumaGuide Calibrate: connect `images` or set `image_folder`.")

        latents, lumas = [], []
        variants = (1.0, 0.6, 1.7) if augment else (1.0,)
        pbar = comfy.utils.ProgressBar(len(imgs) * len(variants))
        for img in imgs:
            base = _prep_image(img.float(), resolution)
            for g in variants:
                im = base.clamp(1e-6, 1).pow(g)
                # Encode ONE image at a time. Krea2's VAE is a video VAE: a batch
                # of images would be encoded as frames of one clip and get mixed
                # by the temporal compression.
                raw = vae.encode(im[:, :, :, :3])
                z = latent_format.process_in(raw.float())  # -> sampler space
                z = _to_nchw(z).cpu().float()
                C, h, w = z.shape
                L = image_to_luma01(im)[0] * 2.0 - 1.0       # -1..1
                L = F.interpolate(L[None, None], size=(h, w), mode="area")[0, 0]
                latents.append(z)
                lumas.append(L)
                pbar.update(1)

        probe, rep = fit_probe(latents, lumas, fit_blur=fit_blur, ridge=ridge, name=save_name or "calibrated")

        # Evaluate: calibrated probe vs the built-in preview factors, at several blurs.
        lines = [f"LumaGuide probe '{probe.name}': {len(latents)} encodes, {rep['n']} pixels, "
                 f"{probe.channels} channels, latent format {type(latent_format).__name__}"]
        preview = None
        if getattr(latent_format, "latent_rgb_factors", None) is not None:
            preview = LumaProbe.from_preview_factors(latent_format.latent_rgb_factors,
                                                     getattr(latent_format, "latent_rgb_factors_bias", None))
        for bl in (0.0, fit_blur, 3.0):
            ev = probe_r2(probe, latents, lumas, bl)
            line = f"  blur {bl:>4.2f}px  calibrated R2={ev['r2']:.3f} corr2={ev['corr2']:.3f}"
            if preview is not None:
                evp = probe_r2(preview, latents, lumas, bl)
                line += f" | preview R2={evp['r2']:.3f} corr2={evp['corr2']:.3f} truth/probe scale={evp['scale']:.2f}"
            lines.append(line)
        lines.append(f"  w.a = {probe.wa():.3f} (should be ~R2 at fit blur; very small = unreliable direction)")
        lines.append("  Rule of thumb: corr2 > 0.9 at blur 1-3 px is good. < 0.75 means the linear "
                     "probe is too crude for this VAE.")
        report = "\n".join(lines)
        print(report)

        if save_name.strip():
            path = os.path.join(PROBE_DIR, save_name.strip() + ".json")
            probe.info["report"] = report
            with open(path, "w") as f:
                json.dump(probe.to_dict(), f, indent=1)
            print(f"[LumaGuide] saved probe to {path}")

        # Diagnostic: truth | calibrated | preview (first image), upsampled.
        z0, L0 = latents[0], lumas[0]
        cols = [L0, probe_luma(z0[None], probe)[0]]
        if preview is not None:
            cols.append(probe_luma(z0[None], preview)[0])
        h, w = L0.shape
        tiles = [F.interpolate(((c + 1) / 2).clamp(0, 1)[None, None], size=(h * 8, w * 8), mode="nearest")[0, 0]
                 for c in cols]
        diag = torch.cat(tiles, dim=1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        return (probe, diag, report)


class LumaGuideLoadProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"probe_file": (_list_probes(),)}}

    RETURN_TYPES = ("LUMA_PROBE", "STRING")
    RETURN_NAMES = ("probe", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, probe_file):
        path = os.path.join(PROBE_DIR, probe_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LumaGuide: no probe file {path}. Run LumaGuide Calibrate Probe first.")
        with open(path) as f:
            d = json.load(f)
        p = LumaProbe.from_dict(d)
        return (p, p.info.get("report", ""))


# =============================================================================
# Check: how well does a decoded image follow the map?
# =============================================================================

class LumaGuideCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "guide": ("LUMA_GUIDE",),
                "blur_px": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 128.0, "step": 1.0,
                                      "tooltip": "Image pixels. Compare at this blur (the 'squint' scale)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("comparison", "report")
    FUNCTION = "check"
    CATEGORY = CATEGORY

    def check(self, image, guide, blur_px):
        cfg = guide.cfg
        L = image_to_luma01(image)                                  # [B,H,W]
        B, H, W = L.shape
        T = resize_map(cfg.target01, H, W, cfg.fit)
        if T.shape[0] != B:
            T = T[torch.arange(B) % T.shape[0]]
        GL = gaussian_blur(L, blur_px)
        GT = gaussian_blur(T, blur_px)
        lines = []
        for i in range(B):
            a = GL[i].flatten() - GL[i].mean()
            b = GT[i].flatten() - GT[i].mean()
            r = float((a @ b) / max(float(a.norm() * b.norm()), 1e-12))
            lines.append(f"image {i}: corr(blurred luma, blurred map) = {r:.3f}")
        report = "\n".join(lines)
        # Panels: blurred map | blurred result luma | result (for eyeballing)
        def gray3(x):
            return x.clamp(0, 1).unsqueeze(-1).repeat(1, 1, 1, 3)
        comp = torch.cat([gray3(GT), gray3(GL), image[..., :3]], dim=2)
        return (comp, report)


NODE_CLASS_MAPPINGS = {
    "LumaGuideSettings": LumaGuideSettings,
    "LumaGuideSampler": LumaGuideSampler,
    "LumaGuideModelPatch": LumaGuideModelPatch,
    "LumaGuideCalibrateProbe": LumaGuideCalibrateProbe,
    "LumaGuideLoadProbe": LumaGuideLoadProbe,
    "LumaGuideCheck": LumaGuideCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LumaGuideSettings": "LumaGuide Settings (luma map)",
    "LumaGuideSampler": "LumaGuide Sampler (exact)",
    "LumaGuideModelPatch": "LumaGuide Model Patch (any sampler)",
    "LumaGuideCalibrateProbe": "LumaGuide Calibrate Probe",
    "LumaGuideLoadProbe": "LumaGuide Load Probe",
    "LumaGuideCheck": "LumaGuide Check Result",
}
