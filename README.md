# ComfyUI-LumaGuide

Training-free, QR-Code-Monster-style luminance guidance for flow models. Built for Krea2, and written to be model-agnostic.

You give it a luminance map (spiral, text, checkerboard, a face, whatever). During sampling it nudges the model's own clean-image estimate so the image's **low-frequency brightness** follows the map. The transformer stays free to invent color, texture and content inside those constraints. No LoRA, no ControlNet, no training of the generator.

---

## How it works (short version)

A luminance map is a linear measurement of the image: `y = Blur(Luma(Decode(z)))`. That makes this a linear inverse problem, which has well-known training-free solutions.

1. **Linear probe instead of the VAE.** Luma of the decoded image is approximated per latent pixel as `w · z + b`, where `w` is one number per latent channel (16 for Krea2). This avoids decoding during sampling, so a guided step costs essentially nothing extra.
2. **Null-space projection on x0 (DDNM-style).** Each step, the model predicts a clean latent `D`. LumaGuide measures the blurred luma of `D`, compares it with the blurred map, and adds a correction along a single latent direction `a`, scaled so the probe sees exactly the requested change. Everything the probe can't see (color, texture, semantics) stays the model's decision.
3. **Coarse-to-fine schedule.** Sampling settles low frequencies first. The guide enforces very coarse brightness early (`blur_start`) and progressively finer scales (`blur_end`), then stops at `end_percent` so the model owns fine detail. This is the principled version of the old ControlNet "end percent ~0.7" trick.
4. **Soft constraints.** `match=relative` keeps the image's own mean brightness (night scene or beach is still the model's choice) and only enforces the pattern. `tolerance` is a dead-band: luma that is already "dark enough" or "bright enough" is left alone. That's where the illusion look comes from, instead of a stamped overlay.

Related papers: DDNM (null-space projection; colorization is the "luma fixed" case), PnP-Flow and FlowChef (training-free steering of flow models without backprop through the transformer), Factorized Diffusion (holding one image component fixed while generating the rest), and Dieleman's "Diffusion is spectral autoregression" (coarse-to-fine).

---

## Install

Copy the `ComfyUI-LumaGuide` folder into `ComfyUI/custom_nodes/` and restart. It has no extra dependencies. Nodes appear under the **LumaGuide** category.

---

## Nodes

| Node | What it does |
|---|---|
| **LumaGuide Settings (luma map)** | Takes the map plus all the knobs and outputs a `LUMA_GUIDE`. Also outputs the processed map so you can preview it. |
| **LumaGuide Sampler (exact)** | Outputs a `SAMPLER` for SamplerCustom / SamplerCustomAdvanced. With the guide off it is mathematically identical to `euler`. **Use this one.** |
| **LumaGuide Model Patch (any sampler)** | Applies the same guide through ComfyUI's post-CFG hook, so it works with KSampler and any sampler. Approximate: most samplers only move part of the way toward x0 per step, so the effective strength is lower and depends on the sampler. Don't combine it with the LumaGuide Sampler (that double-applies). |
| **LumaGuide Calibrate Probe** | Fits the probe for your VAE from a handful of images. Saves a JSON to `probes/`. Recommended once per VAE. |
| **LumaGuide Load Probe** | Loads a saved probe. Click refresh in the browser after calibrating so the dropdown sees the new file. |
| **LumaGuide Check Result** | Compares a decoded image's blurred luma with the blurred map and reports a correlation. Use it to tune settings with numbers instead of squinting. |

---

## Quick start (Krea2)

**Step 1: calibrate (one time, about a minute)**

```
Load Krea2 model ─┐
Load VAE ─────────┼─> LumaGuide Calibrate Probe  ──> (report: read it)
images / folder ──┘        save_name = krea2_luma_probe
```

- Use 8-32 diverse **color** images (photos, illustrations, mixed lighting). Set `image_folder` to a folder path, or feed an image batch.
- The model input is only used for its latent normalization. Nothing is sampled.
- Read the report in the console or the `report` output:

```
blur 0.00px  calibrated R2=… corr2=… | preview R2=… corr2=… truth/probe scale=…
blur 1.00px  ...
blur 3.00px  ...
```

  - **corr2 > 0.9 at 1-3 px**: the linear probe is good, go ahead.
  - **corr2 0.75-0.9**: usable, expect a softer pattern.
  - **corr2 < 0.75**: the linear approximation is too crude for this VAE. Tell me; the fallback is decoder-gradient guidance (see Roadmap).
  - The `preview` columns show how ComfyUI's built-in preview factors do on the same data. If they're close to the calibrated numbers you could skip calibration, but the calibrated probe also gives a better correction direction (see "Probe direction" below).
- The `diagnostic` image shows true luma | calibrated probe | preview probe for the first image. They should look like the same blurry grayscale image.

**Step 2: generate**

```
Load Image (your pattern) ─> LumaGuide Settings ──guide──> LumaGuide Sampler ──sampler──┐
LumaGuide Load Probe ──probe──┘                                                        │
                                                                                       v
Noise + Guider (CFGGuider/BasicGuider with Krea2) + your turbo SIGMAS + latent ─> SamplerCustomAdvanced ─> VAE Decode
                                                                                                                  │
                                                         LumaGuide Check Result (image + guide) <──────────────────┘
```

If you leave the probe unconnected, it falls back to the model's preview RGB factors. That's fine for a first look.

---

## Starting settings

**Krea2 turbo, 8 steps**

| Setting | Value | Why |
|---|---|---|
| strength | 0.8 (0.6-1.0) | Fraction of the remaining luma error removed per guided step. 1.0 = hard projection. |
| start / end_percent | 0.0 / 0.7 | With Krea2's shift (1.15, exponential), 0.7 ends around sigma 0.58, which is about 6 of 8 steps. |
| taper | 0.3 | Fades out so there's no visible "release" seam. |
| match | relative | |
| contrast | 0.35 (0.25-0.5) | Brightness swing between map-black and map-white. Higher = more visible illusion. |
| tolerance | 0.05 (0-0.15) | Higher = more freedom, subtler and more "found" pattern. |
| blur_schedule | coarse_to_fine | |
| blur_start / blur_end | 8 / 1.0 latent px | Latent px x 8 = image px for Krea2. For a bolder, simpler pattern, use blur_end 1.5-2. |
| fit | crop | |
| map_pre_blur | 0-4 image px | Softens jaggy masks. |
| LumaGuide Sampler | inner_iters 0, renoise 0 | Then try inner_iters 1 (costs one extra model call per guided step) or renoise 0.2-0.4. |

Turn on `verbose` to see, per step, the sigma, strength, blur width, luma residual and latent change. If `luma_res_rms` doesn't drop across steps, the guide isn't biting (raise strength, contrast or end_percent).

---

## Troubleshooting

| Symptom | Try |
|---|---|
| Pattern invisible | Raise contrast (0.45+), raise end_percent (0.8), lower blur_end. Check that `luma_res_rms` falls. Calibrate. |
| Pattern looks pasted on or stamped | Raise tolerance (0.1), raise blur_end (1.5-2), lower end_percent, try renoise 0.3. |
| Gray haze or washed-out color | Use a calibrated probe (better direction). Lower contrast. Lower max_delta to about 1.0. |
| Blotchy, mottled texture | Lower strength, lower end_percent, raise blur_end. |
| Global layout ignores the pattern | Raise blur_start (12-16), make sure start_percent is 0, try inner_iters 1. |
| Works with the Sampler but weak with the Model Patch | Expected. The patch is approximate; raise strength there or use the Sampler. |
| Error about channel mismatch | The probe was calibrated for a different VAE or latent format. |

---

## Other models

Nothing here is Krea2-specific.

- Any model whose latent format has `latent_rgb_factors` works out of the box (most do), and any model can be calibrated.
- 4D latents (Flux style) and 5D latents (Wan21 style: Krea2, Qwen-Image) are both handled.
- The LumaGuide Sampler uses the exact flow update for flow models (`CONST` sampling, i.e. everything Flux/Qwen/Krea-like) and the k-diffusion update for EPS/V-pred models (SDXL), so it also works on old models.
- Calibrate once per **VAE**, not per model. Models sharing a VAE share a probe.
- Window percents go through each model's own schedule shift, so the same end_percent means roughly the same stage of generation across models.

---

## Probe direction (why calibrating helps beyond accuracy)

The correction is added along a latent direction `a`, scaled by `1/(w·a)` so the probe sees exactly the requested luma change.

- **Preview probe:** `a = w`. This is the minimum-norm change, the pure DDNM choice. It's mathematically smallest, but it's not necessarily how real latents change when images get brighter, so it can look hazy.
- **Calibrated probe:** `a = cov(z, luma) / var(luma)`, measured from real encodes. This is how latents actually move, on average, when real images get brighter or darker, so corrections stay closer to the data distribution. For an ordinary least-squares fit, `w·a` equals the fit R², which the report prints.

---

## What has and hasn't been verified

These were tested against current ComfyUI source (Nov 2026) on CPU, with a synthetic denoiser:

- The sampler with strength 0 matches ComfyUI `euler` to about 1e-6.
- Through the real `KSAMPLER.sample` path, the guide pulls a random sample's blurred luma to correlation ~0.95 with the map.
- The post-CFG patch path runs through the real `cfg_function`.
- A strength-1 projection closes most of the band-limited error.
- Calibration recovers a known linear map exactly, and probe save/load round-trips.
- 4D and 5D latents both work.

**Not verified:** a run with real Krea2 weights. There was no GPU or model access where this was written. So the defaults are reasoned starting points, not tuned values. The calibration report is the first real test: it tells you within a minute whether the linear-probe approach holds for Krea2's VAE.

---

## Roadmap, if the linear probe isn't enough

1. **Decoder-gradient mode.** Measure luma through a tiny differentiable decoder (the TAE preview decoder) and take a gradient step on z0 only, FlowChef-style (no backprop through the transformer). Use it if the calibration corr2 is low.
2. **Attention-injection assist.** Inject self-attention from a parallel denoising of the noised map into Krea2's composition blocks (14-27) for the first steps, to push layout harder than latent corrections can.
3. **Multi-view illusions.** The same correction applied under a transform (flip, rotate) to combine with the flip-illusion nodes.

---

## Files

- `luma_core.py`: all the math (probe, blur, schedules, correction, calibration fit). No ComfyUI imports, heavily commented.
- `nodes.py`: ComfyUI node classes and the guided sampler.
- `probes/`: saved calibrations (JSON).
- `tests/test_lumaguide.py`: the mechanics tests. They need a ComfyUI checkout; see the top of the file.

Internal knob not exposed in the UI: `deblur_iters` (default 2) in `LumaGuideConfig`. These are Van Cittert iterations that make the *blurred* luma land on target instead of under-shooting. 0 = plain, softer. More = sharper but adds texture.
