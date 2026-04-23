"""
CTFDDIMSampler — Coarse-to-Fine DDIM Sampler with temporal proxy prompt swapping.

Key mechanics:
  - Phase 1 (layout):  progress in [0, layout_end]                  -> CLIP P1
  - Phase 2 (content): progress in (layout_end, content_end - w]    -> CLIP P2
  - Blend window:      progress in (content_end - w, content_end+w) -> 2x forward + cosine blend
  - Phase 3 (style):   progress in [content_end + w, 1]             -> T5 P3 via ArtDapter

  Additionally, once we cross `content_idx`, the sampler anchors to the content
  pred_x0 estimate and softly pulls subsequent P3 latents back toward that
  anchor (SDEdit-lite), controlled by `preserve_strength`. This keeps the
  global layout stable while the style branch refines textures.
"""
import math

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import noise_like
from .custom_ddim import CustomDDIMSampler


class CTFDDIMSampler(CustomDDIMSampler):
    """DDIM sampler for Temporal Proxy Prompt + window blend + content-anchor preserve."""

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               dynamic_threshold=None,
               ucg_schedule=None,
               global_strength=None,
               # CTF params
               layout_end=0.30,
               content_end=0.65,
               blend_window=0.08,
               preserve_strength=0.30,
               **kwargs):
        """Override sample() to pass CTF thresholds + blend/preserve hyperparams down."""
        if conditioning is not None and isinstance(conditioning, dict):
            for k, v in conditioning.items():
                if isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'shape'):
                    cbs = v[0].shape[0]
                    if cbs != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
                    break

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for CTF DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(
            conditioning, size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask, x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            dynamic_threshold=dynamic_threshold,
            ucg_schedule=ucg_schedule,
            global_strength=global_strength,
            layout_end=layout_end,
            content_end=content_end,
            blend_window=blend_window,
            preserve_strength=preserve_strength,
        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None,
                      quantize_denoised=False,
                      mask=None, x0=None, img_callback=None,
                      log_every_t=100, temperature=1., noise_dropout=0.,
                      score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      dynamic_threshold=None, ucg_schedule=None,
                      global_strength=None,
                      layout_end=0.30, content_end=0.65,
                      blend_window=0.08, preserve_strength=0.30):
        """Custom DDIM sampling loop for Temporal Proxy Prompt + blend + preserve."""
        device = self.model.betas.device
        b = shape[0]

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(
                min(timesteps / self.ddim_timesteps.shape[0], 1)
                * self.ddim_timesteps.shape[0]
            ) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {
            'x_inter': [img],
            'pred_x0': [img],
            'stage_structure': None,
            'stage_content': None,
            'stage_style': None,
        }

        time_range = (
            reversed(range(0, timesteps))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running CTF DDIM sampling: {total_steps} steps | "
              f"layout_end={layout_end}, content_end={content_end}, "
              f"blend_window={blend_window}, preserve_strength={preserve_strength}")

        iterator = tqdm(time_range, disable=False, total=total_steps)

        # Phase boundary indices (inclusive upper index of each phase).
        layout_idx = int(layout_end * (total_steps - 1))
        content_idx = int(content_end * (total_steps - 1))

        # Anchor latent (pred_x0 at end of content phase) for SDEdit-lite preserve.
        anchor_x0 = None

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Descriptive label on progress bar
            if i <= layout_idx:
                iterator.set_description("Phase 1: Layout")
            elif i <= content_idx:
                iterator.set_description("Phase 2: Content")
            else:
                iterator.set_description("Phase 3: Style")

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            ug_scale = unconditional_guidance_scale
            if ucg_schedule is not None:
                assert len(ucg_schedule) == total_steps
                ug_scale = ucg_schedule[i]

            img, pred_x0 = self.p_sample_ddim(
                img, cond, ts, index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=ug_scale,
                unconditional_conditioning=unconditional_conditioning,
                dynamic_threshold=dynamic_threshold,
                global_strength=global_strength,
                # CTF params
                step_idx=i,
                total_steps=total_steps,
                layout_end=layout_end,
                content_end=content_end,
                blend_window=blend_window,
                preserve_strength=preserve_strength,
                anchor_x0=anchor_x0,
            )

            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

            # Snapshot at phase boundaries (for Streamlit 3-stage visualization).
            if i == layout_idx:
                intermediates['stage_structure'] = pred_x0.clone()
                print(f" [stage] Layout locked at step {i}")
            elif i == content_idx:
                intermediates['stage_content'] = pred_x0.clone()
                # Anchor for P3 preserve mechanism — use the content pred_x0
                # as the "structural blueprint" of the image.
                anchor_x0 = pred_x0.clone()
                print(f" [stage] Content locked at step {i} (anchor captured)")
            elif i == total_steps - 1:
                intermediates['stage_style'] = pred_x0.clone()
                print(f" [stage] Style finalized at step {i}")

        return img, intermediates

    # ─────────────────────── Single-step logic ──────────────────────

    @torch.no_grad()
    def _apply_model_with_phase(self, x, t, c, uc, cfg_scale, global_strength,
                                progress, layout_end, content_end, blend_window):
        """
        Run U-Net forward (with CFG) according to which CTF phase `progress` is in.
        Handles the blend window by running both content and style branches
        and linearly combining them with a cosine-ramped weight.

        Returns: model_output (the CFG-combined epsilon / v prediction).
        """
        blend_low = content_end - blend_window
        blend_high = content_end + blend_window
        in_blend = (blend_window > 0) and (blend_low < progress < blend_high)

        use_cfg = (uc is not None) and (cfg_scale != 1.)

        def run(cond_dict, uncond_dict):
            if not use_cfg:
                return self.model.apply_model(x, t, cond_dict, global_strength)
            model_t = self.model.apply_model(x, t, cond_dict, global_strength)
            model_u = self.model.apply_model(x, t, uncond_dict, global_strength)
            return model_u + cfg_scale * (model_t - model_u)

        if in_blend:
            # Cosine-ramped blend from content (0) -> style (1) across the window.
            x_norm = (progress - blend_low) / (2.0 * blend_window)  # ∈ (0, 1)
            w = 0.5 * (1.0 - math.cos(math.pi * x_norm))

            c_cont = {'c_crossattn': c['c_content']}
            c_styl = {'c_style_raw': c['c_style']}
            uc_cont = {'c_crossattn': uc['c_content']} if use_cfg else None
            uc_styl = {'c_style_raw': uc['c_style']} if use_cfg else None

            out_cont = run(c_cont, uc_cont)
            out_styl = run(c_styl, uc_styl)
            return (1.0 - w) * out_cont + w * out_styl

        # Outside the blend window -> single branch.
        if progress <= layout_end:
            active_c = {'c_crossattn': c['c_layout']}
            active_uc = {'c_crossattn': uc['c_layout']} if use_cfg else None
        elif progress <= blend_low:
            active_c = {'c_crossattn': c['c_content']}
            active_uc = {'c_crossattn': uc['c_content']} if use_cfg else None
        else:
            active_c = {'c_style_raw': c['c_style']}
            active_uc = {'c_style_raw': uc['c_style']} if use_cfg else None

        return run(active_c, active_uc)

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index,
                      repeat_noise=False, use_original_steps=False,
                      quantize_denoised=False, temperature=1.,
                      noise_dropout=0., score_corrector=None,
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      dynamic_threshold=None, global_strength=None,
                      # CTF params
                      step_idx=0, total_steps=50,
                      layout_end=0.30, content_end=0.65,
                      blend_window=0.08, preserve_strength=0.30,
                      anchor_x0=None):
        """Single DDIM step with temporal proxy prompt swapping, blend window, and preserve."""
        b, *_, device = *x.shape, x.device
        progress = step_idx / max(total_steps - 1, 1)

        # ── U-Net forward (phase-aware, with optional blend) ──
        model_output = self._apply_model_with_phase(
            x=x, t=t, c=c, uc=unconditional_conditioning,
            cfg_scale=unconditional_guidance_scale,
            global_strength=global_strength,
            progress=progress,
            layout_end=layout_end,
            content_end=content_end,
            blend_window=blend_window,
        )

        # ── Convert to epsilon ──
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # ── DDIM update (same formulas as CustomDDIMSampler) ──
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )

        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # ── Preserve content anchor (SDEdit-lite) in P3, outside the blend window ──
        # We renoise the anchor_x0 to the same noise level as x_prev (via a_prev)
        # and softly pull x_prev toward it. Skip at index == 0 (final step) since
        # a_prev ≈ 1 there -> preservation would overwrite the style refinement.
        if (
            anchor_x0 is not None
            and preserve_strength > 0.
            and index > 0
            and progress > (content_end + blend_window)
        ):
            anchor_noise = torch.randn_like(anchor_x0)
            x_anchor_noisy = a_prev.sqrt() * anchor_x0 + (1. - a_prev).sqrt() * anchor_noise
            x_prev = (1. - preserve_strength) * x_prev + preserve_strength * x_anchor_noisy

        return x_prev, pred_x0
