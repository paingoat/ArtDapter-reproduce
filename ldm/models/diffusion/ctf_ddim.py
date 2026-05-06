"""
CTFDDIMSampler — Coarse-to-Fine DDIM Sampler.

Simplified version: no alpha injection or score blending.
Supports Phase Analysis via `no_style` flag in conditioning dict.

Phase Analysis modes:
  - no_style=True  → Upper blocks use content (structure only, no artistic style)
  - no_style=False → Upper blocks use style (full CTF with ArtDapter)
"""
import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import noise_like
from .custom_ddim import CustomDDIMSampler


def _set_no_style(cond, value: bool):
    """Set the no_style flag in a conditioning dict. Pass-through for non-dict cond."""
    if isinstance(cond, dict):
        return {**cond, 'no_style': value}
    return cond


class CTFDDIMSampler(CustomDDIMSampler):
    """
    DDIM sampler for CTF pipeline with Decoupled Context Routing.

    Supports Phase Analysis:
      - no_style=True  → structure-only sampling (Phase 1)
      - no_style=False → full CTF sampling with style (Phase 2)
    """

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
               no_style=False,
               **kwargs):
        """
        Override sample() to support Phase Analysis via no_style flag.

        Args:
            no_style: If True, Upper blocks use content instead of style
                      (structure-only mode for Phase Analysis).
        """
        # Validate conditioning batch size
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
            no_style=no_style,
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
                      no_style=False):
        """
        DDIM sampling loop with no_style flag injection.
        """
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

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = (
            reversed(range(0, timesteps))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        mode_str = "Structure-only (no_style)" if no_style else "Full CTF"
        print(f"Running CTF DDIM Sampling with {total_steps} timesteps, mode={mode_str}")

        iterator = tqdm(time_range, desc='CTF DDIM', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Inpainting mask support
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            # UCG schedule support
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
                no_style=no_style,
            )

            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index,
                      repeat_noise=False, use_original_steps=False,
                      quantize_denoised=False, temperature=1.,
                      noise_dropout=0., score_corrector=None,
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      dynamic_threshold=None, global_strength=None,
                      no_style=False):
        """
        Single DDIM denoising step with no_style flag injection.
        """
        b, *_, device = *x.shape, x.device

        # Inject no_style flag into conditioning dicts
        c_flagged = _set_no_style(c, no_style)
        uc_flagged = _set_no_style(unconditional_conditioning, no_style)

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c_flagged, global_strength)
        else:
            model_t = self.model.apply_model(x, t, c_flagged, global_strength)
            model_uncond = self.model.apply_model(x, t, uc_flagged, global_strength)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        # Predict epsilon
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # DDIM update
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

        # Current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # Direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0
