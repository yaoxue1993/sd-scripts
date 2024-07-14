import argparse
import glob
import math
import os
from typing import List, Optional, Tuple, Union

import torch
from safetensors.torch import save_file
from accelerate import Accelerator

from library import sd3_models, sd3_utils, train_util
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

# from transformers import CLIPTokenizer
# from library import model_util
# , sdxl_model_util, train_util, sdxl_original_unet
# from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from .sdxl_train_util import match_mixed_precision


def load_target_model(
    model_type: str,
    args: argparse.Namespace,
    state_dict: dict,
    accelerator: Accelerator,
    attn_mode: str,
    model_dtype: Optional[torch.dtype],
    device: Optional[torch.device],
) -> Union[
    sd3_models.MMDiT,
    Optional[sd3_models.SDClipModel],
    Optional[sd3_models.SDXLClipG],
    Optional[sd3_models.T5XXLModel],
    sd3_models.SDVAE,
]:
    loading_device = device if device is not None else (accelerator.device if args.lowram else "cpu")

    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            if model_type == "mmdit":
                model = sd3_utils.load_mmdit(state_dict, attn_mode, model_dtype, loading_device)
            elif model_type == "clip_l":
                model = sd3_utils.load_clip_l(state_dict, args.clip_l, attn_mode, model_dtype, loading_device)
            elif model_type == "clip_g":
                model = sd3_utils.load_clip_g(state_dict, args.clip_g, attn_mode, model_dtype, loading_device)
            elif model_type == "t5xxl":
                model = sd3_utils.load_t5xxl(state_dict, args.t5xxl, attn_mode, model_dtype, loading_device)
            elif model_type == "vae":
                model = sd3_utils.load_vae(state_dict, args.vae, model_dtype, loading_device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # work on low-ram device: models are already loaded on accelerator.device, but we ensure they are on device
            if args.lowram:
                model = model.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    return model


def save_models(
    ckpt_path: str,
    mmdit: sd3_models.MMDiT,
    vae: sd3_models.SDVAE,
    clip_l: sd3_models.SDClipModel,
    clip_g: sd3_models.SDXLClipG,
    t5xxl: Optional[sd3_models.T5XXLModel],
    sai_metadata: Optional[dict],
    save_dtype: Optional[torch.dtype] = None,
):
    r"""
    Save models to checkpoint file. Only supports unified checkpoint format.
    """

    state_dict = {}

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            if save_dtype is not None:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    update_sd("model.diffusion_model.", mmdit.state_dict())
    update_sd("first_stage_model.", vae.state_dict())

    if clip_l is not None:
        update_sd("text_encoders.clip_l.", clip_l.state_dict())
    if clip_g is not None:
        update_sd("text_encoders.clip_g.", clip_g.state_dict())
    if t5xxl is not None:
        update_sd("text_encoders.t5xxl.", t5xxl.state_dict())

    save_file(state_dict, ckpt_path, metadata=sai_metadata)


def save_sd3_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    clip_l: sd3_models.SDClipModel,
    clip_g: sd3_models.SDXLClipG,
    t5xxl: Optional[sd3_models.T5XXLModel],
    mmdit: sd3_models.MMDiT,
    vae: sd3_models.SDVAE,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, sd3=mmdit.model_type
        )
        save_models(ckpt_file, mmdit, vae, clip_l, clip_g, t5xxl, sai_metadata, save_dtype)

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd3_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    clip_l: sd3_models.SDClipModel,
    clip_g: sd3_models.SDXLClipG,
    t5xxl: Optional[sd3_models.T5XXLModel],
    mmdit: sd3_models.MMDiT,
    vae: sd3_models.SDVAE,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, sd3=mmdit.model_type
        )
        save_models(ckpt_file, mmdit, vae, clip_l, clip_g, t5xxl, sai_metadata, save_dtype)

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


def add_sd3_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cache_text_encoder_outputs", action="store_true", help="cache text encoder outputs / text encoderの出力をキャッシュする"
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        type=int,
        default=None,
        help="text encoder batch size (default: None, use dataset's batch size)"
        + " / text encoderのバッチサイズ（デフォルト: None, データセットのバッチサイズを使用）",
    )
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment / safetensorsのmmapロードを無効にする。WSL環境等でモデル読み込みを高速化できる",
    )

    parser.add_argument(
        "--clip_l",
        type=str,
        required=False,
        help="CLIP-L model path. if not specified, use ckpt's state_dict / CLIP-Lモデルのパス。指定しない場合はckptのstate_dictを使用",
    )
    parser.add_argument(
        "--clip_g",
        type=str,
        required=False,
        help="CLIP-G model path. if not specified, use ckpt's state_dict / CLIP-Gモデルのパス。指定しない場合はckptのstate_dictを使用",
    )
    parser.add_argument(
        "--t5xxl",
        type=str,
        required=False,
        help="T5-XXL model path. if not specified, use ckpt's state_dict / T5-XXLモデルのパス。指定しない場合はckptのstate_dictを使用",
    )
    parser.add_argument(
        "--save_clip", action="store_true", help="save CLIP models to checkpoint / CLIPモデルをチェックポイントに保存する"
    )
    parser.add_argument(
        "--save_t5xxl", action="store_true", help="save T5-XXL model to checkpoint / T5-XXLモデルをチェックポイントに保存する"
    )

    parser.add_argument(
        "--t5xxl_device",
        type=str,
        default=None,
        help="T5-XXL device. if not specified, use accelerator's device / T5-XXLデバイス。指定しない場合はacceleratorのデバイスを使用",
    )
    parser.add_argument(
        "--t5xxl_dtype",
        type=str,
        default=None,
        help="T5-XXL dtype. if not specified, use default dtype (from mixed precision) / T5-XXL dtype。指定しない場合はデフォルトのdtype（mixed precisionから）を使用",
    )

    # copy from Diffusers
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def verify_sdxl_training_args(args: argparse.Namespace, supportTextEncoderCaching: bool = True):
    assert not args.v2, "v2 cannot be enabled in SDXL training / SDXL学習ではv2を有効にすることはできません"
    if args.v_parameterization:
        logger.warning("v_parameterization will be unexpected / SDXL学習ではv_parameterizationは想定外の動作になります")

    if args.clip_skip is not None:
        logger.warning("clip_skip will be unexpected / SDXL学習ではclip_skipは動作しません")

    # if args.multires_noise_iterations:
    #     logger.info(
    #         f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET}, but noise_offset is disabled due to multires_noise_iterations / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されていますが、multires_noise_iterationsが有効になっているためnoise_offsetは無効になります"
    #     )
    # else:
    #     if args.noise_offset is None:
    #         args.noise_offset = DEFAULT_NOISE_OFFSET
    #     elif args.noise_offset != DEFAULT_NOISE_OFFSET:
    #         logger.info(
    #             f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET} / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されています"
    #         )
    #     logger.info(f"noise_offset is set to {args.noise_offset} / noise_offsetが{args.noise_offset}に設定されました")

    assert (
        not hasattr(args, "weighted_captions") or not args.weighted_captions
    ), "weighted_captions cannot be enabled in SDXL training currently / SDXL学習では今のところweighted_captionsを有効にすることはできません"

    if supportTextEncoderCaching:
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
            logger.warning(
                "cache_text_encoder_outputs is enabled because cache_text_encoder_outputs_to_disk is enabled / "
                + "cache_text_encoder_outputs_to_diskが有効になっているためcache_text_encoder_outputsが有効になりました"
            )


def sample_images(*args, **kwargs):
    return train_util.sample_images_common(SdxlStableDiffusionLongPromptWeightingPipeline, *args, **kwargs)


class Sd3LatentsCachingStrategy(train_util.LatentsCachingStrategy):
    SD3_LATENTS_NPZ_SUFFIX = "_sd3.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.vae = None

    def set_vae(self, vae: sd3_models.SDVAE):
        self.vae = vae

    def get_image_size_from_image_absolute_path(self, absolute_path: str) -> Tuple[Optional[int], Optional[int]]:
        npz_file = glob.glob(os.path.splitext(absolute_path)[0] + "_*" + Sd3LatentsCachingStrategy.SD3_LATENTS_NPZ_SUFFIX)
        if len(npz_file) == 0:
            return None, None
        w, h = os.path.splitext(npz_file[0])[0].split("_")[-2].split("x")
        return int(w), int(h)

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + Sd3LatentsCachingStrategy.SD3_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        expected_latents_size = (bucket_reso[1] // 8, bucket_reso[0] // 8)  # bucket_reso is (W, H)

        try:
            npz = np.load(npz_path)
            if npz["latents"].shape[1:3] != expected_latents_size:
                return False

            if flip_aug:
                if "latents_flipped" not in npz:
                    return False
                if npz["latents_flipped"].shape[1:3] != expected_latents_size:
                    return False

            if alpha_mask:
                if "alpha_mask" not in npz:
                    return False
                if npz["alpha_mask"].shape[0:2] != (bucket_reso[1], bucket_reso[0]):
                    return False
            else:
                if "alpha_mask" in npz:
                    return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def cache_batch_latents(self, image_infos: List[train_util.ImageInfo], flip_aug: bool, alpha_mask: bool, random_crop: bool):
        img_tensor, alpha_masks, original_sizes, crop_ltrbs = train_util.load_images_and_masks_for_caching(
            image_infos, alpha_mask, random_crop
        )
        img_tensor = img_tensor.to(device=self.vae.device, dtype=self.vae.dtype)

        with torch.no_grad():
            latents_tensors = self.vae.encode(img_tensor).to("cpu")
        if flip_aug:
            img_tensor = torch.flip(img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = self.vae.encode(img_tensor).to("cpu")
        else:
            flipped_latents = [None] * len(latents_tensors)

        # for info, latents, flipped_latent, alpha_mask in zip(image_infos, latents_tensors, flipped_latents, alpha_masks):
        for i in range(len(image_infos)):
            info = image_infos[i]
            latents = latents_tensors[i]
            flipped_latent = flipped_latents[i]
            alpha_mask = alpha_masks[i]
            original_size = original_sizes[i]
            crop_ltrb = crop_ltrbs[i]

            if self.cache_to_disk:
                kwargs = {}
                if flipped_latent is not None:
                    kwargs["latents_flipped"] = flipped_latent.float().cpu().numpy()
                if alpha_mask is not None:
                    kwargs["alpha_mask"] = alpha_mask.float().cpu().numpy()
                np.savez(
                    info.latents_npz,
                    latents=latents.float().cpu().numpy(),
                    original_size=np.array(original_size),
                    crop_ltrb=np.array(crop_ltrb),
                    **kwargs,
                )
            else:
                info.latents = latents
                if flip_aug:
                    info.latents_flipped = flipped_latent
                info.alpha_mask = alpha_mask

        if not train_util.HIGH_VRAM:
            clean_memory_on_device(self.vae.device)


# region Diffusers


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps)

        sigmas = timesteps / self.config.num_train_timesteps
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        timesteps = sigmas * self.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator)

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility

        # if self.config.prediction_type == "vector_field":

        denoised = sample - model_output * sigma
        # 2. Convert to an ODE derivative
        derivative = (sample - denoised) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps


# endregion
