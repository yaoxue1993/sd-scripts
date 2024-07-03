import argparse

import torch
import copy
from library.device_utils import init_ipex, clean_memory_on_device
init_ipex()

from library import sd3_models, sd3_utils, sd3_train_utils, sdxl_model_util, sdxl_train_util, train_util
import train_network
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

class SD3NetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True
        self.is_sd3 = True

    def assert_extra_args(self, args, train_dataset_group):
        super().assert_extra_args(args, train_dataset_group)
        #sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        attn_mode = "xformers" if args.xformers else "torch"
        t5xxl_dtype = weight_dtype
        if args.t5xxl_dtype is not None:
            if args.t5xxl_dtype == "fp16":
                t5xxl_dtype = torch.float16
            elif args.t5xxl_dtype == "bf16":
                t5xxl_dtype = torch.bfloat16
            elif args.t5xxl_dtype == "fp32" or args.t5xxl_dtype == "float":
                t5xxl_dtype = torch.float32
            else:
                raise ValueError(f"unexpected t5xxl_dtype: {args.t5xxl_dtype}")
        t5xxl_device = accelerator.device if args.t5xxl_device is None else args.t5xxl_device
        (
            mmdit,
            clip_l,
            clip_g,
            t5xxl,
            vae
        ) = sd3_train_utils.load_target_model(
            args, 
            accelerator, 
            attn_mode,
            weight_dtype,
            t5xxl_device,
            t5xxl_dtype
            )

        text_encoders = [clip_l, clip_g] if t5xxl is None else [clip_l, clip_g, t5xxl]
        return "SD3", text_encoders, sd3_models.VAEWrapper(vae), mmdit

    def load_tokenizer(self, args):
        tokenizer = sd3_models.SD3Tokenizer(t5xxl=False)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            with torch.no_grad(), accelerator.autocast():
                dataset.cache_text_encoder_outputs_sd3(
                    tokenizers[0],
                    text_encoders,
                    (accelerator.device, accelerator.device, accelerator.device),
                    None,
                    (None, None, None),
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )
            accelerator.wait_for_everyone()

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if len(text_encoders) == 3:
            clip_l, clip_g, t5xxl = text_encoders
        else:
            clip_l, clip_g = text_encoders
            t5xxl = None

        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids_clip_l, input_ids_clip_g, input_ids_t5xxl = batch["input_ids"]
            with torch.set_grad_enabled(args.text_encoder_lr>0):
                # TODO support weighted captions
                # TODO support length > 75
                input_ids_clip_l = input_ids_clip_l.to(accelerator.device)
                input_ids_clip_g = input_ids_clip_g.to(accelerator.device)
                input_ids_t5xxl = input_ids_t5xxl.to(accelerator.device)

                # get text encoder outputs: outputs are concatenated
                context, pool = sd3_utils.get_cond_from_tokens(
                    input_ids_clip_l, input_ids_clip_g, input_ids_t5xxl, clip_l, clip_g, t5xxl
                )
        else:
            lg_out = batch["text_encoder_outputs1_list"]
            t5_out = batch["text_encoder_outputs2_list"]
            pool = batch["text_encoder_pool2_list"]
            context = torch.cat([lg_out, t5_out], dim=-2)
            # # verify that the text encoder outputs are correct
            # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
            #     args.max_token_length,
            #     batch["input_ids"].to(text_encoders[0].device),
            #     batch["input_ids2"].to(text_encoders[0].device),
            #     tokenizers[0],
            #     tokenizers[1],
            #     text_encoders[0],
            #     text_encoders[1],
            #     None if not args.full_fp16 else weight_dtype,
            # )
            # b_size = encoder_hidden_states1.shape[0]
            # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # logger.info("text encoder outputs verified")

        return context, pool

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
        latents = sd3_models.SDVAE.process_in(latents)
        
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        u = sd3_train_utils.compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )

        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=weight_dtype)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        # concat embeddings
        context, pool = text_conds
        with accelerator.autocast():
            model_pred = unet(noisy_model_input, timesteps, context=context, y=pool)

        model_pred = model_pred * (-sigmas) + noisy_model_input
        
        weighting = sd3_train_utils.compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        target = latents
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        pass
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sd3_train_utils.add_sd3_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SD3NetworkTrainer()
    trainer.train(args)
