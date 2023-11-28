import torch
import diffusers 
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel   
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
    ImagePipelineOutput
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
import torch.nn.functional as F

class MvFusion(diffusers.StableDiffusionControlNetPipeline):
    def __init__(self,
                vae: AutoencoderKL,
                text_encoder: CLIPTextModel,
                tokenizer: CLIPTokenizer,
                unet: UNet2DConditionModel,
                controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
                scheduler: KarrasDiffusionSchedulers,
                feature_extractor: CLIPImageProcessor,
                safety_checker = None,
                requires_safety_checker: bool = True,):
        StableDiffusionControlNetPipeline.__init__(self, vae=vae,
                                                         text_encoder=text_encoder,
                                                         tokenizer=tokenizer,
                                                         unet=unet,
                                                         controlnet=controlnet,
                                                         scheduler=scheduler,
                                                         safety_checker=None,
                                                         feature_extractor=feature_extractor)
        self.register_modules(
            feature_extractor = feature_extractor, controlnet=controlnet,
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=None,
        )
    
    def __call__(self, 
                 prompt_embeds: Union[str, List[str]] = None, #(positive negative prompt)
                 latents: Optional[torch.FloatTensor] = None,
                 control_image = None,
                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                 controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
                 control_guidance_start: Union[float, List[float]] = 0.0,
                 control_guidance_end: Union[float, List[float]] = 1.0,
                 guidance_scale=7.5,
                 do_classifier_free_guidance=True,
                 eta=0.0,
                 tau=0.5,
                 t=None):
        
        # import pdb; pdb.set_trace()
        control_image = torch.cat([control_image] * 2)
        latent_model_input = torch.cat([latents] * 2)
        control_image = control_image.to(prompt_embeds.dtype)
        latent_model_input = latent_model_input.to(prompt_embeds.dtype)
        #prompt_embeds = torch.cat([prompt_embeds[1], prompt_embeds[0]])
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=controlnet_conditioning_scale,
                    guess_mode=False,
                    return_dict=False,
                )
        
        noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
        
        if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # import pdb; pdb.set_trace()
        latents = self.scheduler.step(noise_pred, t, latents, eta=1., tau=tau, return_dict=False)[0]    
        return latents
        
    def encode_images(self, image, _dtype=None):
        if _dtype == None:
            _dtype = self.unet.dtype
        # import pdb; pdb.set_trace()
        image = self.image_processor.preprocess(image).to(dtype=_dtype)
        latent = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        return latent
    
    def decode_latents(self, latents, _dtype=None):
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = latents.to(_dtype)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    def refine(self, input, strength, guidance_scale=7.5, steps=50, prompt_embeds=None, control_image=None, controlnet_conditioning_scale=1.0):

        pred_rgb_512 = F.interpolate(input, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_images(pred_rgb_512.to(self.unet.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=controlnet_conditioning_scale,
                    guess_mode=False,
                    return_dict=False,
                )
        
            
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=prompt_embeds,
                #cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs
    def encode_prompt(self, prompt, negative_prompt=None, num_images_per_prompt=1, do_classifier_free_guidance=True):
        # import pdb; pdb.set_trace()
        prompt_embeds = self._encode_prompt(
                                            prompt, 
                                            self.device,
                                            num_images_per_prompt,
                                            do_classifier_free_guidance,
                                            negative_prompt,
                                            )
        return prompt_embeds

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        # if do_classifier_free_guidance and not guess_mode:
        #     image = torch.cat([image] * 2)
        return image
    def preprocess_control_image(self, control_image, num_images_per_prompt=1, do_classifier_free_guidance=True):
        # import pdb; pdb.set_trace()
        batch_size = control_image.shape[0]
        height = control_image.shape[2]
        width = control_image.shape[3]
        controlnet = self.controlnet
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                device=self.device,
                dtype=controlnet.dtype,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=False #we need prompt
            )
        elif isinstance(controlnet, MultiControlNetModel):
            images = []
            for image_ in control_image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    device=self.device,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=False,
                )
                images.append(image_)
                image = images
        
        return image

        

    

