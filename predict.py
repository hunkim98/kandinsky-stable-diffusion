import os
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DDIMInverseScheduler,
    StableDiffusionDiffEditPipeline,
)


MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        sd_model_ckpt = "stabilityai/stable-diffusion-2-1"
        self.pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
            sd_model_ckpt,
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True,
        ).to("cuda")
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image",
            default=None,
        ),
        source_prompt: str = Input(
            description="The prompt to guide the semantic mask generation using the method used in the DiffEdit paper",
            default="a bowl of fruits",
        ),
        target_prompt: str = Input(
            description="The prompt to guide the target image generation",
            default="a basket of fruits",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        raw_image = Image.open(str(image)).convert("RGB").resize((768, 768))
        # employ the source and target prompts to generate the editing mask
        mask_image = self.pipeline.generate_mask(
            image=raw_image,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            generator=generator,
            guidance_scale=guidance_scale,
        )
        # employ the caption and the input image to get the inverted latents:
        inv_latents = self.pipeline.invert(
            prompt=source_prompt, image=raw_image, generator=generator
        ).latents

        # generate the image with the inverted latents and semantically generated mask:
        image = self.pipeline(
            prompt=target_prompt,
            mask_image=mask_image,
            image_latents=inv_latents,
            negative_prompt=source_prompt,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        output_path = f"/tmp/out.png"
        image.save(output_path)

        return Path(output_path)
