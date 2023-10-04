import os
from typing import List
from PIL import Image

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    KandinskyInpaintPipeline,
    KandinskyImg2ImgPipeline,
    KandinskyPriorPipeline,
    KandinskyPipeline,
)


MODEL_CACHE = "model_cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pipe_prior = KandinskyPriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1-prior",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.t2i_pipe = KandinskyPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.i2i_pipe = KandinskyImg2ImgPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to("cuda")

        self.inpaint_pipe = KandinskyInpaintPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1-inpaint",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        task: str = Input(
            description="Choose a task",
            choices=["text2img", "text_guided_img2img", "inpaint"],
            default="text2img",
        ),
        prompt: str = Input(
            description="Provide input prompt",
            default="A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output for text2img and text_guided_img2img tasks",
            default="low quality, bad quality",
        ),
        image: Path = Input(
            description="Input image for text_guided_img2img task",
            default=None,
        ),
        mask: Path = Input(
            description="Mask for inpainting task",
            default=None,
        ),
        strength: float = Input(
            description="indicates how much to transform the input iamge, valid for text_guided_img2img task.",
            default=0.3,
            le=1,
            ge=0,
        ),
        width: int = Input(
            description="Width of output image. Reduce the seeting if hits memory limits",
            ge=512,
            le=1024,
            default=256,
        ),
        height: int = Input(
            description="Height of output image. Reduce the seeting if hits memory limits",
            ge=512,
            le=1024,
            default=256,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_steps_prior: int = Input(
            description="Number of denoising steps in prior", ge=1, le=500, default=25
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=100
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=4.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        image_embeds, negative_image_embeds = self.pipe_prior(
            prompt,
            negative_prompt,
            num_images_per_prompt=num_outputs,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps_prior,
            generator=generator,
        ).to_tuple()

        if task == "text2img":
            images = self.t2i_pipe(
                prompt=[prompt] * num_outputs,
                negative_prompt=[negative_prompt] * num_outputs,
                image_embeds=image_embeds,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                negative_image_embeds=negative_image_embeds,
            ).images
        elif task == "text_guided_img2img":
            assert (
                prompt is not None and image is not None
            ), "Please provide prompt and image for text_guided_img2img task"
            original_image = Image.open(str(image)).convert("RGB")
            original_image = original_image.resize((768, 512))

            images = self.i2i_pipe(
                prompt=[prompt] * num_outputs,
                image=[original_image] * num_outputs,
                image_embeds=image_embeds,
                num_inference_steps=num_inference_steps,
                negative_image_embeds=negative_image_embeds,
                width=width,
                height=height,
                strength=strength,
            ).images
        else:  # this is inpaint
            assert (
                prompt is not None and image is not None
            ), "Please provide prompt and image for inpaint task"
            assert (mask is not None), "Please provide mask for inpaint task"
            original_image = Image.open(str(image)).convert("RGB")
            original_image = original_image.resize((768, 512))
            mask_image = Image.open(str(mask)).convert("RGB")
            mask_image = mask_image.resize((768, 512))

            images = self.inpaint_pipe(
                prompt=[prompt] * num_outputs,
                image=[original_image] * num_outputs,
                mask_image=[mask_image] * num_outputs,
                image_embeds=image_embeds,
                num_inference_steps=num_inference_steps,
                negative_image_embeds=negative_image_embeds,
                width=width,
                height=height,
                strength=strength,
            ).images

        output_paths = []
        for i, img in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            img.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
