# Stable Diffusion v2.1 Cog model

[![Replicate](https://replicate.com/cjwbw/stable-diffusion-v2.1/badge)](https://replicate.com/cjwbw/stable-diffusion-v2.1) 

This is an implementation of the [Diffusers Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2-1) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving"
