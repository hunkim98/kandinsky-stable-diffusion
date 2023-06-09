# DiffEdit Stable Diffusion Cog model

[![Replicate](https://replicate.com/cjwbw/diffedit-stable-diffusion/badge)](https://replicate.com/cjwbw/diffedit-stable-diffusion) 

This is an implementation of the [DiffEdit-stable-diffusion](https://github.com/Xiang-cd/DiffEdit-stable-diffusion) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i image=...
