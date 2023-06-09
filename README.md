# Kandinsky-2-1 Cog model

[![Replicate](https://replicate.com/cjwbw/kandinsky-2-1/badge)](https://replicate.com/cjwbw/kandinsky-2-1) 

This is an implementation of the [kandinsky-community/kandinsky-2-1](https://huggingface.co/kandinsky-community/kandinsky-2-1) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt="..."
