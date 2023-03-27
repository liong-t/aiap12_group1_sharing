# Stable Diffusion with Hugging Face API

4 April 2023

**AIAP Group Sharing**

Presented by Group 1: Shu Ying, JF, Jia Hao and Yan Liong

## 1. Introduction

Stable Diffusion, a text-to-image deep learning model, was developed and released by the start-up Stability AI in August 2022. The user can input a text prompt in plain English, and the model would output an image to match the given prompt. Stable Diffusion quickly made news headlines and captured the popular imagination. In February 2023, Forbes reported that Stable Diffusion is used by "more than 10 million people on a daily basis" [1].

Before the arrival of Stable Diffusion, text-to-image diffision models such as DALL-E and Midjourney were publicly available only via cloud services. Stable Diffusion was the first diffision model which had its code and model weights released to the public, allowing users to run the model on their own modest hardware resources. [2]

Hugging Face has gathered recent diffusion models from independent repositories in a single community project, called the `diffusers` library. [3]

In this article, we walk through some code to demonstrate the use of the `diffusers` API, which allows us to perform inferencing to generate synthetic images.

## 2. What is Stable Diffusion?

[JH]

## 3. API Walkthrough

### a) Model Explanation

Diffusion models are denoising algorithms, trained using deep learning to remove random noise in a series of gradual steps. Starting with 100% noise, after a certain number of steps, the model finally outputs an image to match the text prompt [3].

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusion-process.png" width="800">

The neural net typically follows the UNet architecture, which looks like this [3]:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/unet-model.png" width="800">

Here is another view of the architecture:

<img src="https://i0.wp.com/stable-diffusion-art.com/wp-content/uploads/2022/12/image-85.png?resize=768%2C358&ssl=1" width="800">

Highlights of this architecture, explained in simplified terms:

- The size of the input images is the same as that of the output image [3].
- The input image is downsized using a series of ResNet layers and encoded into a much smaller (compressed) latent space (as a latent image tensor). [3, 4]
- As the latent space is small, this makes it faster for the forward and reverse diffusions to take place. [3, 4]
- From this latent space, the output image is decoded and upscaled to full size. [3, 4]
- The steps of the diffusion (computing a less noisy image) are defined in a scheduler.

The 2 main components in latent diffusion are:

1. Autoencoder (VAE).

Made up of 2 parts, an encoder and a decoder. The encoder converts the image into a low dimensional latent representation, which inputs into the U-Net. The decoder transforms the latent representation back into an image.

2. A U-Net.

Computes the predicted denoised image representation. It is conditioned by the text-embeddings from the text prompt.

3. A text-encoder, e.g. CLIP's Text Encoder.

Transforms the input text prompt into an embedding space that can be understood by the U-Net.



### b) Schedulers

[JF]

### c) forward + backward diffusion

[Yan Liong]

## 4. Application -> Text to imageOutput Demo

[Shu Ying]

## 5. End Notes

[Yan Liong]

## References

[1] Six Things You Didn’t Know About ChatGPT, Stable Diffusion And The Future Of Generative AI. https://www.forbes.com/sites/kenrickcai/2023/02/02/things-you-didnt-know-chatgpt-stable-diffusion-generative-ai/?sh=605dc9c1b5e3

[2] Wikipedia: Stable Diffusion. https://en.wikipedia.org/wiki/Stable_Diffusion

[3] Introducing Hugging Face's new library for diffusion models. https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb

[4] How does Stable Diffusion work? https://stable-diffusion-art.com/how-stable-diffusion-work/

[5] Stable Diffusion with 🧨 Diffusers https://huggingface.co/blog/stable_diffusion
