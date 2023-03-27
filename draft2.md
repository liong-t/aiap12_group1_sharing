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

[Below is a draft by JF: Please feel free to use or discard.]

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



## 3. Hugging Face's Pipeline Components

The 3 main components in Hugging Face's diffusion are:

Text-Encoder
- The text-encoder transforms input prompts into an embedding space for the U-Net. A transformer-based encoder maps input tokens to latent text-embeddings. Stable Diffusion uses CLIP's pretrained text encoder, CLIPTextModel, without additional training.

Variational Autoencoder (VAE)
- The VAE consists of an encoder and a decoder, whose main goal is to transform an image back and from its latent space . The encoder converts images into low-dimensional latent representations, serving as input for the U-Net model. The decoder transforms latent representations back into images.

Scheduler
- The scheduling algorithm used to progressively add noise to the image during training.

U-Net
- U-Net has an encoder and a decoder, both composed of ResNet blocks, its goal is to process the latent images obtained from the VAE encoder. The encoder compresses image representation into lower resolution, while the decoder reconstructs the original, less noisy, high-resolution representation. U-Net output predicts noise residual for denoised image calculation. Shortcut connections between encoder's downsampling and decoder's upsampling ResNets prevent information loss. Stable diffusion U-Net conditions its output on text-embeddings via cross-attention layers in both encoder and decoder.


### a) Creating the components

```
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. Create the scheduler
scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

# 4. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")


```

### b) Setup the text encoding

Initially, we obtain the text embeddings for the given prompt, which will serve as a basis for conditioning the UNet model.

```
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

with torch.no_grad():
  text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
```

Furthermore, we will acquire unconditional text embeddings to provide classifier-free guidance, which consist solely of the embeddings for the padding token (empty text). It is crucial to ensure that their dimensions match those of the conditional text embeddings (i.e., batch_size and seq_length).

```
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
  uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   
```

To enable classifier-free guidance, we need to perform two separate forward passes. One pass is with the conditioned input (text_embeddings), while the other is with the unconditional embeddings (uncond_embeddings). To optimize computational efficiency, it is feasible to concatenate both sets of embeddings into a single batch, thus eliminating the need to perform two separate forward passes.

```
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
```

Next we generate the random latent space

```
latents = torch.randn(
  (batch_size, unet.in_channels, height // 8, width // 8),
  generator=generator,
)
latents = latents.to(torch_device)
```

Next, we initialize the scheduler with our chosen num_inference_steps. This will compute the sigmas and exact time step values to be used during the denoising process.

```
scheduler.set_timesteps(num_inference_steps)
```


The K-LMS scheduler necessitates the multiplication of latents with their corresponding sigma values. We can accomplish this step at this point.

```
latents = latents * scheduler.init_noise_sigma
```


We use a pipeline to group together a **model** and a **scheduler** and make it easy for an end-user to run a full denoising loop process.

```
from diffusers import DDPMPipeline

image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")

```

The `from_pretrained()` method downloads the model and its configuration from the Hugging Face Hub community repo.

We use a pre-trained model called "google/ddpm-celebahq-256" which was trained on a dataset of celebrities images.

We can print out the image_pipe to see what is inside:

```
print(image_pipe)
```

Output:

```
DDPMPipeline {
  "_class_name": "DDPMPipeline",
  "_diffusers_version": "0.3.0",
  "scheduler": [
    "diffusers",
    "DDPMScheduler"
  ],
  "unet": [
    "diffusers",
    "UNet2DModel"
  ]
}
```

From the output, we can see a scheduler and a UNet model.

The **model** is a pre-trained neural network used for predicting a slightly less noisy image or residual (difference between the slightly less noisy image and the input image). It takes a noisy sample and a timestep as inputs to predict a less noisy output sample.

The **scheduler** is used during both training and inferencing. During training, it defines the noise schedule which is used to add noise to the model. During inferencing, it defines the algorithm to compute the slightly less noisy sample.[3]

 For the purposes of this blog we will be focusing on the inferencing aspect of the diffusers library which occurs in the hugging face's diffusion pipeline. If you however wish to read up on the training portion please go to this jupyter notebook  https://colab.research.google.com/gist/anton-l/f3a8206dae4125b93f05b1f5f703191d/diffusers_training_example.ipynb#scrollTo=67640279-979b-490d-80fe-65673b94ae00 

### a) Model

The `model` API allows us to download a model's configuration and weights from a repo, using the `from_pretrained()` method. After you download for the first time, it is cached locally, so subsequent execution will be faster. [3]

For example, below we download a UNet2DModel image generation model trained on church images. The model is a PyTorch torch.nn.Module class.

```
from diffusers import UNet2DModel

repo_id = "google/ddpm-church-256"
model = UNet2DModel.from_pretrained(repo_id)
```

We can print out the model's configuration to take a look. Some of the more important config parameters are annotated below in the code.

```
print(model.config)
```
Output:

```
FrozenDict([('sample_size', 256),  # height and width dimension of the input sample.
            ('in_channels', 3),    # number of input channels of the input sample.
            ('out_channels', 3),
            ('center_input_sample', False),
            ('time_embedding_type', 'positional'),
            ('freq_shift', 1),
            ('flip_sin_to_cos', False),
            ('down_block_types',
             ['DownBlock2D',
              'DownBlock2D',
              'DownBlock2D',
              'DownBlock2D',
              'AttnDownBlock2D',
              'DownBlock2D']),
            ('up_block_types',
             ['UpBlock2D',
              'AttnUpBlock2D',
              'UpBlock2D',
              'UpBlock2D',
              'UpBlock2D',
              'UpBlock2D']),
            ('block_out_channels', [128, 128, 256, 256, 512, 512]),
            ('layers_per_block', 2),  # how many ResNet blocks are present in each UNet block.
            ('mid_block_scale_factor', 1),
            ('downsample_padding', 0),
            ('act_fn', 'silu'),
            ('attention_head_dim', None),
            ('norm_num_groups', 32),
            ('norm_eps', 1e-06),
            ('_class_name', 'UNet2DModel'),
            ('_diffusers_version', '0.3.0'),
            ('_name_or_path', 'google/ddpm-church-256')])
```

Notice that model config is a frozen dictionary, which is immutable. That means it contains no attributes that can be changed during inference. [3]

Let's use the model to do some inferencing.

We start by generating a random gaussian sample which is the same shape as the output image.

```
import torch

torch.manual_seed(0)

noisy_sample = torch.randn(
    1, model.config.in_channels, model.config.sample_size, model.config.sample_size
)
print(noisy_sample.shape)
```
Output:
```
torch.Size([1, 3, 256, 256])
```

We pass the noisy sample and a timestep to the model. The timestep is a point between the start (more noisy) and the end (less noisy) of the diffusion process. [3]

Below, we use the model to predict the noise residual (difference between the slightly less noisy image and the input image) at timestep 2. The noise residual will be used by the scheduler to compute a slightly less noisy image.

```
with torch.no_grad():
    noisy_residual = model(sample=noisy_sample, timestep=2).sample
```

### b) Schedulers

Unlike a model, a scheduler has no trainable weights (so is not inherited from torch.nn.Module), but is instantiated by a configuration. It is simply a step-by-step algorithm to compute the slightly less noisy sample. [3]

There are different types of schedulers. Different schedulers work with different models. [3]

Like a model, you can download a scheduler from the repo. In the example below, we load a DDPMScheduler (DDPM stands for Denoising Diffusion Probabilistic Models):

```
from diffusers import DDPMScheduler

scheduler = DDPMScheduler.from_config(repo_id)
```

We can print out the scheduler's configuration to take a look. Some of the more important config parameters are annotated below in the code.

```
print(scheduler.config)
```
Output:
```
FrozenDict([('num_train_timesteps', 1000),  # Length of the denoising process,
                                            # i.e. how many timesteps are need to process
                                            # random gaussian noise to a data sample.
            ('beta_start', 0.0001),         # Smallest noise value of the schedule.
            ('beta_end', 0.02),             # Highest noise value of the schedule.
            ('beta_schedule', 'linear'),    # Type of noise schedule that shall be used
                                            # for inference and training.
            ('trained_betas', None),
            ('variance_type', 'fixed_small'),
            ('clip_sample', True),
            ('_class_name', 'DDPMScheduler'),
            ('_diffusers_version', '0.3.0')])
```

The scheduler has a `step()` function which is used to compute the slightly less noisy image. The step function takes in a few arguments:

- The `noisy_residual` previously predicted by the model (difference between the slightly less noisy image and the input image).
- The timestep.
- The current `noisy_sample`.

```
less_noisy_sample = scheduler.step(
    model_output=noisy_residual, timestep=2, sample=noisy_sample
).prev_sample
print(less_noisy_sample.shape)
```
Output:
```
torch.Size([1, 3, 256, 256])
```

Notice that the output sample's shape is the same as the input, so it is can be looped back into the model again. [3]


### c) Denoising loop 

The denoising loop for DDPM is straightforward:

- Predict the noise residual of a less noisy sample using the model.
- Compute the less noisy sample with the scheduler.
- Display progress every 50 steps.
- Loop over the tensor scheduler.timesteps, which defines the sequence of timesteps for the denoising process. 
- Typically, denoising iterates from the maximum number of timesteps (e.g., 1000) down to 0.

```
import tqdm

sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
  # 1. predict noise residual
  with torch.no_grad():
      residual = model(sample, t).sample

  # 2. compute less noisy image and set x_t -> x_t-1
  sample = scheduler.step(residual, t, sample).prev_sample

  # 3. optionally look at image
  if (i + 1) % 50 == 0:
      display_sample(sample, i + 1)
```
 <img src="denoise_1.jpg" width="500">


## 4. Application -> Text to imageOutput Demo

[Shu Ying]

## 5. End Notes

[Yan Liong]

## References

[1] Six Things You Didn’t Know About ChatGPT, Stable Diffusion And The Future Of Generative AI. https://www.forbes.com/sites/kenrickcai/2023/02/02/things-you-didnt-know-chatgpt-stable-diffusion-generative-ai/?sh=605dc9c1b5e3

[2] Wikipedia: Stable Diffusion. https://en.wikipedia.org/wiki/Stable_Diffusion

[3] Introducing Hugging Face's new library for diffusion models. https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb

[4] How does Stable Diffusion work? https://stable-diffusion-art.com/how-stable-diffusion-work/

[5] Stable Diffusion with 🧨 Diffusers. https://huggingface.co/blog/stable_diffusion
