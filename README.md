# In-Context Video Generalist 


<div align="left">
    <a href="https://huggingface.co/feizhengcong/In-context-Video-Generalist"><img src="https://img.shields.io/static/v1?label=Models&message=HuggingFace&color=red"></a> &ensp;
    <a href="https://huggingface.co/datasets/multimodalart/panda-70m"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=blue"></a> &ensp;
    <a href="https://huggingface.co/feizhengcong/In-context-Video-Generalist"><img src="https://img.shields.io/static/v1?label=Demo&message=HuggingFace&color=green"></a> &ensp;
</div>

## 🔭 Introduction 

<p align="left">
<strong>TL;DR: We explores in-context capabilities in video diffusion transformers, with minimal tuning to activate them.</strong>
</p>

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Prompt</td>
        <td>Generated video</td>
    </tr>
    <tr>
      <td>
	Four video storyboards.  [1] The video captures a serene countryside scene where a group of people are riding black horses. [2] The video captures a serene rural scene where a group of people are riding horses on a dirt road. [3] The video captures a serene rural scene where a person is riding a dark-colored horse along a dirt path. [4] The video captures a serene rural scene where a woman is riding a horse on a country road. 
	  </td>
	  <td>
     		<image src=cases/2.gif width="300">
	  </td>
  	</tr>
                <tr>
      <td>
	    Four video storyboards.  [1] The video captures a serene autumn scene in a forest, where a group of people are riding horses along a dirt path. [2] The video captures a group of horse riders traversing a dirt road in a rural setting. [3] The video captures a group of horse riders in a grassy field, with a backdrop of distant mountains and a clear sky. [4] The video captures a serene autumnal scene in a forest, where a group of horse riders is traversing a dirt trail. 
	  </td>
	  <td>
     		<image src=cases/1.gif width="300">
	  </td>
  	</tr>
  	<tr>
      <td>
	    Four video storyboards of one young boy.  [1] sad. [2] happy. [3] disgusted in cartoon style. [4] contempt in cartoon style.
	  </td>
	  <td>
     		<image src=cases/0.gif width="300">
	  </td>
  	</tr>
</table >

<p align="justify">
  <strong>Abstract:</strong> 
Following In-context-Lora, we directly concatenate both condition and target videos into a single composite video from spacial or time dimension while using natural language to define the task. 
It can serve as a general framework for control video generation, with task-specific fine-tuning. More encouragingly, it can create a consistent multi-scene video more than 30 seconds without any more computation burden.  
</p>
For more detailed information, please read our technique report. 

## 💡 Quick Start

### 1. Setup repository and environment 

Our environment is totally same with CogvideoX and you can install by: 

```
pip install -r requirement.txt
```

### 2. Download checkpoint
Download the lora [checkpoint](https://huggingface.co/feizhengcong/In-context-Video-Generalist) from huggingface, and put it with model path variable. 

We provide the scene and human loras, which generate the cases with different prompt types in technique report. 


### 3. Launch the inference script! 
You can run with mini code as following or refer to `infer.py` which generate cases, after setting the path for lora.  

```
from diffusers.utils import export_to_video
from diffusers import CogVideoXPipeline 

lora_path = /path/to/lora

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights(lora_path, adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [1.0]) 

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)
```

## 🔧  Lora Fine-tuning 

You can training with your own lora for control tasks with finetuning scripts and our experiments can be repeated by simply run the training scripts as:

```
sh finetune.sh 
```

Before, you should prepare:
- Video-text pair data as [formation](https://github.com/feizc/In-Context-Video-Generalist/blob/main/training/dataset.py);
- Prompt template to combine different video clips;

## 🔗 Acknowledgments 

The codebase is based on the awesome [IC-Lora](https://github.com/ali-vilab/In-Context-LoRA), [CogvideoX](https://github.com/THUDM/CogVideo), [Cogvideo-factory](https://github.com/a-r-r-o-w/cogvideox-factory), and [diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py) repos.

