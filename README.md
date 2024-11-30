# In-Context Video Generalist 


<div align="left">
    <a href="https://huggingface.co/feizhengcong/Incontext-Video"><img src="https://img.shields.io/static/v1?label=Models&message=HuggingFace&color=red"></a> &ensp;
    <a href="https://huggingface.co/datasets/feizhengcong/Incontext-Video"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=blue"></a> &ensp;
    <a href="https://huggingface.co/feizhengcong/Incontext-Video"><img src="https://img.shields.io/static/v1?label=Demo&message=HuggingFace&color=green"></a> &ensp;
</div>

Following image generation in [IC-Lora](https://github.com/ali-vilab/In-Context-LoRA), we directly concatenate both condition and target videos into a single composite video from spacial or time dimension while using natural language to define the task.
It can serve as a general framework for control video generation, with task-specific fine-tuning. 
For more detailed information, please read our technique report. 



<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Text</td>
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

## Get Started 


