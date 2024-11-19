import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

prompt1 = "[VIDEO1] A person works on a remote-controlled vehicle, focusing on its underside and chassis. The vehicle has a blue body with a metallic frame, featuring suspension arms, a servo or motor, and off-road tires. The person's hands adjust or inspect components, possibly making repairs or fine-tuning the vehicle. The chassis has a complex design with multiple metal braces and screws, indicating a high-performance vehicle. The person points to a labeled part, \"Teal RC Specialties\" and \"Pro Line Racing,\" and the setting appears to be a workshop or hobbyist space.[VIDEO2] A person stands next to a remote-controlled car with rugged off-road tires and a detailed chassis featuring a visible engine, suspension system, and exhaust pipe. The person holds a remote control in their right hand, indicating they are about to operate or have just finished driving the car. The scene is set outdoors on a paved surface, with the person's blue sneakers visible. The car's off-road capabilities contrast with the flat, paved background."
prompt2 = "[VIDEO1] A city with tall buildings and a large airport is shown. The cityscape features a cluster of modern buildings, including the Bellagio Hotel and Casino, the Paris Las Vegas hotel, and a Ferris wheel. A busy airport is visible in the foreground, with multiple planes parked and in motion. The airport is situated against a backdrop of distant mountains and a clear sky. The city's skyline is dominated by high-rise structures, including hotels and casinos, showcasing a vibrant and modern urban environment. The airport is filled with numerous airplanes, highlighting the city's role as a transportation hub.[VIDEO2] A city is shown in a series of aerial views at night, with a mix of bright and dimly lit areas. The cityscape features a prominent pyramid-shaped structure with a beam of light shining from its peak, serving as a focal point in each view. The surrounding area includes various pools of light, indicating buildings, landmarks, and possibly water bodies. The city's layout and architectural elements are highlighted by the contrast between artificial lights and dark areas."
prompt3 = "[VIDEO1] A city and a mountain are shown in aerial view. The city is a sprawling metropolis with residential and commercial areas, roads, and infrastructure, set against a backdrop of distant mountains and a clear blue sky. A rugged mountain peak stands prominently in the foreground, casting a shadow over the surrounding landscape. The city and mountain are juxtaposed, highlighting the contrast between human development and natural terrain.[VIDEO2] A group of people are hanging from the windows of a tall building, secured by ropes, and engaged in window cleaning or maintenance work. They are positioned at various levels, with some holding buckets and tools, and are wearing safety harnesses. The building has a modern facade with large windows reflecting the golden hues of the surrounding environment. The workers are suspended high above the ground, emphasizing the height and scale of the structure they are working on."
prompt4 = "[VIDEO1] An Asian woman is seated in an office, dressed in a light-colored blazer with a patterned design and a white top, accessorized with a necklace. She is engaged in conversation with the camera, her expression animated. Behind her, wooden shelves hold variously colored binders and a red office chair. She is carrying a black shoulder bag.[VIDEO2] A man and a woman walk through a living room with a cozy and eclectic decor. The room features pink walls with vertical stripes, a wooden door with intricate designs, and a console with a television and electronic devices. The woman is partially visible in the background, dressed in a white outfit, and the man is dressed in a green jacket and beige pants. They move through the room, passing by a plush couch, a small table with a vase, and a unique sculpture of a human figure on the console. The woman holds a white cloth or piece of paper, and the man gestures with his hands as if explaining something. The room also includes a wooden door with an ornate design, adding to the cozy atmosphere."


pipe = CogVideoXPipeline.from_pretrained(
    "ckpts/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("in_context_video/checkpoint-5000/pytorch_lora_weights.safetensors", adapter_name="cogvideox-lora")
print(pipe.lora_scale)
# pipe.fuse_lora(["transformer"], lora_scale=0.7) 
pipe.set_adapters(["cogvideox-lora"], [1.0]) 

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt4,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)
