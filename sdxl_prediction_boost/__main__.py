import pathlib
import torch
import diffusers

from . import pipeline


results_folder =  pathlib.Path('result')
results_folder.mkdir(parents=True, exist_ok=True)

pipe = pipeline.StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16",
    vae=diffusers.AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
)
pipe.scheduler = diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.enable_model_cpu_offload()

width, height = 1024, 1024
prompt = ["portrait, natural light, ultra detailed"]
negative_prompt = ['unknown author, random sketch']

generator = torch.Generator("cuda")
# seed = generator.seed()
# print(f"{seed=}")
seed = 42
generator = generator.manual_seed(seed)

for i in range(100):
    state = generator.get_state()
    for j, (label, num_inference_steps, kwargs) in enumerate([
        ("cfg10", 30, {'guidance_scale': 10.0, 'boost_scale': 0}),
        ("cfg3", 30, {'guidance_scale': 2.5, 'boost_scale': 0}),
        ("cfg3boost", 30, {'guidance_scale': 2.5, 'boost_scale': 0.125}),
    ]):
        generator.set_state(state)

        image = pipe(
            prompt=prompt, 
            height=height, 
            width=width, 
            num_inference_steps=num_inference_steps, 
            negative_prompt=negative_prompt, 
            generator=generator,
            **kwargs
        ).images[0]
        image.save(results_folder / f's{i}_c{j}_{label}_{num_inference_steps}steps.png')
