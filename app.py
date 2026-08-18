import os
import io
import base64
import json
import glob
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, Response, render_template, jsonify

from models.controlnet import ControlNet
from models.injection import ControlledUNet
from models.loader import encode_prompts, load_sd_components
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.image import latent_to_pil, center_crop_square

app = Flask(__name__)

# Global model cache to make subsequent generations extremely fast
GLOBAL_COMPONENTS = None
GLOBAL_CONTROLLED_UNET = None
CURRENT_CKPT_PATH = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def get_models(ckpt_path):
    global GLOBAL_COMPONENTS, GLOBAL_CONTROLLED_UNET, CURRENT_CKPT_PATH
    
    # Load config file (always base.yaml)
    cfg = load_config("configs/base.yaml", [])
    
    if GLOBAL_COMPONENTS is None:
        print("Loading base Stable Diffusion components (one-time)...")
        GLOBAL_COMPONENTS = load_sd_components(cfg.model.pretrained, device=DEVICE, dtype=DTYPE)
        
    if GLOBAL_CONTROLLED_UNET is None or CURRENT_CKPT_PATH != ckpt_path:
        print(f"Loading ControlNet weights from checkpoint: {ckpt_path}...")
        controlnet = ControlNet.from_unet(GLOBAL_COMPONENTS.unet).to(DEVICE, dtype=DTYPE)
        load_checkpoint(ckpt_path, controlnet, map_location=DEVICE)
        controlnet.eval().requires_grad_(False)
        GLOBAL_CONTROLLED_UNET = ControlledUNet(GLOBAL_COMPONENTS.unet, controlnet).eval()
        CURRENT_CKPT_PATH = ckpt_path
        
    return GLOBAL_COMPONENTS, GLOBAL_CONTROLLED_UNET, cfg

def generate_stream(
    doodle_image_base64,
    prompt,
    negative_prompt,
    steps,
    guidance_scale,
    control_scale,
    seed,
    ckpt_path
):
    try:
        # 1. Load models (reusing cached base components, loading checkpoint weights if changed)
        components, controlled, cfg = get_models(ckpt_path)
        
        # 2. Decode the doodle base64 image
        if "," in doodle_image_base64:
            doodle_image_base64 = doodle_image_base64.split(",")[1]
        doodle_bytes = base64.b64decode(doodle_image_base64)
        doodle_img = Image.open(io.BytesIO(doodle_bytes)).convert("RGB")
        
        # 3. Process the doodle to hint tensor (matches the logic in dataset.py / load_doodle)
        resolution = cfg.data.resolution
        doodle_img = center_crop_square(doodle_img).resize((resolution, resolution), Image.NEAREST)
        doodle_arr = np.asarray(doodle_img, dtype=np.uint8)
        
        # Invert if the image is predominantly bright (ControlNet expects white lines on black)
        if doodle_arr.mean() > 127:
            doodle_arr = 255 - doodle_arr
            
        # Send the processed hint back to the UI so the user can see what the model is actually conditioning on
        hint_img = Image.fromarray(doodle_arr)
        buffered = io.BytesIO()
        hint_img.save(buffered, format="JPEG")
        hint_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        yield f"data: {json.dumps({'status': 'hint', 'image': 'data:image/jpeg;base64,' + hint_b64})}\n\n"
        
        # Prepare hint tensor
        hint_tensor = torch.from_numpy(doodle_arr.astype(np.float32) / 255.0).permute(2, 0, 1)
        hints = hint_tensor.unsqueeze(0).to(DEVICE, dtype=DTYPE)
        hints_both = torch.cat([hints, hints])  # Batched for CFG
        
        # 4. Set seed and scheduler
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        # We import here to ensure compatibility
        from diffusers import UniPCMultistepScheduler
        scheduler = UniPCMultistepScheduler.from_config(components.noise_scheduler.config)
        scheduler.set_timesteps(steps, device=DEVICE)
        
        # 5. Encode prompts
        cond = encode_prompts([prompt], components.tokenizer, components.text_encoder, DEVICE)
        uncond = encode_prompts([negative_prompt], components.tokenizer, components.text_encoder, DEVICE)
        context = torch.cat([uncond, cond]).to(DTYPE)
        
        # 6. Initialize latents
        latent_size = resolution // components.vae_scale_factor
        latents = torch.randn(
            (1, components.unet.config.in_channels, latent_size, latent_size),
            generator=generator, device=DEVICE, dtype=DTYPE,
        ) * scheduler.init_noise_sigma
        
        # 7. Denoising loop
        for i, timestep in enumerate(scheduler.timesteps):
            model_in = scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
            
            with torch.no_grad():
                noise_pred = controlled(
                    model_in, timestep, context, hints_both, control_scale=control_scale
                )
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                latents = scheduler.step(noise_pred, timestep, latents).prev_sample
                
                # Decode the intermediate latent representation for real-time visualization
                vae_dtype = next(components.vae.parameters()).dtype
                decoded = components.vae.decode(
                    latents.to(vae_dtype) / components.vae.config.scaling_factor
                ).sample
                
                # Convert back to PIL
                decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
                arr = (decoded.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
                step_img = Image.fromarray(arr[0])
                
                # Save step image to buffer
                buf = io.BytesIO()
                step_img.save(buf, format="JPEG", quality=80)
                step_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                yield f"data: {json.dumps({'status': 'denoising', 'step': i+1, 'total_steps': steps, 'image': 'data:image/jpeg;base64,' + step_b64})}\n\n"
        
        # 8. Final High-Quality decode
        with torch.no_grad():
            vae_dtype = next(components.vae.parameters()).dtype
            final_decoded = components.vae.decode(
                latents.to(vae_dtype) / components.vae.config.scaling_factor
            ).sample
            final_images = latent_to_pil(final_decoded)
            
            # Save final image to buffer
            buf = io.BytesIO()
            final_images[0].save(buf, format="PNG")
            final_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            yield f"data: {json.dumps({'status': 'done', 'image': 'data:image/png;base64,' + final_b64})}\n\n"
            
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(err_msg)
        yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/checkpoints", methods=["GET"])
def get_checkpoints():
    # Scan checkpoints folder
    ckpt_files = glob.glob("checkpoints/**/*.pt", recursive=True)
    # Sort them alphabetically
    ckpt_files = sorted(ckpt_files)
    # If no checkpoints found, return empty list
    return jsonify({"checkpoints": ckpt_files})

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json
    doodle = data.get("doodle")
    prompt = data.get("prompt", "a realistic image")
    negative_prompt = data.get("negative_prompt", "")
    steps = int(data.get("steps", 30))
    guidance_scale = float(data.get("guidance_scale", 9.0))
    control_scale = float(data.get("control_scale", 1.0))
    seed = int(data.get("seed", 0))
    ckpt = data.get("ckpt")
    
    if not doodle:
        return jsonify({"error": "No doodle provided"}), 400
    if not ckpt:
        return jsonify({"error": "No checkpoint specified"}), 400
        
    return Response(
        generate_stream(
            doodle_image_base64=doodle,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            control_scale=control_scale,
            seed=seed,
            ckpt_path=ckpt
        ),
        mimetype="text/event-stream"
    )

if __name__ == "__main__":
    print("Starting ControlNet Doodle Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
