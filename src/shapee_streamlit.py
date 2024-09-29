import torch
import streamlit as st
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
from PIL import Image
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models and diffusion config
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

st.title("3D Asset Generation with Shap-E")
st.write("Generate 3D assets from text prompts using diffusion models.")

prompt = st.text_input("Enter a text prompt:", "a person running")

batch_size = st.slider("Batch Size", 1, 8, 4)
guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 15.0)
render_mode = st.selectbox("Render Mode", ["nerf", "stf"], index=0)
size = st.slider("Image Size", 32, 128, 64)

generate_button = st.button("Generate 3D Assets")

if generate_button:
    with st.spinner("Generating 3D assets..."):
        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        cameras = create_pan_cameras(size, device)

        st.write(f"### Rendered Images for the prompt: '{prompt}'")
        for i, latent in enumerate(latents):
            images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            
            # Display GIF using PIL and streamlit
            gif_bytes = io.BytesIO()
            images[0].save(gif_bytes, format='GIF', save_all=True, append_images=images[1:], duration=100, loop=0)
            st.image(Image.open(gif_bytes))

        st.write("### Downloadable 3D Mesh Files")
        for i, latent in enumerate(latents):
            # Decode the latent into a mesh
            t = decode_latent_mesh(xm, latent).tri_mesh()
            
            # Save as .ply and .obj and provide download links
            ply_bytes = io.BytesIO()
            obj_bytes = io.StringIO()

            t.write_ply(ply_bytes)
            t.write_obj(obj_bytes)

            st.download_button(
                label=f"Download Mesh {i} as PLY",
                data=ply_bytes.getvalue(),
                file_name=f"example_mesh_{i}.ply",
                mime="application/octet-stream"
            )

            st.download_button(
                label=f"Download Mesh {i} as OBJ",
                data=obj_bytes.getvalue(),
                file_name=f"example_mesh_{i}.obj",
                mime="text/plain"
            )
