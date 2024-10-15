import streamlit as st
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(page_title="AI Image Generator", page_icon=":art:", layout="centered")

# Header section
st.title("🎨 AI-Powered Image Generator")
st.subheader("Generate stunning images from your creative prompts using Stable Diffusion")
st.markdown("""
    This app uses the powerful **Stable Diffusion** model to transform text prompts into high-quality images.
    Enter your imagination below and see what art comes to life! 
    """)

# Sidebar for additional options
st.sidebar.title("Settings")
st.sidebar.markdown("Customize your generation process")

guidance_scale = st.sidebar.slider("Guidance Scale", min_value=5.0, max_value=20.0, value=8.5, step=0.5)
height = st.sidebar.slider("Image Height", min_value=256, max_value=768, value=512, step=64)
width = st.sidebar.slider("Image Width", min_value=256, max_value=768, value=512, step=64)

# Input prompt from the user
st.markdown("### 📝 Enter your creative prompt")
textprompt = st.text_area("What do you want to see?", height=150)

# Button to generate the image
generate_button = st.button("🎨 Generate Image")

# Load the model from the saved directory
device = "cuda" if torch.cuda.is_available() else "cpu"

# Show a loading spinner while generating the image
if generate_button and textprompt:
    with st.spinner("Creating your image..."):
        # Load the pre-saved model
        pipe = StableDiffusionPipeline.from_pretrained("./stable_diffusion_model", torch_dtype=torch.float16)
        pipe = pipe.to(device)

        # Generate the image
        with autocast(device):
            image = pipe(textprompt, guidance_scale=guidance_scale, height=height, width=width).images[0]

        # Display the generated image
        st.image(image, caption="✨ Your Generated Image", use_column_width=True)

        # Option to download the image
        img_pil = image
        img_pil.save("generated_image.png")
        
        with open("generated_image.png", "rb") as file:
            st.download_button(
                label="📥 Download Image",
                data=file,
                file_name="generated_image.png",
                mime="image/png",
                help="Click to download your generated image"
            )

# Footer section
st.markdown("---")
st.markdown("**Powered by**: Stable Diffusion, PyTorch, and Hugging Face 🤗")
st.markdown("Created by [AKLLIGENTS](https://your-portfolio-link.com)")
