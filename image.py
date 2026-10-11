import os
import requests
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Groq client
# -----------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨"
)

st.title("🎨 Text to Image Generator")

prompt = st.text_area(
    "Enter your image prompt:",
    placeholder="A futuristic city at night with flying cars..."
)

if st.button("Generate Image"):

    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    # -----------------------------
    # Step 1: Improve prompt using Groq
    # -----------------------------
    with st.spinner("Improving prompt..."):

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an expert image prompt engineer.
                    Convert the user's simple idea into a detailed
                    image-generation prompt.

                    Include:
                    - subject
                    - environment
                    - lighting
                    - camera angle
                    - visual style
                    - details

                    Return ONLY the final image prompt.
                    """
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )

        improved_prompt = response.choices[0].message.content

    st.subheader("✨ Improved Prompt")
    st.write(improved_prompt)

    # -----------------------------
    # Step 2: Generate image
    # -----------------------------
    with st.spinner("Generating image..."):

        # Pollinations image URL
        image_url = (
            "https://image.pollinations.ai/prompt/"
            + requests.utils.quote(improved_prompt)
        )

        image_response = requests.get(image_url)

        if image_response.status_code == 200:

            st.subheader("🖼️ Generated Image")

            st.image(
                image_response.content,
                caption="AI Generated Image",
                use_container_width=True
            )

            # Download button
            st.download_button(
                label="⬇️ Download Image",
                data=image_response.content,
                file_name="generated_image.jpg",
                mime="image/jpeg"
            )

        else:
            st.error("Image generation failed.")