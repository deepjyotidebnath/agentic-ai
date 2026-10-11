import os
import streamlit as st
import replicate

from groq import Groq
from dotenv import load_dotenv

# Load .env
load_dotenv()

# -----------------------------
# API clients
# -----------------------------

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Replicate reads REPLICATE_API_TOKEN
# automatically from environment variables


# -----------------------------
# Streamlit configuration
# -----------------------------

st.set_page_config(
    page_title="AI Text to Video",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 AI Text to Video")
st.write("Generate a short video from a text prompt.")


# -----------------------------
# User input
# -----------------------------

prompt = st.text_area(
    "Enter your video idea",
    placeholder=(
        "A futuristic robot walking through "
        "a neon city at night..."
    ),
    height=120
)


# -----------------------------
# Generate button
# -----------------------------

if st.button("🎥 Generate Video", type="primary"):

    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    # -----------------------------
    # STEP 1: Improve prompt with Groq
    # -----------------------------

    with st.spinner("✨ Creating a better video prompt..."):

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert text-to-video prompt engineer.

Convert the user's simple idea into a detailed
video-generation prompt.

Include:
- Main subject
- Subject movement
- Environment
- Camera movement
- Lighting
- Visual style
- Cinematic details

Keep the prompt suitable for a 5-second video.

Return ONLY the final video prompt.
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


    # -----------------------------
    # Show improved prompt
    # -----------------------------

    st.subheader("✨ Enhanced Prompt")

    st.info(improved_prompt)


    # -----------------------------
    # STEP 2: Generate video
    # -----------------------------

    with st.spinner(
        "🎬 Generating video... This may take some time."
    ):

        try:

            output = replicate.run(
                "wan-video/wan-2.1-1.3b",

                input={
                    "prompt": improved_prompt,
                    "frame_num": 81,
                    "resolution": "480p",
                    "aspect_ratio": "16:9",
                    "sample_steps": 30,
                    "sample_guide_scale": 6
                }
            )


            # -----------------------------
            # STEP 3: Display video
            # -----------------------------

            st.success("🎉 Video generated!")

            # Replicate returns a FileOutput
            video_url = output.url()

            st.video(video_url)


            # -----------------------------
            # Download button
            # -----------------------------

            import requests

            video_response = requests.get(video_url)

            if video_response.status_code == 200:

                st.download_button(
                    label="⬇️ Download Video",
                    data=video_response.content,
                    file_name="ai_video.mp4",
                    mime="video/mp4"
                )

        except Exception as e:

            st.error("Something went wrong.")

            st.exception(e)