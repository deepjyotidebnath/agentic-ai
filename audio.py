import streamlit as st
from gtts import gTTS
import io

st.set_page_config(
    page_title="Text to Audio",
    page_icon="🔊"
)

st.title("🔊 Text to Audio")

# Text input
text = st.text_area(
    "Enter your text:",
    placeholder="Hello! Welcome to my AI application."
)

# Language
language = st.selectbox(
    "Choose language:",
    {
        "English": "en",
        "Hindi": "hi",
        "Bengali": "bn"
    }.items(),
    format_func=lambda x: x[0]
)

if st.button("🎙️ Generate Audio"):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Generating audio..."):

        # Create TTS
        tts = gTTS(
            text=text,
            lang=language[1],
            slow=False
        )

        # Keep audio in memory
        audio = io.BytesIO()

        tts.write_to_fp(audio)
        audio.seek(0)

    st.success("Audio generated!")

    # Play audio
    st.audio(
        audio,
        format="audio/mp3"
    )

    # Download audio
    st.download_button(
        "⬇️ Download Audio",
        data=audio,
        file_name="speech.mp3",
        mime="audio/mpeg"
    )