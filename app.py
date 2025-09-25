import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import wave
import os
import speech_recognition as sr
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, ScoredPoint
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import PointStruct
import re
import numpy as np

# Custom CSS styling
def local_css():
    st.markdown("""
    <style>
        .main-title {
            background: linear-gradient(45deg, #3498db, #2ecc71);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            font-size: 40px !important;
            font-w: bold !important;
            text-align: center !important;
            padding: 20px 0 !important;
            font-family: 'Arial Black', sans-serif !important;
        }

        .stButton > button {
            width: 100%;
            padding: 15px 20px;
            background: linear-gradient(45deg, #3498db, #2ecc71);
            color: white;
            border: none;
            border-radius: 10px;
            font-w: bold;
            transition: all 0.3s ease;
            margin: 10px 0;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            background: linear-gradient(45deg, #2ecc71, #3498db);
        }

        .section-title {
            color: #2c3e50;
            font-size: 24px !important;
            font-w: bold !important;
            margin: 20px 0 !important;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }

        .custom-container {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }

        .qa-container {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #3498db;
        }

        .question {
            color: #2c3e50;
            font-w: bold;
            margin-bottom: 10px;
        }

        .answer {
            color: #34495e;
            padding-left: 20px;
        }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state
if 'run' not in st.session_state:
    st.session_state.run = False


# Audio Processor for streamlit-webrtc
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame


def save_audio(frames, filename="recorded_audio.wav", rate=16000):
    audio_data = np.concatenate(frames, axis=0)
    audio_data = (audio_data * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio_data.tobytes())


# Transcription function
def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"


# Main app
def main():
    st.set_page_config(page_title="IITD Note Maker", page_icon="📝", layout="wide")
    local_css()

    # Title
    st.markdown('<h1 class="main-title">IITD Note Maker: Intelligent Class Companion</h1>', unsafe_allow_html=True)

    filename = "recorded_audio.wav"

    # Recording input
    st.markdown('<div class="section-title">Audio Recording</div>', unsafe_allow_html=True)
    record_seconds = st.number_input("Enter recording duration (in seconds):", min_value=1, step=1, value=5)

    # WebRTC streamer
    audio_recorder = AudioRecorder()
    webrtc_ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=lambda: audio_recorder,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        if st.button("Save Recording"):
            save_audio(audio_recorder.frames, filename=filename)
            st.audio(filename)
            st.success("Audio saved successfully!")

    # Transcription
    if st.button("Transcribe Audio"):
        if os.path.exists(filename):
            transcription = transcribe_audio(filename)
            st.markdown('<div class="custom-container">', unsafe_allow_html=True)
            st.write("Transcription:")
            st.write(transcription)
            st.markdown('</div>', unsafe_allow_html=True)
            st.session_state.run = True
            st.session_state.transcription = transcription
            os.remove(filename)
        else:
            st.error("No audio file found. Please record audio first.")

    # Number of questions and doubts input
    num_questions = st.number_input("Enter number of questions to generate:", min_value=1, step=1, value=5)
    num_doubts = st.number_input("Enter number of doubts you want to ask:", min_value=1, step=1, value=1)

    # Configure API key safely
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        api_available = True
    except Exception as e:
        api_available = False

    if not api_available:
        st.warning("Gemini API key has been exhausted. Note generation and Q/A will not work.")
        return

    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    if 'transcription' in st.session_state:
        transcription = st.session_state.transcription

        # Model for notes creation
        model1 = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction='you are an expert in creating notes with a heading from any text and do not give any options '
        )
        chatsession = model1.start_chat(history=[])
        response = chatsession.send_message(transcription)
        st.session_state.notes = response.text

        # Save to file
        output = open("output.txt", "w")
        output.write(response.text)

        # ------------------- UPDATED Q/A GENERATION ------------------- #
        # Model for question generation
        model2 = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction=f"""
You are an expert in generating questions.
Generate exactly {num_questions} questions from the given text.
Output format must be:
Q1. <question 1>
Q2. <question 2>
...
Q{num_questions}. <question {num_questions}>
Do not add anything else, no extra text or explanation.
"""
        )
        chatsession = model2.start_chat(history=[])
        response2 = chatsession.send_message(response.text)

        # Model for answer generation
        model3 = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction=f"""
You are an expert in generating answers to questions based on context.
Given context:
{response.text}
Answer exactly {num_questions} questions in the format:
Ans1. <answer 1>
Ans2. <answer 2>
...
Ans{num_questions}. <answer {num_questions}>
Do not add anything else.
"""
        )
        chatsession = model3.start_chat(history=[])
        response3 = chatsession.send_message(response2.text)

        # Display notes
        st.markdown('<div class="section-title">Generated Notes</div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.write(st.session_state.notes)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download button
        st.download_button(
            label="Download Notes",
            data=st.session_state.notes,
            file_name="Your_notes.txt",
            mime="text/plain"
        )

        # Display Q&A
        st.markdown('<div class="section-title">Questions and Answers</div>', unsafe_allow_html=True)

        Questions = [q.strip() for q in re.findall(r'^Q\d+\.\s*(.*)', response2.text, flags=re.MULTILINE)]
        Answers = [a.strip() for a in re.findall(r'^Ans\d+\.\s*(.*)', response3.text, flags=re.MULTILINE)]

        for i in range(min(num_questions, len(Questions))):
            st.markdown(f"**{Questions[i]}**")
            st.markdown(f"{Answers[i]}")
        # ------------------- END OF UPDATED Q/A GENERATION ------------------- #

        # Doubts section
        st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)
        for i in range(num_doubts):
            query = st.text_input(f'Doubt {i + 1}:')

            if query:
                # Qdrant search logic
                client = QdrantClient(":memory:")
                model = SentenceTransformer('all-mpnet-base-v2')
                notes = st.session_state.notes
                chunks = notes.split(".\n")
                chunksf = []

                for i in range(len(chunks)):
                    chunksf.append({"id": i, "content": chunks[i]})
                for ch in chunksf:
                    ch["embedding"] = model.encode(f"{ch['content']}").tolist()

                client.recreate_collection(
                    collection_name="chunksf",
                    vectors_config=VectorParams(size=len(chunksf[0]['embedding']), distance="Cosine"),
                )

                client.upsert(
                    collection_name="chunksf",
                    points=[
                        PointStruct(
                            id=ch["id"],
                            vector=ch["embedding"],
                            payload={
                                "id": ch["id"],
                                "content": ch["content"],
                            },
                        )
                        for ch in chunksf
                    ],
                )

                query_embedding = model.encode(query).tolist()
                search_results = client.search(
                    collection_name="chunksf",
                    query_vector=query_embedding,
                    limit=3,
                )

                recommendations = ""
                scor = -float("inf")
                for result in search_results:
                    song_id = result.id
                    ch = next(ch for ch in chunksf if ch["id"] == result.id)
                    if result.score > scor:
                        scor = result.score
                        recommendations = ch["content"]

                modelans = genai.GenerativeModel("gemini-2.0-flash-exp",
                                                 system_instruction="you are an expert in answering questions based on a given context")
                chatsessionans = modelans.start_chat(history=[])
                responseans = chatsessionans.send_message(f"{query} Based on {recommendations}")

                st.markdown('<div class="custom-container">', unsafe_allow_html=True)
                st.write(responseans.text)
                st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
