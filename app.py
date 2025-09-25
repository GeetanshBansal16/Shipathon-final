import streamlit as st
import pyaudio
import wave
import speech_recognition as sr
import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import PointStruct
import re


# Custom CSS styling
def local_css():
    st.markdown("""
    <style>
        .main-title {
            background: linear-gradient(45deg, #3498db, #2ecc71);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            font-size: 40px !important;
            font-weight: bold !important;
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
            font-weight: bold;
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
            font-weight: bold !important;
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
            font-weight: bold;
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


# Recording function
def record_audio(filename, record_seconds=5):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    st.write("Recording...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


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

    if st.button("Start Recording"):
        record_audio(filename, record_seconds=record_seconds)
        st.audio(filename)

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

    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception as e:
        st.error(f"Failed to configure API. Please check your API key. Error: {e}")
        st.stop()

    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    if 'transcription' in st.session_state:
        # --- WRAP ENTIRE GENERATION LOGIC IN ONE TRY-EXCEPT BLOCK ---
        try:
            transcription = st.session_state.transcription

            # Model for notes creation
            with st.spinner('Generating notes...'):
                model1 = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",  # Using a stable, recommended model
                    generation_config=generation_config,
                    system_instruction='you are an expert in creating notes with a heading from any text and do not give any options '
                )
                chatsession = model1.start_chat(history=[])
                response = chatsession.send_message(transcription)
                st.session_state.notes = response.text

            # Save to file
            with open("output.txt", "w") as output:
                output.write(response.text)

            # ------------------- UPDATED Q/A GENERATION ------------------- #
            # Model for question generation
            with st.spinner('Generating questions...'):
                model2 = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
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
                chatsession2 = model2.start_chat(history=[])
                response2 = chatsession2.send_message(response.text)

            # Model for answer generation
            with st.spinner('Generating answers...'):
                model3 = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
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
                chatsession3 = model3.start_chat(history=[])
                response3 = chatsession3.send_message(response2.text)

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

            Questions = [q.strip() for q in re.findall(r'Q\d+\.\s*(.*)', response2.text)]
            Answers = [a.strip() for a in re.findall(r'Ans\d+\.\s*(.*)', response3.text)]

            for i in range(min(len(Questions), len(Answers))):
                st.markdown(f"**{Questions[i]}**")
                st.markdown(f"{Answers[i]}")
            # ------------------- END OF UPDATED Q/A GENERATION ------------------- #

        # --- CATCH BLOCK FOR THE MAIN GENERATION LOGIC ---
        except Exception as e:
            st.error(f"An API error occurred: {e}")
            st.warning(
                "This may be due to an exhausted API key, rate limits, or content safety filters. Please check your Google AI Studio dashboard and try again.")
            st.stop()

        # Doubts section (This part is interactive, so it gets its own error handling)
        st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)
        for i in range(num_doubts):
            query = st.text_input(f'Doubt {i + 1}:', key=f'doubt_{i}')

            if query:
                try:
                    with st.spinner('Thinking...'):
                        # Qdrant search logic
                        client = QdrantClient(":memory:")
                        model = SentenceTransformer('all-mpnet-base-v2')
                        notes = st.session_state.notes
                        chunks = [c for c in notes.split(".\n") if c]  # Filter out empty chunks

                        if not chunks:
                            st.warning("Could not find content in the notes to search.")
                            continue

                        embeddings = model.encode(chunks).tolist()

                        client.recreate_collection(
                            collection_name="chunksf",
                            vectors_config=VectorParams(size=len(embeddings[0]), distance="Cosine"),
                        )

                        client.upsert(
                            collection_name="chunksf",
                            points=[
                                PointStruct(id=idx, vector=emb, payload={"content": chunk})
                                for idx, (emb, chunk) in enumerate(zip(embeddings, chunks))
                            ],
                        )

                        query_embedding = model.encode(query).tolist()
                        search_results = client.search(
                            collection_name="chunksf",
                            query_vector=query_embedding,
                            limit=3,
                        )

                        # Combine the content of the top results to form a context
                        recommendations = " ".join([result.payload['content'] for result in search_results])

                        modelans = genai.GenerativeModel("gemini-1.5-flash",
                                                         system_instruction="You are an expert in answering questions based on a given context. Answer concisely.")
                        chatsessionans = modelans.start_chat(history=[])
                        responseans = chatsessionans.send_message(
                            f"Based on the context: '{recommendations}', answer the following question: '{query}'")

                        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
                        st.write(responseans.text)
                        st.markdown('</div>', unsafe_allow_html=True)

                # --- CATCH BLOCK FOR THE DOUBTS SECTION ---
                except Exception as e:
                    st.error(f"Could not answer your doubt due to an API error: {e}")
                    st.warning("This may be due to an exhausted API key or other issues.")


if __name__ == "__main__":
    main()
