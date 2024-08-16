import os
import uvicorn
import requests
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv
from multiprocessing import Process
import streamlit as st

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Create prompt templates
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser = StrOutputParser()

# Create chain
chain = prompt_template | model | parser

# FastAPI app definition
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces"
)

add_routes(
    app,
    chain,
    path="/chain"
)

# Function to run the FastAPI server
def run_fastapi():
    uvicorn.run(app, host="localhost", port=8000)

# Function to get response from the FastAPI server
def get_groq_response(input_text, target_language):
    json_body = {
        "input": {
            "language": target_language,
            "text": f"{input_text}"
        },
        "config": {},
        "kwargs": {}
    }
    response = requests.post("http://localhost:8000/chain/invoke", json=json_body)
    try:
        response_data = response.json()
        output = response_data.get("output", "No result field in response")
        return output
    except ValueError:
        return "Error: Invalid JSON response"

# Streamlit app configuration
st.set_page_config(page_title="Language Translator", page_icon="🌐", layout="centered")

# Custom CSS for a compact and elegant dark theme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&display=swap');

        body {
            font-family: 'Roboto', sans-serif;
            background: #121212;
            color: #E0E0E0;
            margin: 0;
            padding: 0;
        }

        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            padding: 20px;
        }

        .title {
            font-family: 'Montserrat', sans-serif;
            font-size: 36px;
            font-weight: 700;
            color: #EAEAEA;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.8);
        }

        .description {
            font-size: 16px;
            text-align: center;
            margin-bottom: 20px;
            color: #B0B0B0;
            line-height: 1.4;
            max-width: 600px;
        }

        .stTextInput > div, .stSelectbox > div {
            background: linear-gradient(145deg, #1E1E1E, #2A2A2A);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #333333;
            font-size: 14px;
            color: #E0E0E0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            margin-bottom: 15px; /* Reduced margin for compactness */
        }

        .stTextInput > div:hover, .stSelectbox > div:hover {
            background: linear-gradient(145deg, #2A2A2A, #1E1E1E);
            border-color: #666666;
        }

        .stTextInput input, .stSelectbox select {
            background: transparent;
            color: #E0E0E0;
            border: none;
            font-size: 14px;
            outline: none;
        }

        .stButton > button {
            background: linear-gradient(135deg, #FF4081, #FF80AB);
            color: white;
            font-size: 14px;
            font-weight: 600;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: background 0.3s ease, transform 0.2s ease;
            margin-top: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); /* Shadow for depth */
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #FF80AB, #FF4081);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4); /* Shadow on hover */
        }
        
        .output-container {
            margin-top: 20px;
            padding: 15px;
            background: linear-gradient(145deg, #1E1E1E, #2A2A2A);
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            line-height: 1.4;
            color: #E0E0E0;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }

        .footer {
            margin-top: 20px;
            font-size: 12px;
            color: #B0B0B0;
            text-align: center;
            padding-bottom: 10px;
            border-top: 1px solid #333333;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title and layout
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">Chat-Mate...your trusted translator </div>', unsafe_allow_html=True)
st.markdown('<div class="description">Enter your text, select a target language, and get your translation instantly.</div>', unsafe_allow_html=True)

# Text input
input_text = st.text_input("Enter the text:", placeholder="Type your text here...", key="input_text")

# Language selection
languages = {
    "Hinglish":"Hinglish",
    "Hindi":"Hindi",
    "French": "French",
    "Spanish": "Spanish",
    "German": "German",
    "Italian": "Italian",
    "Chinese": "Chinese",
    "Japanese": "Japanese",
    "Korean": "Korean"
}
target_language = st.selectbox("Select the target language", options=list(languages.values()), key="target_language")

# Display output
if input_text:
    output_text = get_groq_response(input_text, target_language)
    st.markdown(f'<div class="output-container">{output_text}</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© Raunak-2024 | Made in India</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Run FastAPI server and Streamlit app
if __name__ == "__main__":
    # Start FastAPI server in a separate process
    api_process = Process(target=run_fastapi)
    api_process.start()

