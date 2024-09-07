from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import uuid
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load the pre-trained model for embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')exit
# model = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory storage for processed content
processed_content = {}

class URLRequest(BaseModel):
    url: str

class PDFRequest(BaseModel):
    chat_id: str

class ChatRequest(BaseModel):
    chat_id: str
    question: str

@app.post("/process_url")
async def process_url(request: URLRequest):
    response = requests.get(request.url)
    soup = BeautifulSoup(response.content, 'html.parser')
    cleaned_content = soup.get_text(strip=True)

    chat_id = str(uuid.uuid4())
    processed_content[chat_id] = cleaned_content

    return {
        "chat_id": chat_id,
        "message": "URL content processed and stored successfully."
    }

@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...)):
    pdf_reader = PyPDF2.PdfReader(file.file)
    cleaned_text = ""

    for page in pdf_reader.pages:
        cleaned_text += page.extract_text() + "\n"

    cleaned_text = ' '.join(cleaned_text.split())  # Clean up extra spaces and line breaks
    
    chat_id = str(uuid.uuid4())
    processed_content[chat_id] = cleaned_text

    return {
        "chat_id": chat_id,
        "message": "PDF content processed and stored successfully."
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    stored_content = processed_content.get(request.chat_id)
    if not stored_content:
        return {"response": "Chat ID not found."}

    # Split the stored content into sentences or paragraphs
    sections = stored_content.split('. ')  # Splitting by sentences for simplicity

    # Generate embeddings for the user's question
    question_embedding = model.encode(request.question, convert_to_tensor=True)

    # Calculate cosine similarity for each section
    scores = []
    for section in sections:
        section_embedding = model.encode(section, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(question_embedding, section_embedding)
        scores.append((section, cosine_score.item()))

    # Find the section with the highest score
    most_relevant_section = max(scores, key=lambda x: x[1])

    return {
        "response": most_relevant_section[0] # The most relevant section
    }