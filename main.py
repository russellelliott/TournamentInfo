import os

# Disable parallelism for Hugging Face tokenizers to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import json
from openai import OpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
from typing import List

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API credentials from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai_api_model_name = os.getenv("OPENAI_API_MODEL_NAME", "gpt-4o")

if not openai_api_key:
    raise ValueError("Please add your OpenAI API key to the .env file.")

# Configure OpenAI API client
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

# Initialize FastAPI app
app = FastAPI()

@app.post("/query-openai")
def query_openai(prompt: str):
    """
    Endpoint to query OpenAI with a given prompt.
    """
    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model=openai_api_model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract the AI-generated content
        ai_response = response.choices[0].message.content
        return JSONResponse(content={"response": ai_response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying OpenAI: {str(e)}")

PDF_PATH = "CCLSpring2025.pdf"

def reformat_text(text: str):
    """
    Reformat text by replacing newlines with spaces.
    """
    return text.replace("\n", " ")

def chunk_pdf(doc, chunk_size=6):
    """
    Chunk the PDF into smaller text segments.
    """
    text_per_page = []
    for page_number, page in enumerate(doc):
        sentence = ''
        accumulated_text = ''
        for i, text in enumerate(page.get_text("text").split("\n")):
            if text.upper() == text:  # Remove headers
                text = text.replace(text, " ")
            accumulated_text += text
            if i > 0 and i % chunk_size == 0:
                sentence += accumulated_text
                sentence = reformat_text(sentence)
                text_per_page.append({"Text": sentence, "Page_#": page_number})
                accumulated_text = ''
                sentence = ''
    return text_per_page

def retrieve_relevant_chunks(query: str, chunks: List[dict], top_k: int = 3):
    """
    Retrieve the most relevant chunks based on the query.
    """
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_embeddings = model.encode([chunk["Text"] for chunk in chunks], convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_results = scores.topk(k=top_k)
    relevant_chunks = [chunks[idx] for idx in top_results.indices]
    return relevant_chunks

@app.post("/ask-question")
def ask_question(question: str = Query(..., description="The question to ask based on the PDF content")):
    """
    Endpoint to answer a question based on the PDF content.
    """
    try:
        # Open and chunk the PDF
        doc = fitz.open(PDF_PATH)
        chunks = chunk_pdf(doc)

        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(question, chunks)

        # Combine relevant chunks into a single context
        context = " ".join([chunk["Text"] for chunk in relevant_chunks])

        # Query OpenAI with the context and question
        system_prompt = f"""You are a helpful assistant. Use the following context to answer the question:
Context: {context}
Question: {question}
Answer the question based only on the context provided."""
        response = client.chat.completions.create(
            model=openai_api_model_name,
            messages=[{"role": "system", "content": system_prompt}]
        )

        # Extract the AI-generated answer
        ai_response = response.choices[0].message.content
        return JSONResponse(content={"answer": ai_response, "context_used": context})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/ask-multiple-questions")
def ask_multiple_questions():
    """
    Endpoint to ask multiple predefined questions and use OpenAI to generate a structured response.
    """
    try:
        # List of predefined questions
        questions = [
            "What date does registration open?",
            "What date does registration close?",
            "When is the schedule released?",
            "List off all the events in the section 'schedule: '",
            "What time of day are rounds scheduled for the regular season for group A teams? Please list off only the time in PT.",
            "List off the dates for the final rounds; round 1, quarterfinal, semifinal, final/3rd place"
        ]

        # Collect answers for each question
        combined_answers = []
        for question in questions:
            # Call the ask_question function for each question
            response = ask_question(question=question)
            response_content = response.body.decode("utf-8")  # Decode the JSONResponse body
            response_data = json.loads(response_content)  # Parse the JSON string
            combined_answers.append(f"Q: {question}\nA: {response_data['answer']}\n")

        # Combine all answers into a single chunk of text
        final_response = "\n".join(combined_answers)

        # Use OpenAI to generate the structured response
        system_prompt = f"""You are a helpful assistant. Based on the following context, generate a structured JSON object.
The JSON object should contain:
1. A "logistics" section with "registration_open", "registration_close", and "schedule_release" fields, each having a "title" and "date".
2. A "regular_season" section with a list of events, each having "title" and "date" fields.
3. A "playoff_rounds" section with a list of events, each having "title" and "date" fields.
4. Dates should be formatted as "YYYY-MM-DD HH:MM AM/PM PT".
5. Include all events mentioned in the context, separating logistics, regular season rounds, and playoff rounds.

Context:
{final_response}

The output must be a valid JSON object like this:
{{
    "logistics": [
        {{"title": "Registration Opens", "date": "2025-01-01 12:00 AM PT"}},
        {{"title": "Registration Closes", "date": "2025-03-01 11:59 PM PT"}},
        {{"title": "Schedule Release", "date": "2025-03-05 12:00 AM PT"}}
    ],
    "regular_season": [
        {{"title": "Regular Season Round 1", "date": "2025-03-10 10:00 AM PT"}},
        {{"title": "Regular Season Round 2", "date": "2025-03-17 02:00 PM PT"}}
    ],
    "playoff_rounds": [
        {{"title": "Round 1", "date": "2025-03-24 10:00 AM PT"}},
        {{"title": "Quarterfinal", "date": "2025-03-31 02:00 PM PT"}},
        {{"title": "Semifinal", "date": "2025-04-07 10:00 AM PT"}},
        {{"title": "Final/3rd Place", "date": "2025-04-14 02:00 PM PT"}}
    ]
}}"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model=openai_api_model_name,
            messages=[{"role": "system", "content": system_prompt}]
        )

        # Extract the AI-generated structured response
        structured_response = json.loads(response.choices[0].message.content)

        # Return the structured response
        return JSONResponse(content=structured_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing multiple questions: {str(e)}")
