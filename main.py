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
import requests
from bs4 import BeautifulSoup
from datetime import datetime

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
def ask_question(
    question: str = Query(..., description="The question to ask based on the PDF content"),
    chunk_size: int = 6  # Default chunk size is 6
):
    """
    Endpoint to answer a question based on the PDF content.
    """
    try:
        # Open and chunk the PDF
        doc = fitz.open(PDF_PATH)
        chunks = chunk_pdf(doc, chunk_size=int(chunk_size))  # Ensure chunk_size is an integer

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

@app.post("/get-tournament-info")
def get_tournament_info(
    season: str = Query(..., description="The season (Spring or Fall)"),
    year: int = Query(..., description="The year of the tournament")
):
    """
    Endpoint to retrieve tournament PDF, registration form links, and Fair Play Agreement link for a given season and year.
    """
    try:
        # Validate season input
        if season.lower() not in ["spring", "fall"]:
            raise HTTPException(status_code=400, detail="Invalid season. Must be 'Spring' or 'Fall'.")

        # Construct the search query
        query = f"Collegiate Chess League {season.capitalize()} {year} site:chess.com"

        # Google Search API setup
        API_KEY = os.getenv("API_KEY")
        SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
        if not API_KEY or not SEARCH_ENGINE_ID:
            raise HTTPException(status_code=500, detail="Google API credentials are missing.")

        # Perform Google search
        def google_search(query, api_key, cse_id, num=10):
            url = 'https://www.googleapis.com/customsearch/v1'
            params = {
                'q': query,
                'key': api_key,
                'cx': cse_id,
                'num': num
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            links = [item['link'] for item in results.get('items', [])]
            return links

        search_results = google_search(query, API_KEY, SEARCH_ENGINE_ID)

        # Extract relevant links
        def extract_links_from_page(url, keywords):
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                links = []
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if any(keyword in href for keyword in keywords):
                        links.append(href)
                return links
            except requests.RequestException as e:
                print(f"Error fetching {url}: {e}")
                return []

        def find_fair_play_agreement(links):
            """
            Search for a link titled 'Fair Play Agreement' in the provided links.
            """
            for link in links:
                try:
                    response = requests.get(link)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for a_tag in soup.find_all('a', href=True):
                        if "fair play agreement" in a_tag.text.lower():
                            return a_tag['href']
                except requests.RequestException as e:
                    print(f"Error fetching {link}: {e}")
            return None

        tournament_pdf_link = None
        registration_form_link = None
        used_links = []  # Track only the links used to retrieve the required information

        for link in search_results:
            extracted_links = extract_links_from_page(link, ['drive.google.com', 'forms.gle'])
            if extracted_links:
                used_links.append(link)  # Add the Chess.com link to the used links list if it contains relevant links
            for extracted_link in extracted_links:
                if 'drive.google.com' in extracted_link and not tournament_pdf_link:
                    tournament_pdf_link = extracted_link
                elif 'forms.gle' in extracted_link and not registration_form_link:
                    registration_form_link = extracted_link
            if tournament_pdf_link and registration_form_link:
                break

        # Filter used links to include only those that contributed to the results
        filtered_used_links = [
            link for link in used_links if link in search_results
        ]

        # Search for the Fair Play Agreement link in the filtered links
        fairplay_agreement_link = find_fair_play_agreement(filtered_used_links)

        # Return the results
        return JSONResponse(content={
            "tournament_pdf_link": tournament_pdf_link or "Not found",
            "registration_form_link": registration_form_link or "Not found",
            "fairplay_agreement_link": fairplay_agreement_link or "Not found",
            "used_chess_com_links": filtered_used_links  # Only include links used to retrieve the required information
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving tournament information: {str(e)}")

@app.post("/validate-player")
def validate_player(
    username: str = Query(..., description="The Chess.com username of the player"),
    min_account_age_days: int = Query(0, description="Minimum account age in days (default is 0)"),
    min_blitz_games: int = Query(0, description="Minimum number of blitz games played (default is 0)")
):
    """
    Endpoint to validate if a Chess.com player satisfies the requirements for account age and blitz games played.
    """
    try:
        # Define headers with a User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # --- Get account creation date and compute age in days ---
        profile_url = f"https://api.chess.com/pub/player/{username}"
        profile_response = requests.get(profile_url, headers=headers)
        if profile_response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Player '{username}' not found on Chess.com.")
        profile_data = profile_response.json()

        joined_timestamp = profile_data.get("joined")
        if not joined_timestamp:
            raise HTTPException(status_code=400, detail=f"Could not retrieve account creation date for '{username}'.")
        joined_date = datetime.fromtimestamp(joined_timestamp)
        now = datetime.now()
        account_age_days = (now - joined_date).days

        # --- Get number of blitz games played ---
        blitz_url = f"https://www.chess.com/callback/live/stats/{username}/chart?type=blitz"
        blitz_response = requests.get(blitz_url, headers=headers)
        if blitz_response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Could not retrieve blitz games data for '{username}'.")
        blitz_data = blitz_response.json()
        num_blitz_games = len(blitz_data)

        # --- Validate requirements ---
        account_age_valid = account_age_days >= min_account_age_days
        blitz_games_valid = num_blitz_games >= min_blitz_games

        # --- Return validation results ---
        return JSONResponse(content={
            "username": username,
            "account_age_days": account_age_days,
            "num_blitz_games": num_blitz_games,
            "requirements_met": account_age_valid and blitz_games_valid,
            "details": {
                "account_age_valid": account_age_valid,
                "blitz_games_valid": blitz_games_valid
            }
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating player '{username}': {str(e)}")

@app.post("/get-player-requirements")
def get_player_requirements():
    """
    Endpoint to retrieve player requirements for the tournament.
    """
    try:
        # Define the questions to ask
        questions = [
            {"question": "What is the minimum account age?", "chunk_size": 6},
            {"question": "What is the minimum number of blitz games played?", "chunk_size": 8}
        ]

        # Collect answers for each question
        answers = {}
        for q in questions:
            response = ask_question(question=q["question"], chunk_size=q["chunk_size"])
            response_content = response.body.decode("utf-8")  # Decode the JSONResponse body
            response_data = json.loads(response_content)  # Parse the JSON string
            answers[q["question"]] = response_data["answer"]

        # Use OpenAI to extract numeric values and format the response
        system_prompt = f"""You are a helpful assistant. Based on the following context, extract the numeric values for the minimum account age (in days) and the minimum number of blitz games played. 
Return the output as a JSON object with the following format:
{{
    "minimum_account_age": <numeric_value>,
    "minimum_games": <numeric_value>
}}

Context:
{json.dumps(answers)}

The output must be a valid JSON object."""
        response = client.chat.completions.create(
            model=openai_api_model_name,
            messages=[{"role": "system", "content": system_prompt}]
        )

        # Extract the AI-generated structured response
        structured_response = json.loads(response.choices[0].message.content)

        # Return the structured response
        return JSONResponse(content=structured_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving player requirements: {str(e)}")
