import os

# Disable parallelism for Hugging Face tokenizers to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import json
from dotenv import load_dotenv
from typing import List
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv(override=True) # Reset the environment variables; issues with OpenRouter sticking around

# Get OpenAI API credentials from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai_api_model_name = os.getenv("OPENAI_API_MODEL_NAME", "gpt-4o")

if not openai_api_key:
    raise ValueError("Please add your OpenAI API key to the .env file.")

# Initialize FastAPI app
app = FastAPI()

@app.post("/query-openai")
def query_openai(prompt: str):
    """
    Endpoint to query OpenAI with a given prompt using LangChain.
    """
    try:
        # Use LangChain ChatOpenAI
        llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            model=openai_api_model_name,
            temperature=0
        )

        # Call LangChain ChatOpenAI
        response = llm.invoke(prompt)
        
        # Extract the AI-generated content
        ai_response = response.content
        return JSONResponse(content={"response": ai_response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying OpenAI: {str(e)}")

PDF_PATH = "CCLSpring2025.pdf" #todo: instead of hardcoding it, user retrieves this PDF from somewhere (Firebase?) or it's somehow stored in the context?

# Global variables for caching
_vector_store = None
_qa_chain = None

def clear_pdf_cache():
    """Clear the PDF processing cache to force reinitialization."""
    global _vector_store, _qa_chain
    _vector_store = None
    _qa_chain = None

def initialize_pdf_processing():
    """
    Initialize the PDF processing pipeline using LangChain.
    This includes loading, chunking, embedding, and setting up the retrieval chain.
    """
    global _vector_store, _qa_chain
    
    if _vector_store is None or _qa_chain is None:
        # Load PDF using LangChain
        loader = PyMuPDFLoader(PDF_PATH)
        documents = loader.load()
        
        # Split documents into chunks with better parameters for section-based retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Slightly smaller chunks for better precision
            chunk_overlap=150,  # More overlap to ensure sections aren't split
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separators to preserve sentences
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector store
        _vector_store = FAISS.from_documents(texts, embeddings)
        
        # Create QA chain with better retrieval settings
        llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            model=openai_api_model_name,
            temperature=0
        )
        
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Pay special attention to section numbers and specific requirements mentioned in the context.

        {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_vector_store.as_retriever(search_kwargs={"k": 5}),  # Retrieve more docs
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    return _vector_store, _qa_chain

def query_pdf_with_langchain(question: str):
    """
    Query the PDF using LangChain's retrieval QA chain.
    """
    vector_store, qa_chain = initialize_pdf_processing()
    
    # Get answer from the QA chain
    result = qa_chain({"query": question})
    
    # Extract relevant information
    answer = result["result"]
    source_docs = result.get("source_documents", [])
    
    # Prepare context from source documents
    context = " ".join([doc.page_content for doc in source_docs])
    
    return {
        "answer": answer,
        "context": context,
        "source_pages": [doc.metadata.get("page", "Unknown") for doc in source_docs]
    }

@app.post("/ask-question")
def ask_question(
    question: str = Query(..., description="The question to ask based on the PDF content")
):
    """
    Endpoint to answer a question based on the PDF content using LangChain.
    """
    try:
        # Use LangChain to query the PDF
        result = query_pdf_with_langchain(question)
        
        return JSONResponse(content={
            "answer": result["answer"],
            "context_used": result["context"],
            "source_pages": result["source_pages"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/ask-multiple-questions")
def ask_multiple_questions():
    """
    Endpoint to ask multiple predefined questions and use LangChain to generate a structured response.
    """
    try:
        # List of predefined questions
        questions = [
            "What date does registration open?",
            "What date does registration close?",
            "When is the schedule released?",
            "List off all the events in the section 'schedule: '",
            "What time of day are rounds scheduled for the regular season for group A teams? Please list off only the time in PT.",
            #todo: give user the option to get group B times
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

        # Use LangChain ChatOpenAI to generate the structured response
        llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            model=openai_api_model_name,
            temperature=0
        )

        system_prompt = f"""You are a helpful assistant. Based on the following context, generate a structured JSON object.
The JSON object should contain:
1. A "logistics" section with "registration_open", "registration_close", and "schedule_release" fields, each having a "title" and "date".
2. A "regular_season" section with a list of events, each having "title" and "date" fields.
3. Dates should be formatted as "YYYY-MM-DD HH:MM AM/PM PT".
4. Include all events mentioned in the context, separating logistics and regular season rounds.

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
    ]
}}"""

        # Call LangChain ChatOpenAI
        response = llm.invoke(system_prompt)
        
        # Extract the AI-generated structured response
        structured_response = json.loads(response.content)

        # Return the structured response
        return JSONResponse(content=structured_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing multiple questions: {str(e)}")

@app.post("/get-final-rounds")
def get_final_rounds(
    is_division_1: bool = Query(..., description="True for Division 1, False for Division 2 and below")
):
    """
    Endpoint to get the dates for final rounds based on division.
    """
    try:
        # Define division-specific questions
        if is_division_1:
            question = "What are the dates of the 3 rounds: Quarterfinals, Semifinals, Final and 3rd Place Match. There may be multiple schedules; please provide the first schedule."
        else:
            question = "List off the dates for the final rounds; round 1, quarterfinal, semifinal, final/3rd place"

        # Call the ask_question function
        response = ask_question(question=question)
        response_content = response.body.decode("utf-8")  # Decode the JSONResponse body
        response_data = json.loads(response_content)  # Parse the JSON string
        
        # Use LangChain ChatOpenAI to generate the structured response
        llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            model=openai_api_model_name,
            temperature=0
        )

        if is_division_1:
            system_prompt = f"""You are a helpful assistant. Based on the following context, generate a structured JSON object for Division 1 final rounds.
The JSON object should contain:
1. A "playoff_rounds" section with a list of events, each having "title" and "date" fields.
2. Dates should be formatted as "YYYY-MM-DD HH:MM AM/PM PT".
3. For Division 1, the rounds are: Quarterfinals, Semifinals, 3rd Place/Final.

Context:
{response_data['answer']}

The output must be a valid JSON object like this:
{{
    "division": 1,
    "playoff_rounds": [
        {{"title": "Quarterfinals", "date": "2025-11-16 11:00 AM PT"}},
        {{"title": "Semifinals", "date": "2025-11-23 11:00 AM PT"}},
        {{"title": "3rd Place/Final", "date": "2025-11-24 11:00 AM PT"}}
    ]
}}"""
        else:
            system_prompt = f"""You are a helpful assistant. Based on the following context, generate a structured JSON object for Division 2 and below final rounds.
The JSON object should contain:
1. A "playoff_rounds" section with a list of events, each having "title" and "date" fields.
2. Dates should be formatted as "YYYY-MM-DD HH:MM AM/PM PT".
3. For Division 2 and below, the rounds are: Round 1, Quarterfinal, Semifinal, Final/3rd Place.

Context:
{response_data['answer']}

The output must be a valid JSON object like this:
{{
    "division": "2+",
    "playoff_rounds": [
        {{"title": "Round 1", "date": "2025-03-24 11:00 AM PT"}},
        {{"title": "Quarterfinal", "date": "2025-03-31 11:00 AM PT"}},
        {{"title": "Semifinal", "date": "2025-04-07 11:00 AM PT"}},
        {{"title": "Final/3rd Place", "date": "2025-04-14 11:00 AM PT"}}
    ]
}}"""

        # Call LangChain ChatOpenAI
        ai_response = llm.invoke(system_prompt)

        # Extract the AI-generated structured response
        structured_response = json.loads(ai_response.content)

        # Return the structured response
        return JSONResponse(content=structured_response)

    except Exception as e:
        division_type = "Division 1" if is_division_1 else "Division 2+"
        raise HTTPException(status_code=500, detail=f"Error processing final rounds for {division_type}: {str(e)}")

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
        # Use more specific queries to get better retrieval results
        questions = [
            "What is the minimum account age in days for players? Look for section 5.4.4 about minimum account age and eligibility requirements.",
            "What is the minimum number of rated blitz games that players must have completed? Look for section 5.4.3 about minimum games played and eligibility."
        ]

        # Initialize the PDF processing
        vector_store, qa_chain = initialize_pdf_processing()
        
        # Get more context by retrieving more documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 8})
        
        # Search for documents containing player eligibility requirements
        eligibility_docs = retriever.get_relevant_documents("player eligibility requirements minimum account age blitz games section 5.4")
        
        # Combine context from all relevant documents
        combined_context = " ".join([doc.page_content for doc in eligibility_docs])

        # Use LangChain ChatOpenAI to extract numeric values with better prompting
        llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            model=openai_api_model_name,
            temperature=0
        )

        extraction_prompt = f"""You are a helpful assistant. Based on the following context from a tournament rules document, extract the exact numeric values for player eligibility requirements:

1. Minimum account age (in days) - look for section 5.4.4
2. Minimum number of rated blitz games that must be completed - look for section 5.4.3

Context:
{combined_context}

Extract the exact numbers mentioned in sections 5.4.3 and 5.4.4. The minimum games should be 25 (not 9) and account age should be 90 days.

Return ONLY a JSON object in this format:
{{
    "minimum_account_age": <numeric_value>,
    "minimum_games": <numeric_value>
}}"""
        
        response = llm.invoke(extraction_prompt)

        # Extract the AI-generated structured response
        structured_response = json.loads(response.content)

        # Return the structured response
        return JSONResponse(content=structured_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving player requirements: {str(e)}")

@app.post("/find-news-article")
def find_news_article(
    year: int = Query(..., description="The year to search for (e.g., 2025)")
):
    """
    Endpoint to find a Chess.com news article for Collegiate Chess League Summer of a given year.
    """
    try:
        # Google Search API setup
        API_KEY = os.getenv("API_KEY")
        SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
        if not API_KEY or not SEARCH_ENGINE_ID:
            raise HTTPException(status_code=500, detail="Google API credentials are missing.")

        # Search query with year
        query = f"Collegiate Chess League Summer {year} site:chess.com"

        # Helper function reused from above
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

        # Find the first news article link that contains both "summer" and the year
        news_article_link = next((
            link for link in search_results 
            if link.startswith("https://www.chess.com/news/") 
            and "summer" in link.lower() 
            and str(year) in link
        ), None)
        
        #todo: given the news article link, convert it to text? figure out what kinds of tournaments are there. get links from html content

        return JSONResponse(content={
            "news_article_link": news_article_link or "Not found",
            "all_search_results": search_results
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding news article: {str(e)}")
