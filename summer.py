from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.parse import urljoin
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

url = "https://www.chess.com/news/view/announcing-collegiate-chess-league-summer-2025"
# url = "https://www.chess.com/news/view/collegiate-chess-league-2024-summer-season"

# 1. Load & chunk webpage
loader = WebBaseLoader(url)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 2. Generate embeddings & build index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 3. Set up retriever + RAG QA chain
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",  # simplest concatenate approach
    retriever=retriever
)

# Helper functions for link extraction
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def extract_links_and_text(soup, base_url):
    link_data = []
    for a in soup.find_all('a'):
        href = a.get('href')
        text = a.get_text(strip=True)
        if href and text:
            full_url = urljoin(base_url, href)
            link_data.append((full_url, text))
    return link_data

# Extract links from the webpage
resp = requests.get(url, headers={"User-Agent": "MyApp/1.0"})
soup = BeautifulSoup(resp.text, 'html.parser')
links = extract_links_and_text(soup, url)

# Format links for context
links_context = "\n".join([f"- [{text}] → {href}" for href, text in links])

# # Query 1: Championships
# print("=== CHAMPIONSHIPS ===")
# query1 = '''
# Look specifically in the "Schedule" section of the webpage to identify what events are scheduled for the summer.
# What are the different kinds of championships and arenas offered this summer?
# A qualifier and final is considered to be part of a championship, not as separate things.
# '''
# response1 = qa.invoke(query1)
# print(response1['result'])

# Query 2: Schedule for each championship
print("\n=== SCHEDULES ===")
query2 = '''
What is the schedule for each championship? Include dates and times when available.
Provide the schedule for each championship type separately.
Seperately, provide the schedule for weekly prize arenas.
'''
response2 = qa.invoke(query2)
print(response2['result'])

# # Query 3: Registration links and club pages
# print("\n=== REGISTRATION & CLUB LINKS ===")
# query3 = f'''
# Based on the following list of links and their text from the webpage:

# {links_context}

# For each championship type, identify:
# 1. The Google form registration link for that tournament
# 2. The chess.com club page link for that tournament

# Present this information clearly for each championship.
# '''
# response3 = qa.invoke(query3)
# print(response3['result'])

# Query 4: Championship-related links
print("\n=== CHAMPIONSHIP LINKS ===")
query4 = f'''
From the following list of links extracted from the webpage:

{links_context}

Filter and identify all links that are related to championships, tournaments, registration, or club pages. 
Look for links that contain keywords like:
- "championship"
- "tournament" 
- "registration"
- "form"
- "club"
- "arena"
- "chess.com/club"
- "google.com/forms"

Group them by category (registration forms, club pages, tournament pages, etc.) and provide a clear summary.
'''
response4 = qa.invoke(query4)
print(response4['result'])

# Query 5: Combine tournament information into structured format
print("\n=== TOURNAMENT INFORMATION SUMMARY ===")
query5 = f'''
Based on the following information:

SCHEDULES:
{response2['result']}

CHAMPIONSHIP LINKS:
{response4['result']}

Create a structured list of tournament objects with the following format:

For Championships:
- Tournament Name: [name]
- Registration Deadline: [deadline with time and timezone]
- Registration Form: [URL]
- Club Page: [URL if available]
- Rounds:
  - Round Name: [name] - Date: [date and time]
  - Round Name: [name] - Date: [date and time]

For Prize Arenas:
- Tournament Name: Weekly Prize Arenas
- Registration Form: [URL]
- Schedule: [recurring schedule with dates and times]

Present this information in a clear, structured format for each tournament.
'''

response5 = qa.invoke(query5)
print(response5['result'])

# Query 6: Structure tournament information into JSON objects
print("\n=== STRUCTURED TOURNAMENT OBJECTS ===")
query6 = f'''
Based on the following tournament information:

{response5['result']}

Create a structured JSON array of tournament objects. Each tournament object should have:
- "tournament_name": string
- "registration_deadline": string (include date, time, and timezone in PT)
- "rounds": array of objects with "title" and "date" fields

Format dates as "YYYY-MM-DD HH:MM AM/PM PT" when possible.
If exact dates/times are not available, use the information provided.

Return only valid JSON format like this:
[
  {{
    "tournament_name": "Championship Name",
    "registration_deadline": "2025-01-15 11:59 PM PT",
    "rounds": [
      {{"title": "Qualifier", "date": "2025-01-20 10:00 AM PT"}},
      {{"title": "Final", "date": "2025-01-27 02:00 PM PT"}}
    ]
  }}
]
'''

# Use OpenAI to structure the response
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai_api_model_name = os.getenv("OPENAI_API_MODEL_NAME", "gpt-4o")

if not openai_api_key:
    raise ValueError("Please add your OpenAI API key to the .env file.")

# Configure OpenAI API client
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

try:
    response = client.chat.completions.create(
        model=openai_api_model_name,
        messages=[{"role": "user", "content": query6}]
    )
    
    structured_response = response.choices[0].message.content
    # get the json response
    structured_response = json.loads(response.choices[0].message.content)
    print(structured_response)
    # Try to parse as JSON to validate format
    # try:
    #     tournament_objects = json.loads(structured_response)
    #     print(json.dumps(tournament_objects, indent=2))
    # except json.JSONDecodeError:
    #     print("Raw response:")
    #     print(structured_response)
        
except Exception as e:
    print(f"Error structuring tournament data: {e}")
