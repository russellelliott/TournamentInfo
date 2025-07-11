from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.parse import urljoin

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

REGISTRATION FORMS:
1. [Bullet Registration] → https://docs.google.com/forms/d/e/1FAIpQLSfO8zpBifqShGpBf_spuuV1oaTRZ3_bMBnpSeP5kdMda-rGBA/viewform
2. [Team Chess Battle Registration] → https://docs.google.com/forms/d/e/1FAIpQLSd36yXVZ3hegu_bW4VPd_rLZNHR0DbvCow0ttpYPOyKzbXwHQ/viewform
3. [Weekly Prize Arenas] → https://docs.google.com/forms/d/e/1FAIpQLSfXasE8xIS9lQnK85lWjFoABk_TUbkRGwECqo4LLzysTySnVg/viewform

CLUB PAGES:
1. [bullet club page] → https://www.chess.com/club/collegiate-chess-league-summer-2025-bullet-championship
2. [CCL Club Page] → https://www.chess.com/join/collegiate-chess-league

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
