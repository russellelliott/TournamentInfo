import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

#Google Search API stuff in ENV file
load_dotenv()
API_KEY = os.getenv("API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

def google_search(query, api_key, cse_id, num=10):
    """Performs a Google Custom Search and returns the result URLs."""
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

def extract_links_from_page(url, keywords):
    """Extracts links containing specific keywords from a webpage."""
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

def main():
    query = "Collegiate Chess League Spring 2026 site:chess.com"
    print(f"Searching for: {query}")
    search_results = google_search(query, API_KEY, SEARCH_ENGINE_ID)

    tournament_pdf_link = None
    registration_form_link = None

    for link in search_results:
        print(f"Processing: {link}")
        extracted_links = extract_links_from_page(link, ['drive.google.com', 'forms.gle'])
        for extracted_link in extracted_links:
            if 'drive.google.com' in extracted_link and not tournament_pdf_link:
                tournament_pdf_link = extracted_link
                print(f"Found tournament PDF link: {tournament_pdf_link}")
            elif 'forms.gle' in extracted_link and not registration_form_link:
                registration_form_link = extracted_link
                print(f"Found registration form link: {registration_form_link}")
        if tournament_pdf_link and registration_form_link:
            break

    print("\nSummary:")
    print(f"Tournament PDF Link: {tournament_pdf_link or 'Not found'}")
    print(f"Registration Form Link: {registration_form_link or 'Not found'}")

if __name__ == "__main__":
    main()
