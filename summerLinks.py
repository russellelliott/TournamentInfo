import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.parse import urljoin

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def get_visible_text(soup):
    texts = soup.find_all(text=True)
    visible_texts = filter(tag_visible, texts)
    return " ".join(t.strip() for t in visible_texts)

def extract_links_and_text(soup, base_url):
    link_data = []
    for a in soup.find_all('a'):
        href = a.get('href')
        text = a.get_text(strip=True)
        if href and text:
            full_url = urljoin(base_url, href)
            link_data.append((full_url, text))
    return link_data

# Usage
url = "https://www.chess.com/news/view/announcing-collegiate-chess-league-summer-2025"
resp = requests.get(url, headers={"User-Agent": "MyApp/1.0"})
soup = BeautifulSoup(resp.text, 'html.parser')

# 1. Visible page text
visible_text = get_visible_text(soup)
print("PAGE TEXT:")
print(visible_text[:200], "...")  # preview first 200 chars

# 2. Extract links + visible anchor text
links = extract_links_and_text(soup, url)
print("\nLINKS FOUND:")
for href, text in links:
    print(f"- [{text}] → {href}")

#registration link is:

'''
2024
url = https://www.chess.com/news/view/collegiate-chess-league-2024-summer-season

- [Bullet Registration] → https://forms.gle/7sbgiidjMrrsjnrR7
- [Bughouse Registration] → https://forms.gle/Gm8oj35Wpd6qqzUB9
- [Team Chess Battle Registration] → https://forms.gle/f8TnR2Jfg3FbiNZX6

(no prize arena?)

each tournament type also has a club page
- [bullet] → https://www.chess.com/club/collegiate-chess-league-summer-2024-bullet-championship
- [Bughouse] → https://www.chess.com/club/collegiate-chess-league-summer-2024-bughouse-championship
'''

'''
2025
- [Bullet Registration] → https://docs.google.com/forms/d/e/1FAIpQLSfO8zpBifqShGpBf_spuuV1oaTRZ3_bMBnpSeP5kdMda-rGBA/viewform
- [Team Chess Battle Registration] → https://docs.google.com/forms/d/e/1FAIpQLSd36yXVZ3hegu_bW4VPd_rLZNHR0DbvCow0ttpYPOyKzbXwHQ/viewform
- [Weekly Prize Arenas] → https://docs.google.com/forms/d/e/1FAIpQLSfXasE8xIS9lQnK85lWjFoABk_TUbkRGwECqo4LLzysTySnVg/viewform

bullet club page
- [bullet club page] → https://www.chess.com/club/collegiate-chess-league-summer-2025-bullet-championship

not ccl-specific
- [Chess.com Bughouse Chess Championship] → https://www.chess.com/events/2025-chesscom-bughouse-championship
'''
