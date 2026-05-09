import requests
from bs4 import BeautifulSoup
import json
import re

BASE_URL = "https://www.gov.uk"

def get_links(index_url):
    """Fetches all links to self-assessment manual pages from the index page."""
    
    html = requests.get(index_url).text
    soup = BeautifulSoup(html, "html.parser")

    links = []
    for a in soup.select("a"):
        href = a.get("href")
        if href and "/self-assessment-manual/" in href:
            links.append(BASE_URL + href)
        
    return list(set(links))



def extract_section_code(title):
    match = re.search(r"(SAM\d+)", title)
    return match.group(1) if match else None

def extract_content(soup):
    blocks = []
    
    for tag in soup.find_all(["h2", "h3", "p", "li"]):
        text = tag.get_text(strip=True)
        if text:
            blocks.append(text)
    
    return blocks

def parse_page(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    
    title = soup.find_all("h1")[-1].text.strip()
    
    section_code = extract_section_code(title)
    
    content_blocks = extract_content(soup)

    return {
        "url": url,
        "title": title,
        "section_code": section_code,
        "content_blocks": content_blocks
    }


if __name__ == "__main__":
    source_link = "https://www.gov.uk/hmrc-internal-manuals/self-assessment-manual"
    links = get_links(source_link)
    print("=========================index=======================")
    print(links[0])

    page_data = parse_page(links[0]) # for proof of concept, we are keeping it to 1 link for now. We can loop through all the links later.

    # Trim off the boilerplate content at the beginning and end of the content blocks
    content = page_data["content_blocks"]

    start_index = content.index("Home") + 2
    end_index = content.index("Maybe") - 1

    content = content[start_index:end_index]