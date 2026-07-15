from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from retrive.extraction import parse_page

BASE_URL = "https://www.gov.uk"
MANUAL_PATH = "/hmrc-internal-manuals/self-assessment-manual/"


def get_manual_links(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    links = set()

    for a in soup.select("a"):
        href = a.get("href")
        if href and MANUAL_PATH in href:
            full_url = urljoin(BASE_URL, href)
            links.add(full_url.split("#")[0])

    return list(links)

def is_contents_page(page_data):
    """
    Detect pages like:
    SAM141000 Transfer from SA to PAYE: contents

    These should be crawled further, not parsed as final content.
    """
    title = page_data["title"].lower()

    if "contents" in title:
        return True

    # If page mostly contains links and little guidance text, treat as contents page
    content_blocks = page_data.get("content_blocks", [])
    meaningful_blocks = [
        block for block in content_blocks
        if len(block.split()) > 8
    ]

    return len(meaningful_blocks) < 3



def crawl_to_content_pages(start_url, visited=None):
    if visited is None:
        visited = set()

    if start_url in visited:
        return []

    visited.add(start_url)

    page_data = parse_page(start_url)

    # If this is a real manual content page, return it
    if not is_contents_page(page_data):
        return [start_url]

    # Otherwise, keep following sublinks
    child_links = get_manual_links(start_url)

    content_pages = []

    for link in child_links:
        if link not in visited:
            content_pages.extend(
                crawl_to_content_pages(link, visited)
            )

    return content_pages

if __name__ == "__main__":
    source_link = "https://www.gov.uk/hmrc-internal-manuals/self-assessment-manual"

    top_level_links = get_manual_links(source_link)
    print(top_level_links)

    all_content_pages = []

    for link in top_level_links:
        
        print(link)
        
        pages = crawl_to_content_pages(link)
        all_content_pages.extend(pages)

    all_content_pages = list(set(all_content_pages))

    print(f"Found {len(all_content_pages)} content pages:")
    print(all_content_pages)


#     links = get_manual_links("https://www.gov.uk/hmrc-internal-manuals/self-assessment-manual")

#     print(links)
