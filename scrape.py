import time
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

HEADERS = {"User-Agent": "SMK-Plus-Assistant-Bot/1.0 (school project, contact: <your email>)"}
SITEMAP_URL = "https://www.smkpluspnb.sch.id/sitemap.xml"

def get_urls_from_sitemap(sitemap_url):
    resp = requests.get(sitemap_url, headers=HEADERS, timeout=10)
    root = ET.fromstring(resp.content)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text for loc in root.findall(".//ns:loc", ns)]

def scrape_page(url):
    resp = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return {"url": url, "text": text}

urls = get_urls_from_sitemap(SITEMAP_URL)
results = []
for url in urls:
    try:
        results.append(scrape_page(url))
        time.sleep(1)  # be polite even with permission
    except Exception as e:
        print(f"failed {url}: {e}")

import json
with open("data/raw_pages.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")