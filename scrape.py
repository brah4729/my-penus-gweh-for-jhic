"""
Scraper for SMK Plus Pelita Nusantara school data.

The main site (smkpluspnb.sch.id) is a client-side rendered Nuxt app
(data-ssr="false") -- the HTML never contains real content, it's all
fetched client-side from a JSON API. So instead of parsing HTML, we hit
the API directly.

Real endpoint paths (confirmed via browser DevTools Network tab):
    /api/v1/public/teacher/showAll
    /api/v1/public/industry/showAll
    /api/v1/public/event/showAll
    /api/v1/public/eskul/showAll
    /api/v1/public/achievement/showAll
    /api/v1/public/news/getAll
    /api/v1/public/image/show
    /api/v1/public/testimonial/showAll

Permission note: scraping was explicitly permitted by the site owner,
and the main site's robots.txt allows crawling. The API subdomain
(api-penus.smkpluspnb.sch.id) wasn't covered by that robots.txt check --
worth a quick separate confirmation with the site owner since it's a
different host, even though it's clearly the same site's own backend.
"""

import json
import time
from pathlib import Path

import requests

API_BASE = "https://api-penus.smkpluspnb.sch.id/api/v1/public"
HEADERS = {"User-Agent": "SMK-Plus-Assistant-Bot/1.0 (school project, contact: <your email>)"}
REQUEST_DELAY_SECONDS = 1.0  # be polite even with permission

# (endpoint path, key to save under) -- action verbs are inconsistent
# across resources (showAll / getAll / show), so listed explicitly.
ENDPOINTS = [
    ("teacher/showAll", "teacher"),
    ("industry/showAll", "industry"),
    ("event/showAll", "event"),
    ("eskul/showAll", "eskul"),
    ("achievement/showAll", "achievement"),
    ("news/getAll", "news"),
    ("testimonial/showAll", "testimonial"),
    # "image/show" skipped by default -- likely needs a query param (e.g. an id)
    # rather than returning a full list. Uncomment and adjust if needed:
    # ("image/show", "image"),
]

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_endpoint(path: str):
    url = f"{API_BASE}/{path}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
    except requests.RequestException as e:
        print(f"[{path}] request failed: {e}")
        return None

    if resp.status_code != 200:
        print(f"[{path}] HTTP {resp.status_code} -- skipping")
        return None

    try:
        return resp.json()
    except ValueError:
        print(f"[{path}] response was not valid JSON -- skipping")
        return None


def main():
    results = {}

    for path, key in ENDPOINTS:
        print(f"Fetching /{path} ...")
        payload = fetch_endpoint(path)
        if payload is not None:
            results[key] = payload
            out_path = OUTPUT_DIR / f"{key}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"  saved -> {out_path}")
        time.sleep(REQUEST_DELAY_SECONDS)

    combined_path = OUTPUT_DIR / "_all_endpoints.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Combined results saved -> {combined_path}")
    print(f"Fetched {len(results)}/{len(ENDPOINTS)} endpoints successfully.")


if __name__ == "__main__":
    main()
