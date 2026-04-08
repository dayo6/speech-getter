import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── IMDB title IDs ──────────────────────────────────────────
# Movies & TV shows — find the tt##### from any IMDB URL to add more
IMDB_TITLES = {
    # Movies
    "Blood In Blood Out": "tt0106469",
    "American Me": "tt0103671",
    "Training Day": "tt0139654",
    "Mi Vida Loca": "tt0110516",
    "Scarface": "tt0086250",
    "Friday": "tt0113118",
    "Next Friday": "tt0195945",
    "Boyz n the Hood": "tt0101507",
    "Colors": "tt0094894",
    "Menace II Society": "tt0107554",
    "End of Watch": "tt1855199",
    "Boulevard Nights": "tt0078879",
    "Don't Be a Menace": "tt0116126",
    "A Better Life": "tt1554091",
    "Selena": "tt0120094",
    "Gran Torino": "tt1205489",
    "Harsh Times": "tt0433387",
    "The Tax Collector": "tt8461224",
    # TV Shows
    "Breaking Bad": "tt0903747",
    "Queen of the South": "tt1064899",
    "Mayans M.C.": "tt5765986",
    "On My Block": "tt7078842",
    "Sons of Anarchy": "tt1124373",
    "George Lopez (show)": "tt0310095",
    "Narcos": "tt2707408",
}

# ── Wikiquote pages ─────────────────────────────────────────
# Speeches, comedians, public figures
WIKIQUOTE_PAGES = [
    "Cesar_Chavez",
    "Cheech_Marin",
    "George_Lopez",
    "Gabriel_Iglesias",
    "Cypress_Hill",
    "Ice_Cube",
    "Snoop_Dogg",
    "Tupac_Shakur",
    "Emiliano_Zapata",
]


def scrape_imdb_quotes(title_id, title_name):
    url = f"https://www.imdb.com/title/{title_id}/quotes/"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"  [!] Failed to fetch {title_name} (HTTP {resp.status_code})")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    quotes = []
    for block in soup.select('[data-testid="item-id"] .ipc-html-content-inner-div'):
        text = block.get_text(separator=" ").strip()
        if text and len(text) < 300:
            quotes.append(text)
    return quotes


def scrape_wikiquote(page_name):
    url = f"https://en.wikiquote.org/wiki/{page_name}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"  [!] Failed to fetch {page_name} (HTTP {resp.status_code})")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    quotes = []
    for li in soup.select("div.mw-parser-output > ul > li"):
        text = li.get_text(separator=" ").strip()
        # Filter to short, usable quotes
        if text and 20 < len(text) < 300:
            quotes.append(text)
    return quotes


def main():
    all_quotes = {}

    # ── IMDB: Movies & TV ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  MOVIES & TV SHOWS (IMDB)")
    print("=" * 60)

    for name, title_id in IMDB_TITLES.items():
        print(f"\n  --- {name} ---")
        quotes = scrape_imdb_quotes(title_id, name)
        if not quotes:
            print("    No quotes found.")
            continue
        all_quotes[name] = quotes
        for i, q in enumerate(quotes, 1):
            print(f"    {i}. {q}")

    # ── Wikiquote: Speeches, comedians, artists ─────────────
    print("\n" + "=" * 60)
    print("  SPEECHES, COMEDIANS & ARTISTS (Wikiquote)")
    print("=" * 60)

    for page in WIKIQUOTE_PAGES:
        label = page.replace("_", " ")
        print(f"\n  --- {label} ---")
        quotes = scrape_wikiquote(page)
        if not quotes:
            print("    No quotes found.")
            continue
        all_quotes[label] = quotes
        for i, q in enumerate(quotes, 1):
            print(f"    {i}. {q}")

    # ── Summary ─────────────────────────────────────────────
    total = sum(len(v) for v in all_quotes.values())
    print(f"\n{'=' * 60}")
    print(f"  Total: {total} quotes from {len(all_quotes)} sources")
    print(f"{'=' * 60}")

    return all_quotes


if __name__ == "__main__":
    main()
