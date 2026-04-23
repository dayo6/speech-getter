import os
import json
import re
import argparse
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = input("Enter your OpenRouter API key: ").strip()


PROMPT = """You are a YouTube search query generator for a dark Sureno type beat producer.

I need YouTube search queries that will surface videos containing raw, unfiltered speech
that would work as intro samples for dark, aggressive beats.

I'm NOT looking for specific videos — I want SEARCH QUERIES that I can paste into YouTube
to find the kind of content I need. The more specific and creative the query, the better.

CONTENT I'M LOOKING FOR:
- Chicano/Sureno/Latino culture FIRST. Mexican-American street life, barrio confrontations,
  varrio politics, Sureno rap artists going off. This is the PRIMARY focus.
- Real altercations, confrontations, arguments, and heated moments caught on camera.
- Livestream rants, IG live beefs, Facebook live arguments, studio sessions with wild talk.
- Anti-police sentiment — people going off on cops, disrespecting law enforcement,
  confronting officers, ranting about crooked police, "fuck 12" energy.
- Rivals disrespected — talking down on opps, mocking rivals, disrespecting enemies.
- Hood viral moments — someone getting checked, called out, or going crazy on camera.
- NO movies, NO TV shows, NO music, NO scripted content.

WHAT MAKES A GOOD SEARCH QUERY:
- Specific enough to surface niche content, not generic results
- Includes modifiers: "confronts", "IG live", "heated", "argument", "fight", "calls out"
- Targets specific scenes, including but not excluded to: someone getting confronted, someone going off in the studio,
  a heated exchange, someone ranting about opps/fakes/snitches/cops

CATEGORIES TO COVER (generate queries across ALL of these — at least HALF should be Chicano/Sureno specific):
- Chicano/Sureno rap artists from the list above — interviews, IG lives, beefs, studio sessions, confrontations
- Barrio confrontations, varrio politics, Mexican-American street life caught on camera
- Dissing rival gangs, mocking opps, talking down on enemies
- Hood altercations and confrontations caught on camera
- Instagram/Facebook live beefs, rants, and arguments
- Someone getting checked/pressed/called out on camera
- Street arguments that went viral
- People ranting about snitches, fakes, opps, haters, police

Respond ONLY with a valid JSON array of objects. No markdown, no code fences.
Each object must have:
{{
  "query": "the exact YouTube search query to paste",
  "category": "what kind of content this should surface",
  "why": "what kind of speech/moment this query should find"
}}

Give me {num_queries} search queries."""

# Parse flags early so we can use --num-queries in the prompt
parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default=None, help="Output file path")
parser.add_argument("--num-queries", "-n", type=int, default=30, help="Number of search queries to generate")
args = parser.parse_args()

PROMPT = PROMPT.format(num_queries=args.num_queries)

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": "You are a JSON API. Respond ONLY in valid English JSON. No markdown, no explanation."},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0.8,
        "max_tokens": 8000,
    },
)

if response.status_code != 200:
    print(f"Error {response.status_code}: {response.text}")
    raise SystemExit(1)

data = response.json()
content = data["choices"][0]["message"]["content"]

# Save raw response for debugging
raw_path = f"debug_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(raw_path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"[*] Raw response saved to {raw_path}")

# Strip reasoning blocks
if "<think>" in content:
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

# Strip markdown code fences
content = content.strip()
if content.startswith("```"):
    content = content.split("\n", 1)[1]
if content.endswith("```"):
    content = content.rsplit("```", 1)[0]
content = content.strip()

# Replace non-ASCII
content = content.replace('\u2018', "'").replace('\u2019', "'")
content = content.replace('\u201c', '"').replace('\u201d', '"')
content = content.replace('\u2014', '-').replace('\u2013', '-')
content = content.replace('\u2026', '...')
content = re.sub(r'[^\x00-\x7F]+', '', content)

# Extract JSON array
if not content.startswith("["):
    start = content.find("[")
    if start != -1:
        content = content[start:]
if not content.endswith("]"):
    end = content.rfind("]")
    if end != -1:
        content = content[:end + 1]

try:
    suggestions = json.loads(content)
except json.JSONDecodeError as e:
    raw_path = f"debug_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Failed to parse JSON: {e}")
    print(f"Processed content saved to {raw_path}")
    raise SystemExit(1)

# Convert to the format download_clips.py expects
# Each query becomes a "suggestion" with youtube_search field
intros = []
for i, s in enumerate(suggestions, 1):
    intros.append({
        "youtube_search": s.get("query", ""),
        "source_title": s.get("query", f"query_{i}")[:60],
        "type": s.get("category", ""),
        "vibe": "DARK",
        "why_it_works": s.get("why", ""),
    })

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = args.output or f"intros_{timestamp}.json"
os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(intros, f, indent=2, ensure_ascii=False)

# Print summary
print(f"\nSaved {len(intros)} search queries to {output_file}\n")
for i, s in enumerate(intros, 1):
    query = s.get("youtube_search", "?")
    category = s.get("type", "?")
    why = s.get("why_it_works", "")
    print(f"  {i:2}. [{category}]")
    print(f"      Query: {query}")
    print(f"      Why: {why}")
    print()
