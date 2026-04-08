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

# ── Load previous suggestions to avoid repeats ──────────────
previous = []
for f in sorted(os.listdir(".")):
    if f.startswith("intros_") and f.endswith(".json"):
        with open(f, "r", encoding="utf-8") as fh:
            try:
                previous.extend(json.load(fh))
            except json.JSONDecodeError:
                pass

already_used = ""
if previous:
    scenes = [f"{s.get('source_title', s.get('source', ''))}: {s.get('scene_description', '')}" for s in previous]
    already_used = "\n\nALREADY USED — same source is OK, but do NOT repeat any of these specific scenes/moments:\n"
    for scene in scenes:
        already_used += f"- {scene}\n"
    print(f"[*] Found {len(previous)} previous suggestions — will avoid repeating the same scenes.\n")

PROMPT = """You are a music research assistant for a dark west coast Chicano type beat producer.

I need suggestions for SCENES and MOMENTS from real media that would work as intro speeches over
dark, aggressive, street hip hop beats. Think: Chicano gangster rap, west coast G-funk, dark trap.
I will look up the actual clips myself — your job is to point me to the right moments so I can find the real audio.

DO NOT write out exact quotes — you will get them wrong. Instead, describe the scene and what is said in it.

EVERYTHING should be related to: street life, gang culture, drugs, cartels, prison, hustle, survival,
the barrio, loyalty, betrayal, power, violence, or similar themes fitting for a dark hip hop channel.

Example SOURCE TYPES to pull from:
- **Documentaries**: LA gang documentaries, cartel docs, prison docs, drug trade docs, hood docs,
  Crips & Bloods: Made in America, Sin Nombre, LA Originals, The Mexican Mafia
- **Interviews**: Street rappers and OGs on Vlad TV, No Jumper, Bootleg Kev, Million Dollaz Worth of Game,
  Big Boy TV, Soft White Underbelly, street interviews about gang life / drug game / prison
- **Song intros / skits**: spoken word intros from Cypress Hill, Kid Frost, SPM, Brownside, Mr. Capone-E,
  Conejo, Tupac, Snoop Dogg, Ice Cube, NWA, Mobb Deep, Immortal Technique
- **Meme clips**: viral street moments, hood memes, iconic threatening one-liners that became memes,
  viral police chase clips, news reporter in the hood moments
- **News & speeches**: news reports on LA gangs, cartel busts, drug raids, prison riots, street protests
- **Movies**: Blood In Blood Out, American Me, Training Day, Mi Vida Loca, Scarface, Boyz n the Hood,
  Colors, Menace II Society, End of Watch, Harsh Times, The Tax Collector, Sicario, City of God,
  New Jack City, Carlito's Way, A Prophet, Shot Caller, Felon, Animal Factory, Blow
- **TV Shows**: Breaking Bad, Narcos, Queen of the South, Mayans M.C., Sons of Anarchy,
  Snowfall, Power, Oz, The Wire, Top Boy

What makes a perfect beat intro moment:

TONE — Raw street-level authenticity, NOT cinematic polish:
- Should sound like a voice note, a studio conversation, or someone talking real shit to their people.
- Loose, conversational grammar. Verbal fillers that add rhythm ("you feel me", "straight up", "on God") are GOOD.
- Unscripted energy — plain-spoken but heavy truths. Nothing that feels like movie-script dialogue.

TWO STYLES TO LOOK FOR:
1. "Daily Reminder" (Street Wisdom) — direct real-talk advice about the game, loyalty, haters, survival.
   Short punchy statements followed by a confirmation ("keep winning", "you feel me?", "that's on everything").
2. "Street Anecdote" (Storytelling) — first-person story or philosophy of violence/power/hustle.
   Raw, blunt, ends on a high-energy mic-drop moment.

STRUCTURE:
- 2-4 sentences max. Short, punchy, staccato syntax — no complex conjunctions.
- Prefer words with heavy hard consonants (D, K, T, P), especially as last words — they hit harder on a beat.
- Start in the middle of a thought — no introductory fluff, no "I think", no "let me tell you".

THEME:
- Ambition, inevitability, betrayal, the cost of power, silence over noise, the hunt.
- NO jokes, puns, cliches about "working hard", or upbeat energy.

The vibe should be one of:
- **DARK**: menacing, threatening, cold, raw street energy, cartel power, prison hardness, survival mode
- **MEME**: viral hood moments, street humor with an edge, iconic clips that are funny but still grimy

Rules:
- The moment should be short (14-21 seconds of dialogue) and hit hard with no context needed.
- Use a DIFFERENT source for each suggestion.
- Give me a MIX across all source types — not just movies.
- Keep it dark and street. No soft, wholesome, or family-friendly content.
- Only suggest scenes you are confident actually exist.
- IMPORTANT: Only suggest moments that are likely to be findable on YouTube. If a clip probably isn't on YouTube, skip it.

Respond ONLY with a valid JSON array. No markdown, no code fences, no explanation — just the JSON.
Each object in the array must have these fields:
{
  "source_title": "the EXACT official title — e.g. 'Blood In Blood Out' not 'blood in blood out movie', 'Breaking Bad S4E13 - Face Off' not just 'Breaking Bad', include year for movies (e.g. 'Scarface (1983)'), season/episode for TV, full doc title, full interview title or channel + guest + upload title, full song name + artist, full speech name",
  "type": "Movie | TV | Documentary | Interview | Stand-up | Song Intro | Speech | Meme",
  "speaker": "character name AND actor real name for fiction (e.g. 'Alonzo Harris (Denzel Washington)'), or just real name for non-fiction",
  "scene_description": "describe what happens and what is said — NOT an exact quote",
  "youtube_search": "a specific YouTube search query I can paste to find this exact clip (e.g. 'Blood In Blood Out Miklo prison speech scene')",
  "vibe": "DARK | MEME",
  "why_it_works": "one sentence on why it slaps as a beat intro"
}

Give me 20 suggestions.""" + already_used

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "qwen/qwq-32b",
        "messages": [
            {"role": "system", "content": "You are a JSON API. Respond ONLY in valid English JSON. Never use non-English characters. Never use Chinese, Japanese, Korean, or any non-ASCII characters in your output."},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0.6,
        "max_tokens": 16000,
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

# Strip R1 reasoning block (everything inside <think>...</think>)
if "<think>" in content:
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

# Strip markdown code fences if the model wraps them anyway
content = content.strip()
if content.startswith("```"):
    # Remove first line (```json or ```)
    content = content.split("\n", 1)[1]
if content.endswith("```"):
    content = content.rsplit("```", 1)[0]
content = content.strip()

# Strip non-ASCII characters (DeepSeek sometimes injects Chinese chars)
content = re.sub(r'[^\x00-\x7F]+', '', content)

# Try to extract JSON array if model added extra text
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
except json.JSONDecodeError:
    # Save raw response for debugging
    raw_path = f"debug_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Failed to parse JSON. Raw response saved to {raw_path}")
    raise SystemExit(1)

# Parse output flag
parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default=None, help="Output file path")
args = parser.parse_args()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = args.output or f"intros_{timestamp}.json"
os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(suggestions, f, indent=2, ensure_ascii=False)

# Print summary
print(f"\nSaved {len(suggestions)} suggestions to {output_file}\n")
for i, s in enumerate(suggestions, 1):
    vibe = s.get("vibe", "?")
    source = s.get("source_title", "?")
    stype = s.get("type", "?")
    speaker = s.get("speaker", "?")
    yt = s.get("youtube_search", "?")
    print(f"  {i:2}. [{vibe}] {source} ({stype})")
    print(f"      Speaker: {speaker}")
    print(f"      YouTube: {yt}")
    print(f"      {s.get('scene_description', '')}")
    print()
