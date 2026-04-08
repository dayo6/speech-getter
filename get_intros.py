import os
import json
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

PROMPT = """You are a music research assistant for a west coast Chicano type beat producer.

I need suggestions for SCENES and MOMENTS from real media that would work as intro speeches over beats.
I will look up the actual clips myself — your job is to point me to the right moments so I can find the real audio.

DO NOT write out exact quotes — you will get them wrong. Instead, describe the scene and what is said in it.

SOURCE TYPES to pull from (use ALL of these, not just movies):
- **Movies**: Blood In Blood Out, American Me, Training Day, Mi Vida Loca, Scarface, Friday, Boyz n the Hood,
  Colors, Menace II Society, End of Watch, Gran Torino, Harsh Times, Boulevard Nights, Don't Be a Menace,
  A Better Life, Selena, The Tax Collector, Walk Proud
- **TV Shows**: Breaking Bad, Narcos, Queen of the South, Mayans M.C., On My Block, Sons of Anarchy,
  George Lopez Show, Snowfall, Power
- **Documentaries**: LA gang documentaries, lowrider culture docs, Chicano movement docs, cartel docs
- **Interviews**: Chicano rappers on Vlad TV, No Jumper, Bootleg Kev, Million Dollaz Worth of Game, Big Boy TV
- **Stand-up comedy**: George Lopez, Gabriel Iglesias, Cheech Marin, Paul Rodriguez
- **Song intros / skits**: spoken word intros from Cypress Hill, Kid Frost, SPM, Brownside, Mr. Capone-E
- **Speeches & news**: Cesar Chavez speeches, Chicano Moratorium, walkout speeches, news reports on LA

The vibe should be one of:
- **Dark and aggressive**: menacing, street, raw energy, cartel, prison, gang life, survival
- **Fun / playful**: lowrider culture, party energy, Chicano humor, old-school west coast swagger, cruising

Rules:
- The moment should be short (under 15 seconds of dialogue) and hit hard with no context needed.
- Use a DIFFERENT source for each suggestion.
- Give me a MIX across all source types — not just movies.
- Focus on Chicano culture, west coast life, street life, lowriders, barrio culture, hustle, loyalty.
- Only suggest scenes you are confident actually exist.
- IMPORTANT: Only suggest moments that are likely to be findable on YouTube (movie clips, interview uploads, speech recordings, stand-up specials, music videos, etc.). If a clip probably isn't on YouTube, skip it.

Respond ONLY with a valid JSON array. No markdown, no code fences, no explanation — just the JSON.
Each object in the array must have these fields:
{
  "source_title": "the EXACT official title — e.g. 'Blood In Blood Out' not 'blood in blood out movie', 'Breaking Bad S4E13 - Face Off' not just 'Breaking Bad', include year for movies (e.g. 'Scarface (1983)'), season/episode for TV, full doc title, full interview title or channel + guest + upload title, full song name + artist, full speech name",
  "type": "Movie | TV | Documentary | Interview | Stand-up | Song Intro | Speech",
  "speaker": "character name AND actor real name for fiction (e.g. 'Alonzo Harris (Denzel Washington)'), or just real name for non-fiction",
  "scene_description": "describe what happens and what is said — NOT an exact quote",
  "youtube_search": "a specific YouTube search query I can paste to find this exact clip (e.g. 'Blood In Blood Out Miklo prison speech scene')",
  "vibe": "DARK | FUN",
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
        "model": "deepseek/deepseek-chat-v3-0324",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.9,
        "max_tokens": 8000,
    },
)

if response.status_code != 200:
    print(f"Error {response.status_code}: {response.text}")
    raise SystemExit(1)

data = response.json()
content = data["choices"][0]["message"]["content"]

# Strip markdown code fences if the model wraps them anyway
content = content.strip()
if content.startswith("```"):
    content = content.split("\n", 1)[1]
if content.endswith("```"):
    content = content.rsplit("```", 1)[0]
content = content.strip()

suggestions = json.loads(content)

# Save to file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"intros_{timestamp}.json"
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
