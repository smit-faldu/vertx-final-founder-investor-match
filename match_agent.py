import os
import json
import re
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# ────────────────────────────────────────────────────────────────
print("🔹 Loading models and investor data...")
load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and investor metadata
index = faiss.read_index("investors.index")
with open("investors.pkl", "rb") as f:
    investors = pickle.load(f)

# ────────────────────────────────────────────────────────────────
# Gemini LLM initialization
print("🔹 Initializing Gemini LLM...")
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.0,
    google_api_key='GEMINI_API_KEY'
)

# ────────────────────────────────────────────────────────────────
# Format founder profile into a string
def format_founder(form_data):
    return f"Founder: {form_data['founder_name']}, Company: {form_data['company_name']}, Building: {form_data['what_building']}, Industry: {form_data['industry']}, Sectors: {form_data['sectors']}, Stage: {form_data['product_stage']}, Countries: {form_data['target_countries']}, Required Funding: {form_data['required_funding']}"

# Rule-based score function for matching
def rule_score(inv, form_data):
    score = 0
    def norm(x): return x.lower() if isinstance(x, str) else ""

    if norm(form_data["industry"]) in norm(inv.get("Industry", "")):
        score += 10
    if norm(form_data["product_stage"]) in norm(inv.get("Stage", "")):
        score += 15

    founder_countries = [c.strip().lower() for c in form_data.get("target_countries", "").split(",")]
    investor_countries = [c.strip().lower() for c in inv.get("Countries", "").split(",")]
    if any(fc in investor_countries for fc in founder_countries):
        score += 10

    try:
        amt = int(form_data["required_funding"].replace("$", "").replace("K", "000"))
        cheque_range = inv.get("Cheque_range", "")
        if "-" in cheque_range:
            min_c, max_c = cheque_range.replace("$", "").replace("K", "000").split("-")
            if int(min_c) <= amt <= int(max_c):
                score += 15
    except:
        pass

    return score

# Retrieve ranked investors using vector similarity + rule score
def get_ranked_investors(profile_text, form_data, top_k=40):
    founder_vector = embedding_model.encode([profile_text])[0].astype("float32")
    D, I = index.search(founder_vector.reshape(1, -1), top_k)

    investors_scored = []
    for idx, i in enumerate(I[0]):
        inv = investors[i]
        sim_score = float(D[0][idx])
        rules = rule_score(inv, form_data)
        combined = round(min(sim_score * 100 + rules, 100), 2)
        investors_scored.append({
            "name": inv.get("Name"),
            "industry": inv.get("Industry"),
            "stage": inv.get("Stage"),
            "countries": inv.get("Countries"),
            "cheque_range": inv.get("Cheque_range"),
            "overview": inv.get("Overview"),
            "similarity_score": sim_score,
            "rule_based_score": rules,
            "matching_score": combined
        })

    return sorted(investors_scored, key=lambda x: x["matching_score"], reverse=True)

# Main function to be called by Flask app
def run_agent_with_input(form_data):
    profile_text = format_founder(form_data)
    top_matches = get_ranked_investors(profile_text, form_data)

    prompt = f"""
You are an AI assistant helping match founders with investors.
Below is the founder's profile and a list of investor matches with pre-computed scores.
Your job is ONLY to generate a "reason" for why each investor is a good match.

Return a JSON array like this:
[
  {{
    "name": "Investor Name",
    "matching_score": 92.5,
    "reason": "Why this investor is a great fit"
  }},
  ...
]

Do NOT change the score or investor order.
Do NOT add any explanation or Final Answer outside the array.

Founder:
{profile_text}

Investors:
{json.dumps(top_matches, indent=2)}
"""

    result = gemini.invoke(prompt)
    try:
        clean = re.sub(r"```json|```", "", result.content).strip()
        json_start = clean.find("[")
        json_array = clean[json_start:]
        matches = json.loads(json_array)
        return matches
    except Exception as e:
        print("❌ Failed to parse Gemini response")
        print(result.content)
        print("Error:", e)
        return []
