from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from match_agent import run_agent_with_input

app = Flask(__name__)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS investor data (optional)
if os.path.exists("investors.index") and os.path.exists("investors.pkl"):
    index = faiss.read_index("investors.index")
    with open("investors.pkl", "rb") as f:
        investors = pickle.load(f)
else:
    index = None
    investors = []

@app.route("/", methods=["GET", "POST"])
def founder_form():
    if request.method == "POST":
        form_data = request.form.to_dict()

        # Run agent to get JSON matches
        matches = run_agent_with_input(form_data)
        print(matches)
        return render_template("match_results.html", name=form_data.get("founder_name", "Founder"), matches=matches)

    return render_template("form.html")

if __name__ == "__main__":
    app.run(debug=True)

