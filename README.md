# İzmir Tourist Recommendation System 🌊
### A Hybrid AI-Powered Activity Recommender


---

## 📁 Project Structure

```
izmir_recommender/
│
├── app.py            ← Streamlit web interface (run this)
├── engine.py         ← Four-stage recommendation engine (CBF → AHP → TOPSIS → LLM)
├── data.py           ← Dataset of 35 real İzmir activities
├── requirements.txt  ← Python dependencies
├── .streamlit/
│   └── secrets.toml.example  ← Template for server-side API key (copy, don't commit)
└── README.md         ← This file
```
---

## ⚙️ How to Install & Run

### 1. Make sure Python 3.10+ is installed
```bash
python --version
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open automatically at: **http://localhost:8501**

---

## 🔑 API Key Configuration

Stage 4 (AI explanations) uses the Groq API. The key is **never hard-coded**. There are
two supported ways to supply it, and the app picks whichever is available:

### Option A — Server-side key (recommended for deployment / data collection)

Participants never see or type a key. They just open your link and use the app.

**Local:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit the file and paste your real key
```
Or use an environment variable:
```bash
export GROQ_API_KEY="gsk_your_key_here"   # macOS / Linux
setx GROQ_API_KEY "gsk_your_key_here"     # Windows
```

**Streamlit Community Cloud:**
1. Push this repo to GitHub (`secrets.toml` is git-ignored and will not be uploaded).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo,
   branch, and `app.py`.
3. Open **Advanced settings → Secrets** (or, after deploying, **⋮ → Settings → Secrets**).
4. Paste one line:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Save. The app restarts automatically and Stage 4 works for every visitor.

### Option B — User-supplied key (automatic fallback)

If no server key is configured, the app asks each user for their own key at Step 3.
It is held only in that browser session and is never written to disk or logged.
Free keys: [console.groq.com/keys](https://console.groq.com/keys)

> ⚠️ **Do not paste your key into a chat, email, or the repo itself.** Sharing a personal
> API key with participants violates Groq's terms and makes your account responsible for
> all usage. Use Option A instead — it gives the same one-click convenience safely.

### If the AI layer is unavailable

Stage 4 is **additive**: it annotates the ranking produced by Stages 1–3 but never changes it.
If the Groq call fails (rate limit, invalid key, network issue), the app shows a brief notice
and still displays the complete, mathematically derived recommendations. Nothing is lost.

---

## 🧠 How the Hybrid Engine Works

### Stage 1 — Content-Based Filtering (`engine.py`)
Each activity has a vector like `{history:5, beach:0, food:2, nightlife:0}`.
Your selected interests become a user vector (e.g. `{history:5, food:5}`).
Cosine similarity measures how aligned each activity is with your interests.

```
              A · B
sim(A, B) = ─────────
             |A| × |B|
```

Implemented with `sklearn.metrics.pairwise.cosine_similarity`.
The top 14 candidates pass to Stage 2.

### Stage 2 — AHP Weight Derivation (`engine.py`)
The user completes 15 pairwise comparisons across 6 criteria using the Saaty 1–9 scale.
The principal eigenvector of the resulting 6×6 matrix gives the criterion weights, and the
**Consistency Ratio (CR)** is computed. Results are only produced when CR ≤ 0.10, so the
weights used are always internally coherent.

### Stage 3 — TOPSIS Ranking (`engine.py`)
Candidates are scored against the AHP weights using the Technique for Order Preference by
Similarity to Ideal Solution. Each activity receives a closeness coefficient (0–1)
measuring its distance from the ideal and anti-ideal solutions. **This produces the final
ranking.**

### Stage 4 — Grounded LLM Explanation Layer (`engine.py`)
The ranked results, the derived AHP weights, and the TOPSIS scores are passed to
Groq (LLaMA 3.3 70B) as structured inputs. The model writes, for each activity:
- A personalised explanation of why it fits the user's stated priorities
- A practical insider tip
- A suggested best visiting time

> **The LLM does not re-rank.** It receives the ranking as a fixed input and may only
> translate the mathematical reasoning into natural language. This separation of
> *recommendation generation* from *explanation generation* is the core architectural
> feature of the system, and it is what makes the explanations traceable to the
> underlying AHP/TOPSIS computation rather than independently generated.

---

## 📊 Dataset

- **35 activities** across 4 categories: History, Beach, Food, Nightlife
- Sourced from: Lonely Planet, TripAdvisor, Culinary Backstreets,
  Visit İzmir (Official), Goats on the Road, Thrillophilia, and more
- Each activity includes: name, category, district, description, tags,
  best season, price range, interest scores, and source

---

## 🎓 Research Paper Notes

This system implements four identifiable algorithmic contributions:

| Contribution | Method | Library |
|---|---|---|
| Content-based filtering | Cosine similarity on interest vectors | scikit-learn |
| AHP weight derivation | Saaty 1–9 pairwise matrix, principal eigenvector, CR validation | NumPy |
| TOPSIS ranking | Closeness to ideal / anti-ideal solution | NumPy |
| Grounded explanation layer | Post-hoc explanation constrained to AHP weights + TOPSIS scores (no re-ranking) | Groq API |

**Recommended evaluation metrics:**
- Precision@K (top-K relevance)
- NDCG (Normalized Discounted Cumulative Gain)
- User satisfaction survey (Likert scale 1–5)

**Suggested publication venues:**
- Information Technology & Tourism (Springer)
- Journal of Destination Marketing & Management
- ENTER e-Tourism Conference
- IEEE Intelligent Systems Conference
