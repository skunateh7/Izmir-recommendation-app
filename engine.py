# ─────────────────────────────────────────────────────────────────────────────
# engine.py — Four-Stage Hybrid Recommendation Engine
#
# Stage 1: Content-Based Filtering  (cosine similarity on interest vectors)
# Stage 2: AHP                      (pairwise comparison → criterion weights)
# Stage 3: TOPSIS                   (weighted ideal-solution ranking)
# Stage 4: Groq LLM                 (natural language explanation layer)
# ─────────────────────────────────────────────────────────────────────────────

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data import CRITERIA_KEYS


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — CONTENT-BASED FILTERING
# ══════════════════════════════════════════════════════════════════════════════

INTEREST_KEYS = ["beach", "history", "food", "nature", "family", "nightlife"]


def build_user_vector(selected_interests: list[str]) -> np.ndarray:
    """Convert selected interests into a numeric vector."""
    return np.array([[5 if k in selected_interests else 0 for k in INTEREST_KEYS]])


def build_interest_matrix(activities: list[dict]) -> np.ndarray:
    """Stack all activity interest-score vectors into a 2-D matrix."""
    return np.array([
        [a["interest_scores"][k] for k in INTEREST_KEYS]
        for a in activities
    ])


def stage1_content_filter(activities: list[dict],
                           selected_interests: list[str],
                           top_n: int = 10) -> list[dict]:
    """
    Stage 1 — Cosine similarity between user interest vector and each activity.
    Returns top_n candidates that proceed to Stage 2 & 3.
    """
    if not selected_interests:
        return []

    user_vec    = build_user_vector(selected_interests)
    act_matrix  = build_interest_matrix(activities)
    sim_scores  = cosine_similarity(user_vec, act_matrix)[0]

    results = []
    for i, activity in enumerate(activities):
        item = dict(activity)
        item["cbf_score"] = round(float(sim_scores[i]), 4)
        results.append(item)

    results.sort(key=lambda x: x["cbf_score"], reverse=True)
    return results[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — AHP (Analytic Hierarchy Process)
# ══════════════════════════════════════════════════════════════════════════════

# Random Index values for consistency ratio calculation (Saaty, 1980)
# Index corresponds to matrix size n = 1, 2, 3, ... 10
RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
      6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp_compute_weights(pairwise_matrix: np.ndarray) -> dict:
    """
    Compute AHP criterion weights from a pairwise comparison matrix.

    Steps:
    1. Normalise each column by its sum.
    2. Average each row → priority vector (weights).
    3. Compute lambda_max for consistency check.
    4. Calculate Consistency Index (CI) and Consistency Ratio (CR).

    Returns dict with keys: weights, lambda_max, CI, CR, consistent
    """
    n = pairwise_matrix.shape[0]

    # Step 1 — Column normalisation
    col_sums   = pairwise_matrix.sum(axis=0)
    normalised = pairwise_matrix / col_sums

    # Step 2 — Row averages = priority weights
    weights = normalised.mean(axis=1)

    # Step 3 — Lambda max
    weighted_sum = pairwise_matrix @ weights
    lambda_vec   = weighted_sum / weights
    lambda_max   = lambda_vec.mean()

    # Step 4 — Consistency
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri = RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0

    return {
        "weights":    weights,        # numpy array, length = n criteria
        "lambda_max": round(float(lambda_max), 4),
        "CI":         round(float(ci), 4),
        "CR":         round(float(cr), 4),
        "consistent": cr < 0.10,      # Saaty threshold
    }


def build_ahp_matrix_from_preferences(user_preferences: dict) -> np.ndarray:
    """
    Build pairwise comparison matrix from user slider preferences.

    user_preferences: dict mapping criterion_key → importance (1–5)
    Higher importance = more important to the user.

    The pairwise value between criteria i and j is:
        importance[i] / importance[j]
    This produces a valid positive reciprocal matrix.
    """
    n       = len(CRITERIA_KEYS)
    matrix  = np.ones((n, n))
    weights = [user_preferences.get(k, 3) for k in CRITERIA_KEYS]

    for i in range(n):
        for j in range(n):
            if i != j:
                ratio = weights[i] / weights[j]
                matrix[i][j] = round(ratio, 4)

    return matrix


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — TOPSIS (Technique for Order Preference by Similarity to Ideal)
# ══════════════════════════════════════════════════════════════════════════════

def stage3_topsis(candidates: list[dict],
                  ahp_weights: np.ndarray) -> list[dict]:
    """
    Stage 3 — TOPSIS ranking of candidate attractions.

    Steps:
    1. Build decision matrix from criteria scores.
    2. Normalise using vector normalisation.
    3. Apply AHP weights → weighted normalised matrix.
    4. Identify Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS).
    5. Compute separation from PIS and NIS.
    6. Compute TOPSIS closeness coefficient.
    7. Rank by descending closeness coefficient.

    All criteria are BENEFIT type (higher = better).
    """
    n_alt    = len(candidates)
    n_crit   = len(CRITERIA_KEYS)

    # Step 1 — Decision matrix (alternatives × criteria)
    decision_matrix = np.array([
        [a["criteria"][k] for k in CRITERIA_KEYS]
        for a in candidates
    ], dtype=float)

    # Step 2 — Vector normalisation
    col_norms  = np.sqrt((decision_matrix ** 2).sum(axis=0))
    col_norms[col_norms == 0] = 1  # Avoid division by zero
    norm_matrix = decision_matrix / col_norms

    # Step 3 — Weighted normalised matrix
    weighted_matrix = norm_matrix * ahp_weights

    # Step 4 — Ideal solutions (all criteria are BENEFIT)
    pis = weighted_matrix.max(axis=0)   # Positive Ideal Solution
    nis = weighted_matrix.min(axis=0)   # Negative Ideal Solution

    # Step 5 — Euclidean separations
    d_pos = np.sqrt(((weighted_matrix - pis) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted_matrix - nis) ** 2).sum(axis=1))

    # Step 6 — Closeness coefficient
    topsis_scores = d_neg / (d_pos + d_neg + 1e-10)  # epsilon avoids div/0

    # Step 7 — Annotate and sort
    results = []
    for i, activity in enumerate(candidates):
        item = dict(activity)
        item["topsis_score"]    = round(float(topsis_scores[i]), 4)
        item["d_positive"]      = round(float(d_pos[i]), 4)
        item["d_negative"]      = round(float(d_neg[i]), 4)
        item["criteria_weighted"] = {
            k: round(float(weighted_matrix[i][j]), 4)
            for j, k in enumerate(CRITERIA_KEYS)
        }
        results.append(item)

    results.sort(key=lambda x: x["topsis_score"], reverse=True)

    # Assign final rank
    for rank, item in enumerate(results, start=1):
        item["topsis_rank"] = rank

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — GROQ LLM EXPLANATION LAYER
# ══════════════════════════════════════════════════════════════════════════════

def build_llm_prompt(selected_interests: list[str],
                     ahp_weights_dict: dict,
                     ranked_results: list[dict]) -> str:
    """Build the prompt for the LLM explanation layer."""

    interest_labels = ", ".join(i.replace("_", " ").title() for i in selected_interests)

    weight_lines = "\n".join(
        f"  {k.replace('_', ' ').title()}: {round(v*100, 1)}%"
        for k, v in ahp_weights_dict.items()
    )

    attraction_lines = "\n".join(
        f"{i+1}. {a['name']} ({a['district']}) "
        f"[TOPSIS={a['topsis_score']}, "
        f"Beach={a['criteria']['beach_score']}, "
        f"Culture={a['criteria']['cultural_score']}, "
        f"Price={a['criteria']['price_score']}, "
        f"Festival={a['criteria']['festival_score']}, "
        f"Crowding={a['criteria']['tourist_density']}, "
        f"Weather={a['criteria']['weather_comfort']}]"
        for i, a in enumerate(ranked_results)
    )

    return f"""You are an expert travel guide for İzmir, Turkey.

A tourist selected these interests: {interest_labels}.

The system used AHP to compute criterion weights based on their preferences:
{weight_lines}

TOPSIS then ranked the following attractions by mathematical closeness to the ideal tourist experience:
{attraction_lines}

Your tasks — for each attraction:
1. Write a warm, 1–2 sentence personalised explanation of WHY this attraction 
   was recommended, referencing the tourist's specific interests and the 
   criterion scores that drove its ranking.
2. Add one practical insider tip (best time to visit, what to bring, 
   local secret, seasonal advice, etc.).
3. Add a short best visiting hours note (morning / afternoon / evening / any time).

Reply ONLY with a valid JSON array. No markdown. No extra text. Structure:
[
  {{
    "id": <number 1 to {len(ranked_results)}>,
    "explanation": "<personalised why>",
    "tip": "<practical insider tip>",
    "best_time": "<best visiting hours>"
  }}
]"""


def stage4_llm_explain(ranked_results: list[dict],
                       selected_interests: list[str],
                       ahp_weights_dict: dict,
                       groq_api_key: str) -> list[dict]:
    """
    Stage 4 — Groq LLM explanation layer.
    Adds explanation, tip, and best_time to each ranked result.
    Does NOT change the TOPSIS ranking.
    """
    from groq import Groq

    client   = Groq(api_key=groq_api_key)
    prompt   = build_llm_prompt(selected_interests, ahp_weights_dict, ranked_results)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    raw  = response.choices[0].message.content.strip()
    raw  = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    llm_map = {item["id"]: item for item in data}

    for i, activity in enumerate(ranked_results, start=1):
        meta = llm_map.get(i, {})
        activity["explanation"] = meta.get("explanation", "")
        activity["tip"]         = meta.get("tip", "")
        activity["best_time"]   = meta.get("best_time", "")

    return ranked_results


# ══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE — runs all four stages in sequence
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(activities: list[dict],
                 selected_interests: list[str],
                 user_preferences: dict = None,
                 pairwise_matrix: np.ndarray = None,
                 groq_api_key: str = "",
                 use_llm: bool = False,
                 top_n_cbf: int = 12,
                 top_n_final: int = 8) -> dict:
    """
    Run the full four-stage pipeline.

    Parameters
    ----------
    activities         : full activity list from data.py
    selected_interests : list of interest keys selected by the user
    user_preferences   : dict {criterion_key: importance_value (1–5)} OR None
    pairwise_matrix    : np.ndarray (6×6) true AHP matrix — if provided, skips
                         build_ahp_matrix_from_preferences entirely
    groq_api_key       : Groq API key (optional, needed for Stage 4)
    use_llm            : whether to run Stage 4
    top_n_cbf          : how many candidates Stage 1 passes to Stage 2/3
    top_n_final        : how many final results to return

    Returns
    -------
    dict with keys: candidates, ahp_result, ranked, stages_run
    """
    # ── Stage 1 ───────────────────────────────────────────────────────────────
    candidates = stage1_content_filter(activities, selected_interests, top_n=top_n_cbf)

    if not candidates:
        return {"candidates": [], "ahp_result": {}, "ranked": [], "stages_run": []}

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    # Accept either a true pairwise matrix (new true AHP) or legacy preferences dict
    if pairwise_matrix is not None:
        ahp_result  = ahp_compute_weights(pairwise_matrix)
    else:
        _matrix    = build_ahp_matrix_from_preferences(user_preferences or {k: 3 for k in CRITERIA_KEYS})
        ahp_result  = ahp_compute_weights(_matrix)
    ahp_weights = ahp_result["weights"]

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    ranked = stage3_topsis(candidates, ahp_weights)[:top_n_final]

    stages_run = ["Stage 1 — Content-Based Filtering",
                  "Stage 2 — AHP Weight Generation",
                  "Stage 3 — TOPSIS Ranking"]

    # ── Stage 4 (optional) ───────────────────────────────────────────────────
    # Stage 4 is additive: it annotates the ranking produced by Stages 1–3 but
    # never alters it. If the LLM call fails (rate limit, invalid key, network,
    # malformed JSON), we degrade gracefully and return the mathematically
    # derived ranking rather than failing the whole run.
    llm_status = "off"
    llm_error  = ""
    if use_llm and groq_api_key:
        ahp_weights_dict = {k: float(ahp_weights[i])
                            for i, k in enumerate(CRITERIA_KEYS)}
        try:
            ranked = stage4_llm_explain(ranked, selected_interests,
                                        ahp_weights_dict, groq_api_key)
            stages_run.append("Stage 4 — Groq LLM Explanations")
            llm_status = "ok"
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "quota" in msg or "limit" in msg:
                llm_status = "rate_limited"
            elif "auth" in msg or "401" in msg or "api key" in msg or "invalid" in msg:
                llm_status = "auth_error"
            else:
                llm_status = "error"
            llm_error = str(e)
            # Ensure the explanation fields exist so the UI renders cleanly.
            for activity in ranked:
                activity.setdefault("explanation", "")
                activity.setdefault("tip", "")
                activity.setdefault("best_time", "")

    return {
        "candidates":  candidates,
        "ahp_result":  ahp_result,
        "ranked":      ranked,
        "stages_run":  stages_run,
        "llm_status":  llm_status,
        "llm_error":   llm_error,
    }
