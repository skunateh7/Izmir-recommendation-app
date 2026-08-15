# ─────────────────────────────────────────────────────────────────────────────
# app.py — AI-Powered Hybrid Tourism Recommendation System for İzmir
# Stages: Content-Based Filtering → AHP → TOPSIS → Groq LLM
# Run with:  streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import base64, os
from data import ACTIVITIES, CATEGORY_META, CRITERIA_META, CRITERIA_KEYS
from engine import run_pipeline, build_ahp_matrix_from_preferences, ahp_compute_weights

# ── LOAD BANNER IMAGE ─────────────────────────────────────────────────────────
def get_banner_b64():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "izmir_banner.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

BANNER_B64 = get_banner_b64()

# ── GROQ API KEY ──────────────────────────────────────────────────────────────
# Two ways the key can be supplied, in priority order:
#
#   1. SERVER-SIDE (recommended for data collection / deployment)
#      Set GROQ_API_KEY in Streamlit Cloud → App settings → Secrets, or as an
#      environment variable when running locally. Participants then never see
#      or type a key — they just open the link and use the app.
#
#   2. USER-SUPPLIED (fallback, e.g. someone running this from the repo)
#      If no server key is configured, the app asks the user for their own key
#      at Step 3. It is held only in that browser session (st.session_state)
#      and is never written to disk, logged, or persisted.
#
# The key is never hard-coded in this file.
def _load_server_key() -> str:
    """Return a server-configured Groq key, or '' if none is set."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        # No secrets.toml present (normal for local runs) — fall through.
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()

SERVER_GROQ_KEY = _load_server_key()
HAS_SERVER_KEY  = bool(SERVER_GROQ_KEY)

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="İzmir Tourism Recommender", page_icon="🌊", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');

html, body, [class*="css"], .stApp { background:#ffffff !important; font-family:'Source Sans 3',sans-serif; color:#111; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:0 !important; max-width:800px; }

.hero {
    background:#111111;
    border-radius:0 0 28px 28px;
    padding:48px 40px 38px;
    text-align:center;
    margin-top: -60px;
    margin-bottom: 36px;
    box-shadow:0 8px 40px rgba(0,0,0,0.25);
    /* Full-bleed trick that works at any screen width, unlike a fixed px margin:
       pull the element to the viewport edges using 50vw, then push back by half
       of the (centered) content column's own width. */
    width: 100vw;
    margin-left: calc(50% - 50vw);
    margin-right: calc(50% - 50vw);
}
@media (max-width: 600px) {
    .hero { padding:36px 20px 28px; border-radius:0 0 20px 20px; }
    .hero h1 { font-size:1.9rem; }
    .footer { padding:20px 12px 14px; font-size:10.5px; }
}
.hero .eyebrow { font-size:10px; letter-spacing:.3em; text-transform:uppercase; color:rgba(255,255,255,.45); margin-bottom:12px; }
.hero h1 { font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:700; color:#fff !important; margin:0 0 10px; line-height:1.15; }
.hero .tagline { font-size:14px; color:rgba(255,255,255,.6); line-height:1.65; max-width:500px; margin:0 auto 16px; }
.hero .stages { display:flex; justify-content:center; gap:8px; flex-wrap:wrap; }
.hero .stage-pill {
    background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.2);
    border-radius:20px; padding:4px 12px; font-size:10px; color:rgba(255,255,255,.55);
    letter-spacing:.06em;
}

.section-label { font-family:'Playfair Display',serif; font-size:1.25rem; font-weight:700; color:#111; margin:0 0 4px; }
.section-sub { font-size:12px; color:#777; margin-bottom:16px; }
.divider { border:none; border-top:1.5px solid #eaecf4; margin:26px 0; }

div.stButton > button {
    box-sizing: border-box !important;
    width:100%; background:#1a4d2e !important; color:#ffffff !important;
    border:2px solid #1a4d2e !important; border-radius:12px !important;
    padding:0 16px !important; font-size:13px !important; font-weight:600 !important;
    font-family:'Source Sans 3',sans-serif !important;
    text-align:center !important; justify-content:center !important;
    display:flex !important; align-items:center !important;
    height:52px !important; min-height:52px !important;
    transition:all .2s ease !important; margin-bottom:3px;
    box-shadow:0 2px 8px rgba(26,77,46,0.18) !important;
    line-height:1.2 !important;
}
div.stButton > button > div,
div.stButton > button p {
    margin:0 !important; padding:0 !important;
    width:100% !important; text-align:center !important;
}
div.stButton > button:hover {
    background:#2d6a4f !important; border-color:#2d6a4f !important;
    box-shadow:0 4px 14px rgba(26,77,46,0.28) !important;
    transform:translateY(-1px);
}
div.stButton > button[kind="primary"] {
    background:#1a4d2e !important; color:#fff !important; border:none !important;
    font-size:15px !important; font-weight:700 !important; text-align:center !important;
    box-shadow:0 4px 18px rgba(26,77,46,.35) !important; padding:15px 20px !important;
    border-radius:12px !important;
    width: 770px !important;
}
div.stButton > button[kind="primary"]:hover { background:#2d6a4f !important; box-shadow:0 6px 22px rgba(26,77,46,.40) !important; }
div.stButton > button[kind="primary"]:disabled { background:#aaa !important; box-shadow:none !important; border-color:#aaa !important; }

.chips-row { display:flex; flex-wrap:wrap; gap:7px; margin:12px 0 4px; }
.chip { display:inline-flex; align-items:center; gap:5px; background:#1a4d2e; border-radius:20px; padding:4px 13px; font-size:12px; font-weight:600; color:#fff; }

/* Sliders */
div[data-testid="stSlider"] label { font-size:12px !important; color:#444 !important; font-weight:600 !important; }

/* Expander */
div[data-testid="stExpander"] { border:1.5px solid #e2e4eb !important; border-radius:14px !important; box-shadow:0 2px 8px rgba(0,0,0,.05) !important; margin-bottom:10px !important; background:#fff !important; overflow:hidden; }
div[data-testid="stExpander"] summary { font-weight:700 !important; color:#111 !important; padding:14px 18px !important; background:#f8f9fc !important; font-size:14px !important; }
div[data-testid="stExpander"] summary:hover { background:#edf5f0 !important; }
div[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] { padding:4px 18px 16px !important; }

.stage-badge { display:inline-block; background:#e6f4ea; color:#1a4d2e; border:1px solid #a8d5b0; font-size:10px; font-weight:700; padding:3px 10px; border-radius:20px; letter-spacing:.06em; margin-right:6px; }
.algo-box { background:#f0f7f2; border:1px solid #b8d9c4; border-left:4px solid #1a4d2e; border-radius:0 10px 10px 0; padding:13px 16px; font-size:12px; color:#333; line-height:1.65; margin-bottom:20px; }
.algo-box strong { color:#1a4d2e; }
.algo-box code { background:#dff0e8; color:#0f3d22; padding:2px 7px; border-radius:4px; font-size:11px; font-family:'Courier New',monospace; border:1px solid #a8d5b0; }

.result-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; flex-wrap:wrap; gap:8px; }
.result-count { background:#e6f4ea; color:#1a4d2e; border:1px solid #a8d5b0; font-size:11px; font-weight:700; padding:4px 13px; border-radius:20px; }

.score-row { display:flex; gap:20px; flex-wrap:wrap; padding:10px 14px; background:#f8f9fc; border-radius:10px; margin-bottom:12px; border:1px solid #eaecf4; }
.score-item { display:flex; flex-direction:column; align-items:center; gap:2px; min-width:52px; }
.score-emoji { font-size:16px; }
.score-dots { font-size:8px; letter-spacing:2px; color:#1a4d2e; }
.score-label { font-size:9px; color:#aaa; text-transform:uppercase; letter-spacing:.07em; font-weight:600; }

.topsis-bar-wrap { margin:10px 0; }
.topsis-bar-label { font-size:11px; font-weight:600; color:#1a4d2e; margin-bottom:4px; }
.topsis-bar-bg { background:#e8f0eb; border-radius:20px; height:10px; width:100%; }
.topsis-bar-fill { background:linear-gradient(90deg,#1a4d2e,#52b788); border-radius:20px; height:10px; }

.ai-box { background:#f0f7f2; border:1px solid #b8d9c4; border-left:4px solid #1a4d2e; border-radius:0 10px 10px 0; padding:9px 13px; font-size:12px; color:#1a3d24; font-style:italic; line-height:1.55; margin-bottom:10px; }
.ai-label { font-style:normal; font-weight:700; font-size:9px; text-transform:uppercase; letter-spacing:.1em; color:#1a4d2e; display:block; margin-bottom:3px; }
.tip-box { background:#fffbf0; border:1px solid #f0e0a0; border-radius:10px; padding:9px 13px; font-size:12px; color:#5a4a00; line-height:1.55; margin-bottom:10px; }
.time-box { display:inline-block; background:#f0f3fb; border:1px solid #c5cde8; border-radius:20px; padding:3px 12px; font-size:11px; color:#0a1f5c; font-weight:600; margin-bottom:10px; }

.card-desc { font-size:13px; color:#333; line-height:1.65; margin-bottom:12px; }
.meta-row { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:10px; align-items:center; }
.meta-badge { background:#f4f5f8; border:1px solid #d0d4e0; border-radius:20px; padding:3px 11px; font-size:11px; color:#111; font-weight:600; }
.topsis-badge { margin-left:auto; background:#1a4d2e; color:#fff; border-radius:20px; padding:3px 11px; font-size:10px; font-weight:700; font-family:'Courier New',monospace; }
.tags-row { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:10px; }
.tag { background:#f4f5f8; color:#555; font-size:10px; padding:2px 9px; border-radius:20px; border:1px solid #e2e4eb; }
.source-line { font-size:10px; color:#aaa; border-top:1px solid #f0f0f0; padding-top:8px; }

.ahp-box { background:#f8f9fc; border:1.5px solid #d0d4e0; border-radius:12px; padding:16px 18px; margin-bottom:8px; }
.ahp-title { font-family:'Playfair Display',serif; font-size:1rem; font-weight:700; color:#111; margin-bottom:3px; }
.ahp-sub { font-size:12px; color:#777; margin-bottom:14px; }
.cr-ok { display:inline-block; background:#e6f4ea; border:1px solid #a8d5b0; color:#1e6e35; font-size:11px; font-weight:700; padding:3px 10px; border-radius:20px; }
.cr-warn { display:inline-block; background:#fff3cd; border:1px solid #f0c040; color:#5a4000; font-size:11px; font-weight:700; padding:3px 10px; border-radius:20px; }

.empty-state { text-align:center; padding:40px 20px; color:#aaa; font-size:14px; line-height:1.7; }
.footer { text-align:center; padding:24px 0 14px; font-size:11px; color:#bbb; border-top:1.5px solid #eaecf4; margin-top:36px; line-height:1.9; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [("selected", []), ("results", {}), ("ahp_pairs", {})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── HERO ──────────────────────────────────────────────────────────────────────
if BANNER_B64:
    _bg_style = (
        f"background-image:url('data:image/png;base64,{BANNER_B64}');"
        "background-size:cover;background-position:center 40%;position:relative;"
    )
    _overlay = (
        "<div style='position:absolute;inset:0;"
        "background:linear-gradient(160deg,rgba(0,0,0,0.65) 0%,rgba(0,0,0,0.48) 60%,rgba(0,0,0,0.60) 100%);"
        "border-radius:0 0 28px 28px;'></div>"
    )
else:
    _bg_style = "background:#111111;"
    _overlay  = ""

st.markdown(f"""
<div class="hero" style="{_bg_style}">
{_overlay}
<div style="position:relative;z-index:1;">
    <div class="eyebrow">AI-Powered Hybrid Tourism Recommendation System · İzmir, Turkey</div>
    <h1>Discover İzmir</h1>
    <div class="tagline">
        Select your interests, set your priorities, and let our four-stage hybrid engine
        find and explain the best activities in İzmir just for you.
    </div>
    <div class="stages">
        <span class="stage-pill">① Content-Based Filtering</span>
        <span class="stage-pill">② AHP Weight Generation</span>
        <span class="stage-pill">③ TOPSIS Ranking</span>
        <span class="stage-pill">④ Groq LLM Explanation</span>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# ── STEP 1: INTERESTS ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Step 1 — Select Your Interests</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Choose one or more. The system will filter relevant attractions first.</div>', unsafe_allow_html=True)

DESCS = {
    "beach":     "Swimming, water sports & sunsets",
    "history":   "Ruins, museums & Ottoman heritage",
    "food":      "Local cuisine, tours & dining",
    "nature":    "Parks, hiking & countryside",
    "family":    "Kid-friendly & group activities",
    "nightlife": "Bars, live music & beach clubs",
}

col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3, col1, col2, col3]
for col, (key, meta) in zip(cols, CATEGORY_META.items()):
    with col:
        is_sel = key in st.session_state.selected
        tick   = "✓  " if is_sel else ""
        if st.button(f"{tick}{meta['emoji']} {meta['label']}", key=f"int_{key}"):
            if key in st.session_state.selected:
                st.session_state.selected.remove(key)
            else:
                st.session_state.selected.append(key)
            st.session_state.results = {}
            st.rerun()

if st.session_state.selected:
    chips = " ".join(
        f'<span class="chip">{CATEGORY_META[i]["emoji"]} {CATEGORY_META[i]["label"]}</span>'
        for i in st.session_state.selected
    )
    st.markdown(f'<div class="chips-row">{chips}</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── STEP 2: TRUE AHP PAIRWISE COMPARISON ────────────────────────────────────
st.markdown('<div class="section-label">Step 2 — Compare What Matters Most to You</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">You\'ll see two things at a time. Just pick which one matters '
    'more for your trip — there are no right or wrong answers.</div>',
    unsafe_allow_html=True,
)

with st.expander("❓  New to this? Read this first (30 seconds)", expanded=False):
    st.markdown("""
<div style="font-size:13px;line-height:1.75;color:#25352b;">

<strong>What you're about to do</strong><br>
Instead of asking you to rate everything from 1 to 10, we show you two things at a time
and ask a simpler question: <em>which of these two matters more to me?</em><br><br>

People are much better at comparing two things than at scoring things in isolation.
It's like packing a suitcase: deciding "do I need the jacket more than the extra shoes?"
is easier than giving the jacket a score out of 10.<br><br>

<strong>How to answer</strong><br>
For each row, use the dropdown in the middle:
<ul style="margin:6px 0 6px 18px;padding:0;">
  <li>Pick <strong>Equal importance</strong> if you genuinely don't mind either way.</li>
  <li>Pick <strong>slightly</strong> more important for a mild preference.</li>
  <li>Pick <strong>strongly</strong> or <strong>very strongly</strong> for a clear preference.</li>
  <li>Save <strong>extremely</strong> for the rare case where one thing barely matters to you at all.</li>
</ul>
Answer quickly and go with your gut — your first instinct is usually your real preference.<br><br>

<strong>Why there are 15 questions</strong><br>
There are 6 things to compare, and every one gets compared against every other one.
That works out to 15 pairs. It looks like a lot, but each takes just a couple of seconds.<br><br>

<strong>What happens next</strong><br>
Your answers get turned into a set of <em>priority weights</em> — a percentage showing how
much each factor counts in your final ranking. You'll see those weights update live below,
so you can watch your own priorities take shape.<br><br>

<strong>About the consistency check</strong><br>
The system checks whether your answers contradict each other. For example: if you say
beaches matter more than culture, and culture matters more than price, then beaches
should also matter more than price. If the answers don't line up, you'll get a friendly
warning and a chance to revise — this is normal and happens to almost everyone at least once.
It isn't a test, and it doesn't mean you did anything wrong.

</div>
    """, unsafe_allow_html=True)

CRIT_ICONS = {
    "beach_score":     "🏖️",
    "cultural_score":  "🏛️",
    "price_score":     "💰",
    "festival_score":  "🎪",
    "tourist_density": "👥",
    "weather_comfort": "☀️",
}

# Saaty scale options shown to the user
SAATY_OPTIONS = [
    "Extremely more important (9)",
    "Very strongly more important (7)",
    "Strongly more important (5)",
    "Slightly more important (3)",
    "Equal importance (1)",
    "Slightly more important (3)",
    "Strongly more important (5)",
    "Very strongly more important (7)",
    "Extremely more important (9)",
]
# Maps option index → (value, direction)
# direction: "left" means left criterion is more important
# direction: "right" means right criterion is more important
SAATY_VALUES = [9, 7, 5, 3, 1, 3, 5, 7, 9]
SAATY_DIRECTION = ["left","left","left","left","equal","right","right","right","right"]

# Generate all 15 unique pairs for 6 criteria
PAIRS = []
for i in range(len(CRITERIA_KEYS)):
    for j in range(i+1, len(CRITERIA_KEYS)):
        PAIRS.append((CRITERIA_KEYS[i], CRITERIA_KEYS[j]))

# Initialise pairwise selections in session state
if "ahp_pairs" not in st.session_state:
    # Default: equal importance (index 4) for all pairs
    st.session_state.ahp_pairs = {f"{a}_vs_{b}": 4 for a, b in PAIRS}

# Render pairwise comparisons in groups of 5
with st.expander("⚖️  Open the 15 comparisons", expanded=True):
    st.markdown("""
    <div style="background:#f0f7f2;border:1px solid #b8d9c4;border-radius:10px;
                padding:12px 16px;margin-bottom:14px;font-size:12.5px;color:#1a3d24;line-height:1.7;">
        <strong>Reading each row:</strong> the item on the <em>left</em> and the item on the
        <em>right</em> are the two being compared. The dropdown in the middle names whichever
        one you're choosing, so you can read it as a sentence.<br>
        <span style="color:#3d6b4d;">Example — for
        <strong>🏖️ Beach Quality</strong> vs <strong>💰 Affordability</strong>, choosing
        &ldquo;Affordability strongly more important&rdquo; means staying on budget matters
        clearly more to you than being near a great beach.</span><br>
        Not fussed either way? Leave it on <strong>Equal importance</strong> — that's a valid answer.
    </div>
    """, unsafe_allow_html=True)

    for a, b in PAIRS:
        a_label = f"{CRIT_ICONS[a]} {CRITERIA_META[a]['label']}"
        b_label = f"{CRIT_ICONS[b]} {CRITERIA_META[b]['label']}"
        pair_key = f"{a}_vs_{b}"

        col_l, col_m, col_r = st.columns([2, 3, 2])
        with col_l:
            st.markdown(
                f'<div style="text-align:right;font-size:13px;font-weight:600;'
                f'color:#1a4d2e;padding-top:6px;">{a_label}</div>',
                unsafe_allow_html=True,
            )
        with col_m:
            # Build display options — left side uses left criterion name, right side uses right
            display_options = [
                f"{CRITERIA_META[a]['label']} extremely more important",
                f"{CRITERIA_META[a]['label']} very strongly more important",
                f"{CRITERIA_META[a]['label']} strongly more important",
                f"{CRITERIA_META[a]['label']} slightly more important",
                "Equal importance",
                f"{CRITERIA_META[b]['label']} slightly more important",
                f"{CRITERIA_META[b]['label']} strongly more important",
                f"{CRITERIA_META[b]['label']} very strongly more important",
                f"{CRITERIA_META[b]['label']} extremely more important",
            ]
            chosen_idx = st.selectbox(
                label=f"{a_label} vs {b_label}",
                options=range(len(display_options)),
                format_func=lambda x: display_options[x],
                index=st.session_state.ahp_pairs.get(pair_key, 4),
                key=f"pair_{pair_key}",
                label_visibility="collapsed",
            )
            st.session_state.ahp_pairs[pair_key] = chosen_idx
        with col_r:
            st.markdown(
                f'<div style="text-align:left;font-size:13px;font-weight:600;'
                f'color:#1a4d2e;padding-top:6px;">{b_label}</div>',
                unsafe_allow_html=True,
            )

# Build the 6×6 pairwise matrix from selections
n = len(CRITERIA_KEYS)
pairwise_mat = np.ones((n, n))
for i, a in enumerate(CRITERIA_KEYS):
    for j, b in enumerate(CRITERIA_KEYS):
        if i == j:
            pairwise_mat[i][j] = 1.0
        elif i < j:
            pair_key = f"{a}_vs_{b}"
            idx = st.session_state.ahp_pairs.get(pair_key, 4)
            val = float(SAATY_VALUES[idx])
            direction = SAATY_DIRECTION[idx]
            if direction == "left":
                pairwise_mat[i][j] = val
                pairwise_mat[j][i] = round(1.0 / val, 4)
            elif direction == "right":
                pairwise_mat[j][i] = val
                pairwise_mat[i][j] = round(1.0 / val, 4)
            else:  # equal
                pairwise_mat[i][j] = 1.0
                pairwise_mat[j][i] = 1.0

# Compute AHP weights from the pairwise matrix
ahp_preview = ahp_compute_weights(pairwise_mat)
cr_val      = ahp_preview["CR"]
lmax_val    = ahp_preview["lambda_max"]
weights_arr = ahp_preview["weights"]

# CR status display
if ahp_preview["consistent"]:
    cr_html = f'<span class="cr-ok">✓ Your answers are consistent &nbsp;·&nbsp; CR = {cr_val:.3f}</span>'
else:
    cr_html = f'<span class="cr-warn">⚠ Some answers conflict &nbsp;·&nbsp; CR = {cr_val:.3f}</span>'

# Weights pills
weight_pills = " ".join(
    f'<span style="font-size:11px;background:#1a4d2e;color:#fff;padding:3px 10px;'
    f'border-radius:20px;margin:2px;display:inline-block;">'
    f'{CRIT_ICONS[k]} {CRITERIA_META[k]["label"]}: '
    f'{round(float(weights_arr[i])*100,1)}%</span>'
    for i, k in enumerate(CRITERIA_KEYS)
)

st.markdown(f"""
<div style="background:#f0f7f2;border:1px solid #b8d9c4;border-radius:12px;
            padding:14px 18px;margin-top:14px;">
    <div style="display:flex;align-items:center;justify-content:space-between;
                flex-wrap:wrap;gap:8px;margin-bottom:10px;">
        <span style="font-size:12px;font-weight:700;color:#1a4d2e;">
            Your priority weights
            <span style="font-weight:400;color:#4a7357;">— how much each factor counts (λ_max = {lmax_val})</span>
        </span>
        {cr_html}
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;">{weight_pills}</div>
</div>
""", unsafe_allow_html=True)

if not ahp_preview["consistent"]:
    st.markdown("""
    <div style="background:#fff3cd;border:1px solid #f0c040;border-left:4px solid #e0a000;
                border-radius:0 10px 10px 0;padding:12px 16px;font-size:12.5px;color:#5a4000;
                margin-top:8px;line-height:1.7;">
        ⚠️ <strong>A few of your answers don't quite line up.</strong> This is common and easy to fix —
        it doesn't mean you did anything wrong.<br><br>
        <strong>What it means:</strong> if you said A matters more than B, and B matters more than C,
        then A should also matter more than C. Somewhere in your answers, that chain doesn't hold.<br><br>
        <strong>How to fix it:</strong>
        <ul style="margin:6px 0 0 18px;padding:0;">
            <li>Reopen the comparisons above and look for the ones you found hardest to decide.</li>
            <li>Try moving a few strong answers (&ldquo;extremely&rdquo;, &ldquo;very strongly&rdquo;)
                one step milder.</li>
            <li>If two things really do matter the same to you, set them to <strong>Equal importance</strong>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── STEP 3: AI TOGGLE ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Step 3 — AI Explanations (Optional)</div>', unsafe_allow_html=True)
use_ai = st.toggle(
    "✨  Enable Groq AI explanations, insider tips & best visiting hours",
    value=False,
    help="Uses LLaMA 3.3 70B via Groq (free). Adds a few seconds."
)

if use_ai:
    if HAS_SERVER_KEY:
        # Server-side key configured — participants don't need to do anything.
        st.markdown('<div style="background:#e6f4ea;border:1px solid #a8d5b0;border-radius:10px;padding:10px 14px;font-size:12px;color:#1e6e35;margin-top:8px;">🤖 <strong>Stage 4 ON</strong> — Groq LLM will generate personalised explanations after TOPSIS ranking. No setup needed.</div>', unsafe_allow_html=True)
    else:
        st.session_state.groq_api_key = st.text_input(
            "Enter your Groq API key",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_...",
            help="Your key is used only for this session and is never stored or logged.",
        )
        st.markdown(
            '<div style="font-size:12px;color:#555;margin-top:-8px;margin-bottom:8px;">'
            'Don\'t have a key? Get a free one at '
            '<a href="https://console.groq.com/keys" target="_blank" style="color:#1a4d2e;font-weight:700;">console.groq.com/keys</a>.'
            '</div>', unsafe_allow_html=True)

        if st.session_state.groq_api_key:
            st.markdown('<div style="background:#e6f4ea;border:1px solid #a8d5b0;border-radius:10px;padding:10px 14px;font-size:12px;color:#1e6e35;margin-top:8px;">🤖 <strong>Stage 4 ON</strong> — Groq LLM will generate personalised explanations after TOPSIS ranking.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#fff3cd;border:1px solid #f0c040;border-left:4px solid #e0a000;border-radius:10px;padding:10px 14px;font-size:12px;color:#5a4000;margin-top:8px;">⚠️ <strong>API key required.</strong> Enter a Groq API key above to enable Stage 4, or turn the toggle off to continue with Stages 1–3 only.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="background:#f4f5f8;border:1px solid #d0d4e0;border-radius:10px;padding:10px 14px;font-size:12px;color:#555;margin-top:8px;">🔬 <strong>Stage 4 OFF</strong> — Stages 1–3 only. Fast and deterministic.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── STEP 4: RUN ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Step 4 — Get Recommendations</div>', unsafe_allow_html=True)
btn_label = "🔍  Find My Activities  +  ✨ AI Explanations" if use_ai else "🔍  Find My Activities"

# The key actually used for this run: server key takes priority, else user-supplied.
ACTIVE_GROQ_KEY = SERVER_GROQ_KEY or st.session_state.groq_api_key

# Disable button if selections missing, AHP is inconsistent, or AI is on without any key
_btn_disabled = (
    (not st.session_state.selected)
    or (not ahp_preview["consistent"])
    or (use_ai and not ACTIVE_GROQ_KEY)
)

if not ahp_preview["consistent"] and st.session_state.selected:
    st.markdown("""
    <div style="background:#fff3cd;border:1px solid #f0c040;border-radius:10px;
                padding:10px 14px;font-size:12px;color:#5a4000;margin-bottom:8px;">
        🔒 <strong>Almost there.</strong> Just revisit Step 2 and adjust a few answers so they
        don't conflict — then this button will unlock. See the tips in the yellow box above.
    </div>
    """, unsafe_allow_html=True)

run = st.button(btn_label, key="run_btn",
                disabled=_btn_disabled,
                use_container_width=True, type="primary")

if run:
    spinners = [
        "⚙️  Stage 1 — Filtering by interests...",
        "📐  Stage 2 — Running AHP...",
        "📊  Stage 3 — Running TOPSIS...",
    ]
    with st.spinner(spinners[0]):
        results = run_pipeline(
            activities         = ACTIVITIES,
            selected_interests = st.session_state.selected,
            pairwise_matrix    = pairwise_mat,
            groq_api_key       = ACTIVE_GROQ_KEY,
            use_llm            = use_ai,
            top_n_cbf          = 14,
            top_n_final        = 8,
        )
    st.session_state.results = results

# ── RESULTS ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    R        = st.session_state.results
    ranked   = R.get("ranked", [])
    ahp_res  = R.get("ahp_result", {})
    stages   = R.get("stages_run", [])
    llm_done = "Stage 4 — Groq LLM Explanations" in stages

    if not ranked:
        st.warning("No results found. Try selecting different interests.")
    else:
        # Stage 4 status notice — recommendations below remain fully valid either way.
        _llm_status = R.get("llm_status", "off")
        if _llm_status == "rate_limited":
            st.markdown("""
            <div style="background:#eef3fb;border:1px solid #b9cbe8;border-left:4px solid #4a7bc8;
                        border-radius:0 10px 10px 0;padding:12px 16px;font-size:12.5px;
                        color:#1e3a5f;margin-bottom:14px;line-height:1.7;">
                ⏳ <strong>The AI explanation service is briefly busy.</strong>
                Your recommendations below are complete and fully valid — they come from the
                mathematical ranking (Stages 1–3), which does not depend on the AI layer.
                Only the written explanations are unavailable right now.
                Wait a moment and press the button again if you'd like to see them.
            </div>
            """, unsafe_allow_html=True)
        elif _llm_status == "auth_error":
            st.markdown("""
            <div style="background:#fff3cd;border:1px solid #f0c040;border-left:4px solid #e0a000;
                        border-radius:0 10px 10px 0;padding:12px 16px;font-size:12.5px;
                        color:#5a4000;margin-bottom:14px;line-height:1.7;">
                ⚠️ <strong>The AI explanation layer couldn't be authenticated.</strong>
                Your recommendations below are still complete and valid — they come from the
                mathematical ranking (Stages 1–3). If you entered your own API key, please
                check it at
                <a href="https://console.groq.com/keys" target="_blank" style="color:#1a4d2e;font-weight:700;">console.groq.com/keys</a>.
            </div>
            """, unsafe_allow_html=True)
        elif _llm_status == "error":
            st.markdown("""
            <div style="background:#eef3fb;border:1px solid #b9cbe8;border-left:4px solid #4a7bc8;
                        border-radius:0 10px 10px 0;padding:12px 16px;font-size:12.5px;
                        color:#1e3a5f;margin-bottom:14px;line-height:1.7;">
                ℹ️ <strong>The AI explanations couldn't be generated this time.</strong>
                Your recommendations below are complete and fully valid — they come from the
                mathematical ranking (Stages 1–3), which runs independently of the AI layer.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Algorithm transparency box
        vec_str   = "{" + ", ".join(f"{i}: 5" for i in st.session_state.selected) + "}"
        cr_status = f"CR={ahp_res.get('CR', 0):.3f} ({'✓ consistent' if ahp_res.get('consistent') else '⚠ check weights'})"
        llm_note  = "✅ Stage 4 LLM explanations applied." if llm_done else "Stage 4 off."
        stages_html = " → ".join(f'<span class="stage-badge">{s.split("—")[0].strip()}</span>' for s in stages)

        st.markdown(f"""
        <div class="algo-box">
            <strong>🔬 Pipeline executed:</strong> {stages_html}<br>
            <strong>Stage 1:</strong> User vector <code>{vec_str}</code> → cosine similarity on {len(ACTIVITIES)} activities → top {len(R.get('candidates',[]))} candidates selected.<br>
            <strong>Stage 2:</strong> AHP pairwise matrix computed → λ_max={ahp_res.get('lambda_max','?')}, {cr_status}.<br>
            <strong>Stage 3:</strong> TOPSIS with AHP weights → ranked by closeness to ideal solution. {llm_note}
        </div>
        """, unsafe_allow_html=True)

        # AHP weights summary
        if ahp_res.get("weights") is not None:
            w = ahp_res["weights"]
            w_html = " ".join(
                f'<span style="font-size:11px;background:#1a4d2e;color:#fff;padding:2px 9px;border-radius:20px;margin:2px;">'
                f'{CRIT_ICONS[k]} {CRITERIA_META[k]["label"]}: {round(float(w[i])*100,1)}%</span>'
                for i, k in enumerate(CRITERIA_KEYS)
            )
            st.markdown(f'<div style="margin-bottom:18px;">{w_html}</div>', unsafe_allow_html=True)

        # Results heading
        ints_str = "  ·  ".join(f"{CATEGORY_META[i]['emoji']} {CATEGORY_META[i]['label']}"
                                for i in st.session_state.selected)
        st.markdown(f"""
        <div class="result-header">
            <div class="section-label" style="margin:0">Your Top {len(ranked)} Attractions</div>
            <div class="result-count">{ints_str}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Click any card to expand full details and scores.")

        def dots(score, max_score=5):
            filled = round(score)
            return "●" * filled + "○" * (max_score - filled)

        for activity in ranked:
            rank        = activity["topsis_rank"]
            cat         = activity["category"]
            emoji       = CATEGORY_META.get(cat, {}).get("emoji", "📍")
            ts          = activity["topsis_score"]
            explanation = activity.get("explanation", "")
            tip         = activity.get("tip", "")
            best_time   = activity.get("best_time", "")

            with st.expander(
                f"#{rank}  {emoji}  {activity['name']}   —   📍 {activity['district']}   |   TOPSIS: {ts:.3f}",
                expanded=(rank <= 2)
            ):
                # CBF interest score dots
                score_items = "".join(
                    f'<div class="score-item">'
                    f'<span class="score-emoji">{CATEGORY_META[c]["emoji"]}</span>'
                    f'<span class="score-dots">{dots(activity["interest_scores"][c])}</span>'
                    f'<span class="score-label">{CATEGORY_META[c]["label"].split("&")[0].strip()}</span>'
                    f'</div>'
                    for c in st.session_state.selected
                )
                st.markdown(f'<div class="score-row">{score_items}</div>', unsafe_allow_html=True)

                # TOPSIS score bar
                bar_w = int(ts * 100)
                st.markdown(f"""
                <div class="topsis-bar-wrap">
                    <div class="topsis-bar-label">TOPSIS Closeness Score: {ts:.4f}</div>
                    <div class="topsis-bar-bg"><div class="topsis-bar-fill" style="width:{bar_w}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

                # AI explanation
                if explanation:
                    st.markdown(f'<div class="ai-box"><span class="ai-label">✨ AI Insight — Groq LLM</span>{explanation}</div>', unsafe_allow_html=True)

                # Description
                st.markdown(f'<div class="card-desc">{activity["description"]}</div>', unsafe_allow_html=True)

                # Tip
                if tip:
                    st.markdown(f'<div class="tip-box">💡 <strong>Insider Tip:</strong> {tip}</div>', unsafe_allow_html=True)

                # Best time
                if best_time:
                    st.markdown(f'<span class="time-box">🕐 Best time: {best_time}</span>', unsafe_allow_html=True)

                # TOPSIS criteria scores
                with st.expander("📊 View detailed TOPSIS criteria scores"):
                    crit_cols = st.columns(3)
                    for ci, (ck, cm) in enumerate(CRITERIA_META.items()):
                        raw_val      = activity["criteria"][ck]
                        weighted_val = activity.get("criteria_weighted", {}).get(ck, 0)
                        with crit_cols[ci % 3]:
                            st.metric(
                                label=f"{CRIT_ICONS[ck]} {cm['label']}",
                                value=f"{raw_val}/10",
                                delta=f"weighted: {weighted_val:.4f}"
                            )

                # Meta
                st.markdown(f"""
                <div class="meta-row">
                    <span class="meta-badge">💰 {activity['price_range']}</span>
                    <span class="meta-badge">🗓 {activity['best_season']}</span>
                    <span class="meta-badge">📂 {activity['sub_category']}</span>
                    <span class="topsis-badge">CBF: {activity['cbf_score']:.3f}</span>
                </div>
                """, unsafe_allow_html=True)

                # Tags
                tags_html = "".join(f'<span class="tag">{t}</span>' for t in activity["tags"])
                st.markdown(f'<div class="tags-row">{tags_html}</div>', unsafe_allow_html=True)

                # Source
                st.markdown(f'<div class="source-line">📖 Source: {activity["source"]}</div>', unsafe_allow_html=True)

elif not st.session_state.selected:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:38px;margin-bottom:10px;">🗺️</div>
        Select at least one interest above,<br>
        then click <strong style="color:#1a4d2e">Find My Activities</strong>.
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    AI-Powered Hybrid Tourism Recommendation System for İzmir &nbsp;·&nbsp; {len(ACTIVITIES)} attractions<br>
    Content-Based Filtering · AHP · TOPSIS · Groq LLM &nbsp;·&nbsp;
    Sources: Lonely Planet · TripAdvisor · Culinary Backstreets · Visit İzmir & more
</div>
""", unsafe_allow_html=True)
