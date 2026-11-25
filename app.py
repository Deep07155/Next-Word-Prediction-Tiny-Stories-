"""TaleCraft: a polished, beginner-friendly Streamlit UI for TinyStories LSTM."""

from pathlib import Path
from html import escape

import streamlit as st
from inference import load_model_files, predict_words, generate_story


st.set_page_config(
    page_title="TaleCraft | AI Story Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_files"
FILES_NEEDED = ["best_lstm.pth", "tokenizer.json", "config.json"]

# Default prompts can be edited, and also demonstrate what this model can do.
STARTERS = {
    "🌲 Forest adventure": "Once upon a time a little girl went into the forest",
    "🐉 Friendly dragon": "One day a little dragon wanted to make a friend",
    "🌟 Magic discovery": "There was a small boy who found a magic stone",
}

DARK = {
    "background": "#0D1020", "sidebar": "#10152A", "surface": "#181D34",
    "surface2": "#222740", "text": "#F7F4FF", "muted": "#ABB1CF",
    "border": "#313651", "accent": "#B59AF7", "accent2": "#77DEC8",
    "hero_from": "#2C2355", "hero_to": "#182747", "input": "#191F38",
    "shadow": "rgba(0,0,0,.26)",
}
LIGHT = {
    "background": "#F8F7FC", "sidebar": "#FFFFFF", "surface": "#FFFFFF",
    "surface2": "#F0EBFA", "text": "#292543", "muted": "#6F7087",
    "border": "#E6E1F0", "accent": "#7357BA", "accent2": "#167F78",
    "hero_from": "#E9E0FD", "hero_to": "#DDF3F2", "input": "#FFFFFF",
    "shadow": "rgba(43,34,80,.06)",
}


def set_theme(theme):
    """Switch the app's custom page colors without changing global Streamlit settings."""
    colors = DARK if theme == "🌙 Dark" else LIGHT
    css = """
    <style>
      :root {
        --bg: __background__; --sidebar: __sidebar__; --panel: __surface__;
        --panel-2: __surface2__; --fg: __text__; --muted: __muted__;
        --stroke: __border__; --accent: __accent__; --mint: __accent2__;
        --input: __input__; --shade: __shadow__;
      }
      html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important; color: var(--fg) !important;
      }
      [data-testid="stHeader"] { background: transparent !important; }
      [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: var(--sidebar) !important; color: var(--fg) !important;
      }
      [data-testid="stSidebar"] { border-right: 1px solid var(--stroke); }
      .stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
      .stApp [data-testid="stWidgetLabel"] { color: var(--fg) !important; }
      .stApp [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
      [data-testid="stMainBlockContainer"] { max-width: 1100px; padding-top: 1.5rem; }
      .stApp div[data-baseweb="input"] input,
      .stApp div[data-baseweb="textarea"] textarea,
      .stApp [data-testid="stTextArea"] textarea,
      .stApp [data-testid="stTextInput"] input {
        background: var(--input) !important; color: var(--fg) !important;
      }
      .stApp div[data-baseweb="input"], .stApp div[data-baseweb="textarea"],
      .stApp [data-testid="stTextInput"], .stApp [data-testid="stTextArea"] {
        border-color: var(--stroke) !important;
      }
      .stApp [data-testid="stForm"] {
        background: var(--panel) !important; border: 1px solid var(--stroke);
        border-radius: 20px; padding: 22px 24px;
      }
      .stApp [data-testid="stTabs"] button {
        color: var(--muted) !important; font-weight: 650;
      }
      .stApp [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--accent) !important;
      }
      .stApp button[kind="primary"] {
        border: 0 !important; border-radius: 10px !important;
        background: var(--accent) !important;
        color: __primarytext__ !important; font-weight: 750 !important;
      }
      .stApp button[kind="secondary"] {
        border: 1px solid var(--stroke) !important;
        border-radius: 10px !important; background: var(--panel) !important;
        color: var(--fg) !important;
      }
      .stApp button:focus-visible { outline: 2px solid var(--accent) !important; }
      .hero {
        padding: 40px 42px; border-radius: 24px; margin: 4px 0 25px;
        background: linear-gradient(125deg, __hero_from__ 0%, __hero_to__ 100%);
        border: 1px solid var(--stroke); box-shadow: 0 12px 32px var(--shade);
        position: relative; overflow: hidden;
      }
      .hero::after { content: '✦'; position: absolute; right: 44px; top: 4px;
        color: var(--accent); font-size: 104px; opacity: .23; transform: rotate(-18deg); }
      .eyebrow { color: var(--accent) !important; font-size: .75rem; font-weight: 800;
        letter-spacing: .17em; text-transform: uppercase; margin-bottom: 12px; }
      .hero h1 { font-weight: 820; font-size: clamp(2rem, 4vw, 3.2rem); max-width: 640px;
        line-height: 1.13; letter-spacing: -.04em; margin: 0 0 14px; }
      .hero p { color: var(--muted) !important; font-size: 1.03rem; max-width: 610px;
        margin: 0; line-height: 1.65; }
      .brand { font-weight: 850; font-size: 1.55rem; letter-spacing: -.04em;
        color: var(--fg); margin: 8px 0 4px; }
      .brand-mark { color: var(--accent) !important; margin-right: 4px; }
      .small-note { color: var(--muted) !important; font-size: .86rem; line-height: 1.6; }
      .section-kicker { text-transform: uppercase; letter-spacing: .14em; font-size: .75rem;
        color: var(--accent); font-weight: 800; margin: 20px 0 4px; }
      .story-card { border: 1px solid var(--stroke); background: var(--panel);
        border-radius: 22px; padding: 30px; line-height: 1.93; font-size: 1.04rem;
        color: var(--fg); box-shadow: 0 7px 25px var(--shade); white-space: pre-wrap; }
      .result-title { font-size: 1.15rem; font-weight: 800; margin: 8px 0 14px; }
      .metric-line { color: var(--muted); font-size: .86rem; margin-top: 8px; }
      .pill { border-radius: 30px; padding: 5px 10px; background: var(--panel-2);
        color: var(--accent); font-size: .78rem; font-weight: 800;
        border: 1px solid var(--stroke); display: inline-block; margin: 2px 4px 5px 0; }
      .empty-note { background: var(--panel); border: 1px dashed var(--stroke);
        padding: 20px 22px; border-radius: 16px; color: var(--muted); }
      .word-card { border: 1px solid var(--stroke); border-radius: 12px;
        background: var(--panel); padding: 12px 16px; margin: 8px 0;
        display: flex; justify-content: space-between; align-items: center; }
      .word-card strong { color: var(--fg); font-size: 1.03rem; }
      .word-card span { color: var(--accent); font-weight: 800; }
      @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .7rem; }
        .hero { padding: 28px 22px; }
        .hero h1 { font-size: 2rem; }
        .hero::after { right: 0; font-size: 70px; }
        .story-card { padding: 20px; }
      }
    </style>
    """
    for name, value in colors.items():
        css = css.replace("__" + name + "__", value)
    css = css.replace("__primarytext__", "#130F22" if theme == "🌙 Dark" else "#FFFFFF")
    st.markdown(css, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_assets():
    # The large PyTorch model is loaded ONCE, not on every interaction.
    return load_model_files(MODEL_DIR)


# Initialize simple app state.
if "story_prompt" not in st.session_state:
    st.session_state.story_prompt = "Once upon a time there was a little girl"
if "story_result" not in st.session_state:
    st.session_state.story_result = ""
if "story_token_count" not in st.session_state:
    st.session_state.story_token_count = 0
if "word_prompt" not in st.session_state:
    st.session_state.word_prompt = "Once upon a time there was a little"
if "word_results" not in st.session_state:
    st.session_state.word_results = []

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">✦</span> TaleCraft</div>', unsafe_allow_html=True)
    st.caption("TinyStories · LSTM Story Studio")
    st.divider()
    theme_choice = st.radio("Appearance", ["🌙 Dark", "☀️ Light"], horizontal=True, key="theme_choice")
    st.divider()
    st.markdown("**Story controls**")
    max_tokens = st.slider("Story length (new tokens)", 30, 160, 90, 10)
    temperature = st.slider("Creativity", 0.5, 1.4, 0.85, 0.05)
    top_k = st.slider("Words considered per step", 5, 40, 15, 5)
    st.caption("Lower creativity = safer wording. Higher creativity = more surprising results.")
    st.divider()
    st.markdown("**About the model**")
    st.markdown('<div class="small-note">Trained with PyTorch on TinyStories using Kaggle GPU. The website runs inference on CPU—no API key, paid AI service, or training dataset required.</div>', unsafe_allow_html=True)

set_theme(theme_choice)

missing_files = [name for name in FILES_NEEDED if not (MODEL_DIR / name).is_file()]
ready = not missing_files
assets_error = None
model = tokenizer = config = None

if ready:
    try:
        model, tokenizer, config = get_assets()
    except Exception as exc:
        assets_error = str(exc)
        ready = False

st.markdown('''
<div class="hero">
  <div class="eyebrow">✧ The little AI writing studio</div>
  <h1>Every great story starts with a spark.</h1>
  <p>Write a sentence. Let your own TinyStories LSTM predict what comes next, one token at a time.</p>
</div>''', unsafe_allow_html=True)

if ready:
    st.markdown(
        f'<span class="pill">✓ Model ready</span>'
        f'<span class="pill">{int(config["num_layers"])} LSTM layers</span>'
        f'<span class="pill">{int(config["seq_len"])}-token context</span>'
        f'<span class="pill">CPU inference</span>',
        unsafe_allow_html=True,
    )
else:
    st.warning("The interface is ready, but your trained model files must be added before predictions will work.")
    with st.expander("How to connect your Kaggle model", expanded=True):
        if missing_files:
            st.write("Missing files:", ", ".join(missing_files))
        if assets_error:
            st.error(f"Model loading error: {assets_error}")
        st.markdown(
            "1. Run your **next-word-predict-LSTM.ipynb** on Kaggle.\n"
            "2. Download **trained_model_files.zip** from Kaggle Outputs.\n"
            "3. Extract **best_lstm.pth**, **tokenizer.json**, and **config.json** into `model_files/`.\n"
            "4. Restart Streamlit locally, or commit those files to your GitHub repository and redeploy."
        )

story_tab, word_tab, about_tab = st.tabs(["✦ Generate a story", "↗ Next-word predictions", "ⓘ About this project"])

with story_tab:
    st.markdown('<div class="section-kicker">01 / Create</div>', unsafe_allow_html=True)
    st.subheader("Give your story a beginning")
    st.caption("Choose a starter or write your own. Your prompt stays at the beginning of the story.")

    cols = st.columns(3)
    for (name, starter), col in zip(STARTERS.items(), cols):
        with col:
            if st.button(name, use_container_width=True, key=f"starter_{name}"):
                st.session_state.story_prompt = starter
                st.session_state.story_result = ""

    with st.form("story_form"):
        prompt = st.text_area(
            "Your opening sentence", key="story_prompt", height=118,
            placeholder="One day, a tiny dragon found a magical key...",
        )
        generate = st.form_submit_button("✦ Generate my story", type="primary", use_container_width=True, disabled=not ready)

    if generate:
        if not prompt.strip():
            st.warning("Please enter a starting sentence.")
        else:
            with st.spinner("Writing your tiny story..."):
                try:
                    text, count = generate_story(
                        model, tokenizer, config, prompt,
                        max_new_tokens=max_tokens, temperature=temperature, top_k=top_k,
                    )
                    st.session_state.story_result = text
                    st.session_state.story_token_count = count
                except Exception as exc:
                    st.error(f"Unable to generate story: {exc}")

    if st.session_state.story_result:
        st.markdown('<div class="result-title">Your generated story</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="story-card">{escape(st.session_state.story_result)}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Generated {st.session_state.story_token_count} new tokens. Each run can produce a different story.")
        st.download_button(
            "↓ Download as .txt", st.session_state.story_result,
            file_name="my_tiny_story.txt", mime="text/plain", use_container_width=True,
        )
    else:
        st.markdown('<div class="empty-note">✦ Your story will appear here after you press <strong>Generate my story</strong>.</div>', unsafe_allow_html=True)

with word_tab:
    st.markdown('<div class="section-kicker">02 / Predict</div>', unsafe_allow_html=True)
    st.subheader("What word comes next?")
    st.caption("Find the five most likely alphabetic next tokens, with probabilities based on your LSTM scores.")

    with st.form("predict_form"):
        next_prompt = st.text_input("Enter a sentence", key="word_prompt", placeholder="The little girl was very")
        predict = st.form_submit_button("↗ Predict next words", type="primary", use_container_width=True, disabled=not ready)

    if predict:
        if not next_prompt.strip():
            st.warning("Please enter a sentence first.")
        else:
            try:
                st.session_state.word_results = predict_words(model, tokenizer, config, next_prompt, 5)
            except Exception as exc:
                st.error(f"Could not predict: {exc}")

    if st.session_state.word_results:
        st.markdown('<div class="result-title">Top 5 word suggestions</div>', unsafe_allow_html=True)
        for word, probability in st.session_state.word_results:
            st.markdown(
                f'<div class="word-card"><strong>{escape(word)}</strong>'
                f'<span>{100*probability:.1f}%</span></div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(probability, 0.0), 1.0))
        st.caption("Probabilities are normalized across alphabetic vocabulary tokens only; other valid continuations may exist.")
    else:
        st.markdown('<div class="empty-note">↗ Type a sentence to see the model\'s highest-scoring next words.</div>', unsafe_allow_html=True)

with about_tab:
    st.markdown('<div class="section-kicker">03 / Behind the scenes</div>', unsafe_allow_html=True)
    st.subheader("A small LSTM with a big imagination")
    st.markdown(
        "**How it works:** The tokenizer turns the sentence into IDs. An embedding layer converts IDs "
        "to vectors. An LSTM processes the last few tokens, and a linear layer scores the next token. "
        "The generator repeats this process to write a longer story."
    )
    st.markdown("**Training:** TinyStories on Kaggle · **Deployment:** Streamlit Community Cloud · **Inference:** PyTorch on CPU")
    if ready:
        m1, m2, m3 = st.columns(3)
        m1.metric("Vocabulary", f"{config['vocab_size']:,}")
        m2.metric("LSTM layers", str(config["num_layers"]))
        m3.metric("Context length", f"{config['seq_len']} tokens")
    st.info(
        "This is a small research/portfolio model. It predicts tokens using limited context; "
        "its stories may repeat, drift from the topic, or make grammatical mistakes."
    )

st.markdown('<div class="metric-line">TaleCraft ✦ Powered by your PyTorch TinyStories LSTM · Built with Streamlit</div>', unsafe_allow_html=True)
