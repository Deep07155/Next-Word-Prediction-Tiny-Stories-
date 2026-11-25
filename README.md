# ✦ TaleCraft — TinyStories LSTM Story Studio

A polished **dark/light Streamlit website** for your TinyStories **PyTorch next-word prediction LSTM**. Train on **Kaggle** and deploy the frontend on **Streamlit Community Cloud**.

## LIVE APP - https://next-word-prediction-tinystories.streamlit.app/

## Features

- 🌙 **Dark** and ☀️ **Light** appearance selector
- ✦ Continue a sentence into a short TinyStory
- ↗ Show the **top 5 next-word suggestions** and normalized probabilities
- 🎚️ Adjust story length, temperature (creativity) and Top-K sampling
- 💾 Download generated stories as `.txt`
- 📱 Responsive design for desktop and mobile
- 🚀 CPU-only inference with a cached model (loads once)
- ⚠️ Helpful setup instructions if the model files are missing

## Folder structure

```text
TinyStories_TaleCraft_Streamlit/
├── app.py                  # Streamlit frontend (dark/light UI)
├── model.py                # Exactly the same LSTM architecture as training
├── inference.py            # Loading, next-word prediction, story generation
├── requirements.txt        # Includes CPU-only PyTorch
├── .streamlit/
│   └── config.toml         # Default visual theme
├── model_files/
│   ├── best_lstm.pth        # YOU download this from Kaggle
│   ├── tokenizer.json      # YOU download this from Kaggle
│   └── config.json         # YOU download this from Kaggle
├── training/
│   └── next-word-predict-LSTM.ipynb  # Your uploaded Kaggle notebook
└── README.md
```

**IMPORTANT:** The notebook upload includes training code and saved output logs, but **not the binary model files**. The website will not make predictions until you add the three Kaggle output files. We do not include fake pretrained weights.

## Step 1 — Download the trained files from Kaggle

1. Open your **trained** `next-word-predict-LSTM.ipynb` on Kaggle.
2. The notebook saves these in `/kaggle/working/`:
   - `best_lstm.pth` — the weights from the best validation-loss epoch
   - `tokenizer.json` — Hugging Face WordLevel tokenizer
   - `config.json` — model configuration (including `num_layers` and `seq_len`)
3. The last cell also creates `trained_model_files.zip` containing these three files.
4. Download `trained_model_files.zip` from the Kaggle notebook's **Output** panel, then extract it.
5. Place the three files **directly** inside this project's `model_files/` folder.

> Do not change or retrain the tokenizer after saving. All three files must come from the **same** training run. Do not put the TinyStories training data in your web app.

## Step 2 — Run on your Windows laptop (VS Code)

Use **Python 3.12** for consistent deployment with the supplied CPU-only PyTorch dependency.

From the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local link printed by Streamlit (usually `http://localhost:8501`). If the environment is missing any model file, the website explains which one.

> `requirements.txt` includes the official PyTorch CPU wheel index. It avoids downloading GPU libraries for a CPU-only web app. `torch==2.5.1+cpu` targets supported Python 3.11 installations. If your local system's pip cannot resolve `+cpu`, install PyTorch with the official CPU command first and use only the other dependencies from the requirements file.

## Step 3 — Upload to GitHub

**After** you have copied the three real model files into `model_files/`, create a new repository on GitHub, then run from the project folder:

```bash
git init
git add .
git commit -m "Build TinyStories LSTM Streamlit website"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tinystories-lstm-app.git
git push -u origin main
```

Replace `YOUR_USERNAME` with the username of the **GitHub account you are signed into**, and create the same repository name before pushing. Make sure `model_files/best_lstm.pth`, `model_files/tokenizer.json` and `model_files/config.json` appear on GitHub. GitHub's regular file limit is 100 MB; for larger weights, use Git LFS or another supported model storage method.

## Step 4 — Deploy on Streamlit Community Cloud

1. Open **https://share.streamlit.io/** and sign in with GitHub.
2. Select **Create app** / **Deploy an app** (wording can vary).
3. Select your GitHub repository and the `main` branch.
4. Set the **main file path** to `app.py`.
5. In **Advanced settings**, select **Python 3.11**, if offered.
6. Click **Deploy**. Streamlit installs dependencies from `requirements.txt`.
7. When finished, Streamlit provides a public `.streamlit.app` website link.

When GitHub changes, Streamlit normally rebuilds/redeploys the app.

**Memory tips:** No dataset, no `datasets` dependency, no optimizer, no GPU, and no model reload on each request. The model runs in CPU eval mode. Community Cloud still has shared memory and usage limits and cannot guarantee zero out-of-memory errors.

## What happens inside the model?

1. **Tokenizer:** converts the typed sentence into token IDs.
2. **Embedding:** changes token IDs into numerical vectors.
3. **LSTM:** examines the last `seq_len` tokens (8 in your uploaded notebook).
4. **Linear layer:** outputs a score for every vocabulary token.
5. **Next-word page:** picks the highest scoring alphabetic tokens and displays their probabilities.
6. **Story page:** uses Top-K sampling to select one token, appends it, and repeats.

The frontend automatically reads dimensions from `config.json` and supports the notebook's **2-layer, hidden-size-256** architecture (or any compatible configuration trained with the same model class).

## Troubleshooting

| Problem | Fix |
|---|---|
| Missing model files message | Copy the three Kaggle files directly into `model_files/` |
| `size mismatch` while loading | The tokenizer, config and model weights are from different runs, or the architecture changed |
| `ModuleNotFoundError: tokenizers` | Install dependencies from `requirements.txt` |
| App runs out of RAM | Ensure you use CPU-only PyTorch, no dataset, one process; reduce concurrent load or choose a larger host |
| Small/repetitive stories | The model uses only a short token window. Try creativity 0.8–1.0; a larger context requires retraining |
| Need a different Streamlit theme | Use the built-in **Appearance** selector. It changes the page palette per session |

## Model limitations

This is a compact LSTM trained on TinyStories, not a large language model. A prompt may lead to repetitive text or shift topics. The website creates continuations based on the model's next-token probabilities; it **does not fully understand the prompt** and cannot guarantee a complete, coherent story.

---

**Built with:** Python, PyTorch, Hugging Face Tokenizers, Streamlit, Kaggle, GitHub.
