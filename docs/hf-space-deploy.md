# Hugging Face Space deploy

The Gradio app is Space-ready. A live Space was **not** created from this agent run because no Hugging Face write token (`HF_TOKEN`) was available in the environment.

## Exact steps (Waleed)

1. Create a token at https://huggingface.co/settings/tokens with **Write** + Spaces access.
2. Locally (or in CI):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the token
huggingface-cli repo create spamemaildetection --type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/<YOUR_HF_USER>/spamemaildetection
git push space main
```

3. Confirm the Space builds and the four presets classify correctly (prize SMS / bank scam → spam; work email / casual text → ham).
4. Update the Live Demo badge URL in `README.md` if the Space slug differs from `waleedarif1011/spamemaildetection`.

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter)
- `src/`
- `artifacts/` (`spam_pipeline.joblib`, `metrics.json`)
- `assets/` (optional plots; not required for the app to run)
