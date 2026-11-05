# PARAM BRAIN V1 — iPad Edition

A tiny character-level, single-head causal Transformer, implemented in TensorFlow.js. The model is initialized randomly and trained on your own text directly in Safari. No pretrained model and no paid AI API.

## Run on an iPad
1. Extract the ZIP in the Files app.
2. Upload the three files (`index.html`, `style.css`, `brain.js`) to a new GitHub repository, keeping them in the repository root.
3. On Netlify, deploy the GitHub repository as a static site (no build command; publish directory `.`). Alternatively use GitHub Pages.
4. Open the deployed HTTPS URL in Safari. Internet is required to load TensorFlow.js from the CDN; the actual model training stays on the iPad.
5. Tap **Create model**, then **Train 50 steps** repeatedly, then **Generate 80 characters**.
6. Tap **Export weights** to save a JSON checkpoint to Files. Import it later using the same training text character set.

## Notes
- Keep Safari in the foreground while training. Start with the sample text and short training sessions.
- This is an educational character-prediction model, not an instruction-following assistant.
- Training changes model weights but does not change the training text automatically.
- Checkpoint restores model weights but not optimizer state; resumed training restarts Adam's moment estimates.
- TensorFlow.js is loaded from jsDelivr. The page is not guaranteed to work offline until its dependencies are locally bundled and cached.
- Export and retain your weights before refreshing the page.
