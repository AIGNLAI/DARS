# DARS project page

The paper page is a buildless static site in `docs/`, hosted by GitHub Pages from the `main` branch's `/docs` directory at <https://aignlai.github.io/DARS/>. `.nojekyll` prevents unintended processing of research documentation.

Edit `index.html`, `assets/style.css`, and `assets/site.js` to maintain the page. The full Table 1 values are embedded in `index.html` for offline task switching; keep them consistent with `paper_results.csv` and `RESULTS.md`. The initial GPQA table remains readable with JavaScript disabled. The framework image is Figure 3 from the arXiv paper and is also used by the repository README. The preferred citation lives in `CITATION.cff`.

`HF_DATASET_CARD.md` is the maintained source for the Hugging Face dataset card. Publish it as that dataset repository's `README.md`, with `assets/framework.png`. Its schema declarations permit the train/test field differences without rewriting the released JSONL data.

Preview locally with `python -m http.server 8000 --directory docs`, then visit <http://localhost:8000/>. GitHub Pages publishes after changes are pushed to `main`.
