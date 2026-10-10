'use strict';
const rows = JSON.parse(document.getElementById('results-data').textContent);
const datasetLabels = {'gpqa': 'GPQA', 'math-500': 'MATH-500', 'drop-800': 'DROP-800'};
const routerOrder = ['MLP', 'kNN', 'EmbedLLM', 'AvengersPro', 'GraphRouter', 'MIRT', 'RM-Softmax'];
function renderResults(dataset) {
  const body = document.getElementById('results-body');
  const fragment = document.createDocumentFragment();
  for (const router of routerOrder) {
    const tr = document.createElement('tr');
    const label = document.createElement('th');
    label.scope = 'row'; label.textContent = router; tr.append(label);
    for (const view of ['rewrite', 'decoding']) {
      const row = rows.find(r => r.dataset === dataset && r.router === router && r.view === view);
      const values = document.createElement('td');
      values.append(document.createTextNode(`${row.single_shot.toFixed(3)} → `));
      const value = document.createElement('strong'); value.textContent = row.dars.toFixed(3); values.append(value);
      const delta = document.createElement('td');
      delta.className = `delta ${row.delta > 0 ? 'positive' : 'negative'}`;
      delta.textContent = `${row.delta >= 0 ? '+' : ''}${row.delta.toFixed(3)}`;
      tr.append(values, delta);
    }
    fragment.append(tr);
  }
  body.replaceChildren(fragment);
  document.getElementById('results-caption').textContent = `${datasetLabels[dataset]} · Table 1 reported routing utility (higher is better)`;
  document.querySelectorAll('[data-dataset]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.dataset === dataset)));
}
document.querySelectorAll('[data-dataset]').forEach(button => button.addEventListener('click', () => renderResults(button.dataset.dataset)));
document.getElementById('copy-citation').addEventListener('click', async () => {
  const code = document.getElementById('bibtex');
  const status = document.getElementById('copy-status');
  try {
    if (!navigator.clipboard) throw new Error('Clipboard unavailable');
    await navigator.clipboard.writeText(code.textContent.trim());
    status.textContent = 'BibTeX copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange(); range.selectNodeContents(code);
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
