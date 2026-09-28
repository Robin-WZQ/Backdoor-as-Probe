const copyButton = document.getElementById('copy-citation');
const citation = document.getElementById('bibtex');

copyButton?.addEventListener('click', async () => {
  const text = citation?.innerText ?? '';
  let copied = false;
  try {
    await navigator.clipboard.writeText(text);
    copied = true;
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    copied = document.execCommand('copy');
    selection.removeAllRanges();
  }
  copyButton.textContent = copied ? 'Copied' : 'Select and copy';
  window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1500);
});
