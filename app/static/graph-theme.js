/* Color-only updates for the sandboxed graph; topology and navigation stay intact. */
window.applyPrismGraphTheme = () => {
  if (typeof nodes === 'undefined' || typeof edges === 'undefined') return;
  const style = getComputedStyle(document.documentElement);
  const color = token => style.getPropertyValue(token).trim();
  const category = id => ({ sources: '--info', concepts: '--teal', queries: '--ai' })[id.split('/')[0]] || '--text-muted';
  nodes.update(nodes.get().map(node => ({
    id: node.id,
    color: {
      background: color('--surface2'), border: color(category(node.id)),
      highlight: { background: color('--primary-light'), border: color('--primary') },
      hover: { background: color('--hover'), border: color(category(node.id)) },
    },
    font: { color: color('--text'), strokeColor: color('--bg') },
  })));
  edges.update(edges.get().map(edge => ({
    id: edge.id,
    color: { color: color('--border-strong'), highlight: color('--primary'), hover: color('--primary') },
  })));
};
window.addEventListener('message', event => {
  if (event.source !== window.parent || event.data?.type !== 'prismTheme') return;
  if (!['light', 'dark'].includes(event.data.theme)) return;
  document.documentElement.dataset.theme = event.data.theme;
  window.applyPrismGraphTheme();
});
