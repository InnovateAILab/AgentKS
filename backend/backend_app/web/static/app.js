// Minimal JS to support sidebar drawer (mobile) and collapsible submenus.
// No external dependencies.

(function () {
  function qs(sel, root) { return (root || document).querySelector(sel); }
  function qsa(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }

  // Drawer toggle
  qsa('[data-drawer-toggle]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-drawer-toggle');
      const el = qs('#' + id);
      if (!el) return;
      el.classList.toggle('open');
    });
  });

  // Close sidebar on outside click (mobile)
  document.addEventListener('click', (e) => {
    const sidebar = qs('#sidebar');
    if (!sidebar) return;
    const isLarge = window.matchMedia('(min-width: 1024px)').matches;
    if (isLarge) return;
    const toggle = e.target.closest('[data-drawer-toggle="sidebar"]');
    if (toggle) return;
    if (!sidebar.classList.contains('open')) return;
    if (e.target.closest('#sidebar')) return;
    sidebar.classList.remove('open');
  });

  // Collapse toggles
  qsa('[data-collapse-toggle]').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-collapse-toggle');
      const el = qs('#' + targetId);
      if (!el) return;
      el.classList.toggle('hidden');
    });
  });
})();
