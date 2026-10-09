// Pulsebot — LinkedIn content script
// Detects the author you're viewing and injects a floating "Chat" button.

(function () {
  'use strict';

  let currentHandle = null;

  // ── Author detection ─────────────────────────────────────────────────────────

  function detectAuthor() {
    const path = location.pathname;

    // Profile page: linkedin.com/in/handle
    const profileMatch = path.match(/^\/in\/([^/?#]+)/);
    if (profileMatch) {
      const handle = profileMatch[1];
      const name = document.querySelector('h1')?.innerText?.trim() || handle;
      return { handle, name, source: 'profile' };
    }

    // Article / pulse page: linkedin.com/pulse/...
    if (path.startsWith('/pulse/')) {
      // Author link inside article: varies by layout, try common selectors
      const authorEl = document.querySelector(
        '.article-author__name, .base-main-card__title, ' +
        '[data-control-name="actor"] .actor-name, ' +
        '.update-author__name'
      );
      const name = authorEl?.innerText?.trim();
      if (name) return { handle: slugify(name), name, source: 'article' };

      // Fall back: look for /in/ links on the page as a proxy for author
      const inLink = document.querySelector('a[href*="/in/"]');
      if (inLink) {
        const m = inLink.href.match(/\/in\/([^/?#]+)/);
        if (m) {
          const fallbackName = inLink.innerText?.trim() || m[1];
          return { handle: m[1], name: fallbackName, source: 'article' };
        }
      }
    }

    // Newsletter home page: linkedin.com/newsletters/...
    if (path.startsWith('/newsletters/')) {
      const authorEl = document.querySelector(
        '.newsletter-details-author__name, .follow-member-action, ' +
        '.newsletter-header .actor-name'
      );
      const name = authorEl?.innerText?.trim();
      if (name) return { handle: slugify(name), name, source: 'newsletter' };
    }

    return null;
  }

  function slugify(name) {
    return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  }

  // ── Floating button ───────────────────────────────────────────────────────────

  function removeBtn() {
    document.getElementById('pulsebot-chat-btn')?.remove();
    currentHandle = null;
  }

  async function maybeInject() {
    const author = detectAuthor();
    if (!author) { removeBtn(); return; }

    const cfg = await chrome.storage.sync.get(['serverUrl']);
    const serverUrl = (cfg.serverUrl || '').replace(/\/$/, '');
    if (!serverUrl) return; // not configured — skip injection

    // Nothing to do if the same author is already showing
    if (author.handle === currentHandle) return;
    currentHandle = author.handle;

    // Persist context so the popup can pick it up immediately
    chrome.storage.local.set({ linkedinContext: author });

    // Remove any stale button
    document.getElementById('pulsebot-chat-btn')?.remove();

    const btn = document.createElement('button');
    btn.id = 'pulsebot-chat-btn';
    btn.innerHTML = `<span aria-hidden="true">⚡</span> Chat`;
    btn.title = `Chat with ${author.name}'s newsletter archive`;
    Object.assign(btn.style, {
      position:    'fixed',
      bottom:      '24px',
      right:       '24px',
      zIndex:      '2147483647',
      background:  '#1e1b4b',
      color:       '#fff',
      border:      'none',
      borderRadius: '24px',
      padding:     '9px 17px',
      fontSize:    '13px',
      fontWeight:  '600',
      fontFamily:  "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      cursor:      'pointer',
      display:     'flex',
      alignItems:  'center',
      gap:         '6px',
      boxShadow:   '0 4px 18px rgba(0,0,0,.28)',
      transition:  'background .15s, transform .1s',
      userSelect:  'none',
      lineHeight:  '1',
    });

    btn.addEventListener('mouseenter', () => { btn.style.background = '#312e81'; });
    btn.addEventListener('mouseleave', () => { btn.style.background = '#1e1b4b'; });
    btn.addEventListener('mousedown', () => { btn.style.transform = 'scale(.95)'; });
    btn.addEventListener('mouseup',   () => { btn.style.transform = ''; });

    btn.addEventListener('click', () => {
      const url = `${serverUrl}/c/${encodeURIComponent(author.handle)}`;
      window.open(url, '_blank', 'noopener noreferrer');
    });

    document.body.appendChild(btn);
  }

  // ── SPA navigation detection ──────────────────────────────────────────────────
  // LinkedIn is a React SPA — URL changes via history.pushState don't fire popstate.

  function onNavigate() {
    removeBtn();
    chrome.storage.local.remove('linkedinContext');
    // Give the new page's DOM a moment to render
    setTimeout(maybeInject, 600);
  }

  const origPush    = history.pushState.bind(history);
  const origReplace = history.replaceState.bind(history);

  history.pushState = function (...args) {
    origPush(...args);
    onNavigate();
  };
  history.replaceState = function (...args) {
    origReplace(...args);
    onNavigate();
  };

  window.addEventListener('popstate', onNavigate);

  // ── Boot ──────────────────────────────────────────────────────────────────────

  // Run once immediately, then again after LinkedIn hydrates the DOM
  maybeInject();
  setTimeout(maybeInject, 1200);
  setTimeout(maybeInject, 2800);

  // Also watch for DOM changes (lazy-loaded author info)
  let watchTimer = null;
  const observer = new MutationObserver(() => {
    if (currentHandle) return; // already injected
    clearTimeout(watchTimer);
    watchTimer = setTimeout(maybeInject, 300);
  });
  observer.observe(document.body, { childList: true, subtree: true });

  // Stop watching once we've successfully injected
  const stopWatcher = setInterval(() => {
    if (currentHandle) { observer.disconnect(); clearInterval(stopWatcher); }
  }, 3000);
})();
