// Pulsebot Chrome Extension — popup.js

const $ = id => document.getElementById(id);

let serverUrl = '';
let authorHandle = '';
let streaming = false;
let history = [];

async function init() {
  const cfg = await chrome.storage.sync.get(['serverUrl', 'authorHandle']);
  serverUrl = (cfg.serverUrl || '').replace(/\/$/, '');
  authorHandle = cfg.authorHandle || '';

  $('settings-btn').onclick = () => chrome.runtime.openOptionsPage();
  $('open-options').onclick = () => chrome.runtime.openOptionsPage();

  if (!serverUrl) {
    $('setup-screen').hidden = false;
    return;
  }

  $('setup-screen').hidden = true;
  $('chat-screen').hidden = false;

  $('author-input').value = authorHandle;

  $('author-go').onclick = () => loadAuthor($('author-input').value.trim());
  $('author-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') loadAuthor($('author-input').value.trim());
  });

  $('send-btn').onclick = sendMessage;
  $('q').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  $('q').addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 88) + 'px';
  });

  if (authorHandle) await loadAuthor(authorHandle);
}

async function loadAuthor(handle) {
  if (!handle) return;
  authorHandle = handle.toLowerCase().replace(/[^a-z0-9\-_]/g, '');
  if (!authorHandle) return;
  $('author-input').value = authorHandle;
  history = [];
  clearMessages();

  try {
    const res = await fetch(`${serverUrl}/authors/${encodeURIComponent(authorHandle)}`);
    if (!res.ok) throw new Error(`Author "${authorHandle}" not found on this server (${res.status})`);
    const data = await res.json();

    const displayName = data.display_name ||
      authorHandle.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    $('header-name').textContent = displayName;
    $('header-sub').textContent = `${data.articles ?? 0} articles · ${data.chunks ?? 0} chunks`;
    document.title = displayName;

    const av = $('header-avatar');
    if (data.photo_url) {
      av.style.background = 'none';
      av.innerHTML = `<img src="${esc(data.photo_url)}" alt="">`;
    } else {
      av.style.background = '#6366f1';
      av.textContent = (displayName[0] || '⚡').toUpperCase();
    }

    addMsg('assistant', `Hi! I'm a chatbot trained on ${displayName}'s newsletter archive. What would you like to know?`);
    $('q').focus();
  } catch (err) {
    $('header-name').textContent = 'Pulsebot';
    $('header-sub').textContent = 'LinkedIn Newsletter Chat';
    addMsg('error', err.message);
  }
}

async function sendMessage() {
  if (streaming) return;
  const text = $('q').value.trim();
  if (!text || !authorHandle) return;

  $('q').value = '';
  $('q').style.height = 'auto';

  addMsg('user', text);
  history.push({ role: 'user', content: text });

  const typingEl = addTyping();
  $('send-btn').disabled = true;
  streaming = true;

  let assistantText = '';
  let msgEl = null;

  try {
    const res = await fetch(`${serverUrl}/chat/${encodeURIComponent(authorHandle)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text, history }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        let ev;
        try { ev = JSON.parse(raw); } catch { continue; }

        if (ev.type === 'token' && ev.text) {
          if (!msgEl) {
            typingEl.remove();
            msgEl = document.createElement('div');
            msgEl.className = 'msg assistant';
            $('messages').appendChild(msgEl);
          }
          assistantText += ev.text;
          msgEl.textContent = assistantText;
          scrollBottom();
        } else if (ev.type === 'sources' && ev.sources?.length) {
          const sourcesEl = document.createElement('div');
          sourcesEl.className = 'sources';
          sourcesEl.textContent = 'Sources: ';
          ev.sources.slice(0, 4).forEach((s, i) => {
            const a = document.createElement('a');
            a.href = s.url || '#';
            a.target = '_blank';
            a.rel = 'noopener';
            a.textContent = s.title || `[${i + 1}]`;
            sourcesEl.appendChild(a);
          });
          $('messages').appendChild(sourcesEl);
          scrollBottom();
        } else if (ev.type === 'error') {
          typingEl?.remove();
          addMsg('error', ev.message);
        } else if (ev.type === 'limit') {
          typingEl?.remove();
          addMsg('limit', `Daily limit reached (${ev.daily_limit} questions/day on the free tier). Upgrade for more.`);
        }
      }
    }

    if (assistantText) {
      history.push({ role: 'assistant', content: assistantText });
    }
  } catch (err) {
    typingEl?.remove();
    msgEl?.remove();
    addMsg('error', `Error: ${err.message}`);
  } finally {
    $('send-btn').disabled = false;
    streaming = false;
    $('q').focus();
  }
}

function addMsg(type, text) {
  const el = document.createElement('div');
  el.className = `msg ${type}`;
  el.textContent = text;
  $('messages').appendChild(el);
  scrollBottom();
  return el;
}

function addTyping() {
  const el = document.createElement('div');
  el.className = 'typing';
  el.innerHTML = '<span></span><span></span><span></span>';
  $('messages').appendChild(el);
  scrollBottom();
  return el;
}

function clearMessages() { $('messages').innerHTML = ''; }
function scrollBottom() { const m = $('messages'); m.scrollTop = m.scrollHeight; }
function esc(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

init();
