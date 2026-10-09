// Pulsebot Chrome Extension — options.js

const $ = id => document.getElementById(id);

async function load() {
  const cfg = await chrome.storage.sync.get(['serverUrl', 'authorHandle']);
  $('serverUrl').value = cfg.serverUrl || '';
  $('authorHandle').value = cfg.authorHandle || '';
}

async function save() {
  const serverUrl = $('serverUrl').value.trim().replace(/\/$/, '');
  const authorHandle = $('authorHandle').value.trim();

  if (serverUrl && !/^https?:\/\/.+/.test(serverUrl)) {
    showStatus('Server URL must start with http:// or https://', 'err');
    return;
  }

  await chrome.storage.sync.set({ serverUrl, authorHandle });
  showStatus('Settings saved!', 'ok');
}

async function testConnection() {
  const url = $('serverUrl').value.trim().replace(/\/$/, '');
  if (!url) { showStatus('Enter a server URL first.', 'err'); return; }

  showStatus('Connecting…', '');

  try {
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(6000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    showStatus(`Connected! Server status: ${data.status}`, 'ok');
  } catch (err) {
    showStatus(`Could not reach server: ${err.message}`, 'err');
  }
}

function showStatus(msg, type) {
  const el = $('status');
  el.textContent = msg;
  el.className = `status ${type}`;
}

$('save-btn').onclick = save;
$('test-btn').onclick = testConnection;

load();
