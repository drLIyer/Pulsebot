// Pulsebot Chrome Extension — background service worker

chrome.runtime.onInstalled.addListener(({ reason }) => {
  if (reason === 'install') {
    // Open settings on first install so the user can configure the server URL
    chrome.storage.sync.get(['serverUrl'], cfg => {
      if (!cfg.serverUrl) chrome.runtime.openOptionsPage();
    });
  }
});
