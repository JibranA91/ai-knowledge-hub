/* Appearance is local to this browser, independent of accounts and organizations. */
(() => {
  const STORAGE_KEY = 'akh.appearance.v1';
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  let preference = readPreference();
  let dialog;
  let locationRequest = 0;

  function validLocation(location) {
    return location && Number.isFinite(location.latitude) && Number.isFinite(location.longitude)
      && Math.abs(location.latitude) <= 90 && Math.abs(location.longitude) <= 180;
  }

  function readPreference() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
      return {
        mode: ['auto', 'light', 'dark'].includes(saved?.mode) ? saved.mode : 'auto',
        location: validLocation(saved?.location) ? saved.location : null,
      };
    } catch {
      return { mode: 'auto', location: null };
    }
  }

  function savePreference() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(preference)); }
    catch { /* Private/restricted storage: keep working for this page's lifetime. */ }
    applyTheme();
  }

  function applyTheme() {
    let theme = preference.mode;
    if (theme === 'auto') {
      theme = preference.location && window.PrismSolar
        ? (PrismSolar.isDaylight(new Date(), preference.location.latitude, preference.location.longitude) ? 'light' : 'dark')
        : (systemTheme.matches ? 'dark' : 'light');
    }
    const changed = document.documentElement.dataset.theme !== theme;
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.themeMode = preference.mode;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = theme === 'dark' ? '#161924' : '#f7f8fc';
    document.querySelectorAll('[data-appearance-open]').forEach(button => {
      button.title = `Appearance: ${preference.mode === 'auto' ? 'automatic · ' : ''}${theme}`;
    });
    if (dialog) {
      dialog.querySelector('#prism-theme-mode').value = preference.mode;
      dialog.querySelector('#prism-theme-status').textContent = preference.mode !== 'auto'
        ? `${theme === 'dark' ? 'Dark' : 'Light'} appearance stays on until you choose Automatic.`
        : preference.location
          ? `${theme === 'light' ? 'Daylight' : 'Night'} at your saved location. Switches at sunrise/sunset, checked every minute and when you return to the page.`
          : 'No location saved. Automatic currently follows your device appearance. Set a location below to switch at sunrise and sunset.';
    }
    if (changed) window.dispatchEvent(new Event('prismthemechange'));
  }

  function updateGraphTheme() {
    // The graph intentionally has an opaque sandbox origin. Send only appearance,
    // never location or account data, and keep its existing sandbox restrictions.
    document.querySelector('#graph-frame')?.contentWindow?.postMessage({
      type: 'prismTheme', theme: document.documentElement.dataset.theme,
    }, '*');
  }
  window.addEventListener('prismthemechange', updateGraphTheme);

  // Runs in <head> before styles/body paint, using the last saved preference.
  applyTheme();
  setInterval(applyTheme, 60000);
  window.addEventListener('focus', applyTheme);
  window.addEventListener('pageshow', applyTheme);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) applyTheme();
  });
  systemTheme.addEventListener('change', applyTheme);
  window.addEventListener('storage', event => {
    if (event.key !== STORAGE_KEY && event.key !== null) return;
    preference = readPreference();
    locationRequest += 1;
    if (dialog) fillLocation();
    applyTheme();
  });

  function fillLocation() {
    dialog.querySelector('#prism-latitude').value = preference.location?.latitude ?? '';
    dialog.querySelector('#prism-longitude').value = preference.location?.longitude ?? '';
  }

  function locationMessage(message) {
    dialog.querySelector('#prism-location-message').textContent = message;
  }

  function useDeviceLocation() {
    if (!window.isSecureContext || !navigator.geolocation) {
      locationMessage('Device location needs HTTPS or localhost. You can enter coordinates instead.');
      return;
    }
    const request = ++locationRequest;
    const button = dialog.querySelector('#prism-device-location');
    button.disabled = true;
    locationMessage('Waiting for location permission…');
    navigator.geolocation.getCurrentPosition(position => {
      button.disabled = false;
      if (request !== locationRequest) return;
      // City-level precision is enough; do not persist exact device coordinates.
      const location = {
        latitude: Math.round(position.coords.latitude * 100) / 100,
        longitude: Math.round(position.coords.longitude * 100) / 100,
      };
      if (!validLocation(location)) {
        locationMessage('Location was unavailable. Enter coordinates instead.');
        return;
      }
      preference.location = location;
      fillLocation();
      savePreference();
      locationMessage('Approximate location saved only in this browser.');
    }, () => {
      button.disabled = false;
      if (request !== locationRequest) return;
      locationMessage('Location was not shared. Enter coordinates instead, or keep using your device appearance.');
    }, { enableHighAccuracy: false, timeout: 10000, maximumAge: 3600000 });
  }

  document.addEventListener('DOMContentLoaded', () => {
    const header = document.querySelector('#app-root > header');
    if (header) {
      // Wrapped navigation must not overlap the fixed notification panel.
      const observer = new ResizeObserver(() => {
        document.documentElement.style.setProperty('--app-header-height', `${header.getBoundingClientRect().height}px`);
      });
      observer.observe(header);
    }
    document.querySelector('#graph-frame')?.addEventListener('load', updateGraphTheme);
    updateGraphTheme();
    const triggers = document.querySelectorAll('[data-appearance-open]');
    if (!triggers.length) return;
    dialog = document.createElement('dialog');
    dialog.className = 'prism-appearance';
    dialog.id = 'prism-appearance';
    dialog.setAttribute('aria-labelledby', 'prism-theme-title');
    dialog.innerHTML = `
      <h2 id="prism-theme-title">Appearance</h2>
      <p>A little light. A little shade. Make this space yours.</p>
      <label for="prism-theme-mode">Theme</label>
      <select id="prism-theme-mode">
        <option value="auto">Automatic · sunrise / sunset</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
      <p id="prism-theme-status" role="status"></p>
      <form id="prism-location-form">
        <fieldset>
          <legend>Location for automatic mode</legend>
          <p class="prism-privacy">Enter your city's coordinates, or share an approximate device location. Saved only in this browser; never sent to the app server. Sunrise calculations work offline.</p>
          <div class="prism-location-fields">
            <div><label for="prism-latitude">Latitude</label><input id="prism-latitude" type="number" min="-90" max="90" step="any" placeholder="e.g. 40.71" required></div>
            <div><label for="prism-longitude">Longitude</label><input id="prism-longitude" type="number" min="-180" max="180" step="any" placeholder="e.g. -74.01" required></div>
          </div>
          <div class="prism-theme-actions">
            <button type="submit">Save location</button>
            <button type="button" id="prism-device-location">Use device location</button>
            <button type="button" id="prism-clear-location">Forget location</button>
          </div>
          <p id="prism-location-message" role="status"></p>
        </fieldset>
      </form>
      <button type="button" id="prism-theme-close">Done</button>`;
    document.body.appendChild(dialog);
    triggers.forEach(button => button.addEventListener('click', () => {
      fillLocation();
      locationMessage('');
      applyTheme();
      dialog.showModal();
    }));
    dialog.querySelector('#prism-theme-mode').addEventListener('change', event => {
      preference.mode = event.target.value;
      savePreference();
    });
    dialog.querySelector('#prism-theme-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => { locationRequest += 1; });
    dialog.querySelector('#prism-location-form').addEventListener('submit', event => {
      event.preventDefault();
      const location = {
        latitude: dialog.querySelector('#prism-latitude').valueAsNumber,
        longitude: dialog.querySelector('#prism-longitude').valueAsNumber,
      };
      if (!validLocation(location)) return;
      locationRequest += 1;
      preference.location = location;
      savePreference();
      locationMessage('Location saved only in this browser.');
    });
    dialog.querySelector('#prism-device-location').addEventListener('click', useDeviceLocation);
    dialog.querySelector('#prism-clear-location').addEventListener('click', () => {
      locationRequest += 1;
      preference.location = null;
      fillLocation();
      savePreference();
      locationMessage('Saved location removed.');
    });
    applyTheme();
  });
})();
