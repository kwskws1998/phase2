/**
 * Internationalization (i18n) module.
 * Loads locale JSON files from the language/ directory and provides
 * translation via t() and DOM binding via data-i18n attributes.
 */

const AVAILABLE_LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'ko', name: '한국어' },
  { code: 'ja', name: '日本語' },
  { code: 'zh', name: '简体中文' },
  { code: 'fr', name: 'Français' },
  { code: 'de', name: 'Deutsch' },
  { code: 'es', name: 'Español' },
  { code: 'it', name: 'Italiano' },
  { code: 'pt', name: 'Português' },
  { code: 'nl', name: 'Nederlands' },
  { code: 'ru', name: 'Русский' },
  { code: 'ar', name: 'العربية' },
  { code: 'hi', name: 'हिन्दी' },
  { code: 'tr', name: 'Türkçe' },
  { code: 'pl', name: 'Polski' },
  { code: 'cs', name: 'Čeština' },
  { code: 'sv', name: 'Svenska' },
  { code: 'da', name: 'Dansk' },
  { code: 'no', name: 'Norsk' },
  { code: 'fi', name: 'Suomi' },
  { code: 'el', name: 'Ελληνικά' },
  { code: 'hu', name: 'Magyar' },
  { code: 'ro', name: 'Română' },
  { code: 'uk', name: 'Українська' },
  { code: 'vi', name: 'Tiếng Việt' },
  { code: 'th', name: 'ภาษาไทย' },
  { code: 'id', name: 'Bahasa Indonesia' }
];

const _localeCache = {};
let _currentLocale = {};
let _fallbackLocale = {};
let _currentLang = 'en';

function getCurrentLanguage() {
  return localStorage.getItem('ui_language') || 'en';
}

function getAvailableLanguages() {
  return AVAILABLE_LANGUAGES;
}

async function loadLocale(lang) {
  if (_localeCache[lang]) {
    return _localeCache[lang];
  }
  try {
    const resp = await fetch(`language/${lang}.json`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    _localeCache[lang] = data;
    return data;
  } catch (e) {
    console.warn(`[i18n] Failed to load locale '${lang}':`, e);
    return null;
  }
}

/**
 * Translate a key with optional parameter interpolation.
 * Falls back to English locale, then to the raw key.
 */
function t(key, params) {
  let str = _currentLocale[key] || _fallbackLocale[key] || key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      str = str.replace(new RegExp(`\\{${k}\\}`, 'g'), v);
    }
  }
  return str;
}

/**
 * Scan the DOM for data-i18n attributes and update text/placeholder/title.
 *   data-i18n="key"              -> sets textContent
 *   data-i18n-placeholder="key"  -> sets placeholder attribute
 *   data-i18n-title="key"        -> sets title attribute
 */
function applyLocale() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (key) el.textContent = t(key);
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    if (key) el.placeholder = t(key);
  });
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    const key = el.getAttribute('data-i18n-title');
    if (key) el.title = t(key);
  });
}

async function setLanguage(lang) {
  _currentLang = lang;
  localStorage.setItem('ui_language', lang);

  if (lang !== 'en') {
    _fallbackLocale = await loadLocale('en') || {};
  }
  _currentLocale = await loadLocale(lang) || _fallbackLocale;

  applyLocale();
}

async function initI18n() {
  const lang = getCurrentLanguage();
  _fallbackLocale = await loadLocale('en') || {};
  if (lang !== 'en') {
    _currentLocale = await loadLocale(lang) || _fallbackLocale;
  } else {
    _currentLocale = _fallbackLocale;
  }
  _currentLang = lang;
  applyLocale();
}
