"use strict";

const telegram = window.Telegram?.WebApp ?? null;
const state = {
  settings: null,
  corsOrigins: [],
  refreshing: false,
  toastTimer: null,
};

const elements = {
  authState: document.getElementById("authState"),
  dashboard: document.getElementById("dashboard"),
  refreshButton: document.getElementById("refreshButton"),
  profilePhoto: document.getElementById("profilePhoto"),
  profileFallback: document.getElementById("profileFallback"),
  profileName: document.getElementById("profileName"),
  profileHandle: document.getElementById("profileHandle"),
  systemBadge: document.getElementById("systemBadge"),
  totalUsers: document.getElementById("totalUsers"),
  messageCount: document.getElementById("messageCount"),
  redisLatency: document.getElementById("redisLatency"),
  uptime: document.getElementById("uptime"),
  lastUpdated: document.getElementById("lastUpdated"),
  botHealth: document.getElementById("botHealth"),
  redisHealth: document.getElementById("redisHealth"),
  databaseHealth: document.getElementById("databaseHealth"),
  botMode: document.getElementById("botMode"),
  maintenanceToggle: document.getElementById("maintenanceToggle"),
  maintenanceNotice: document.getElementById("maintenanceNotice"),
  corsForm: document.getElementById("corsForm"),
  corsOrigin: document.getElementById("corsOrigin"),
  corsList: document.getElementById("corsList"),
  corsEmpty: document.getElementById("corsEmpty"),
  runtimeForm: document.getElementById("runtimeForm"),
  runtimeFields: document.getElementById("runtimeFields"),
  saveRuntimeButton: document.getElementById("saveRuntimeButton"),
  toast: document.getElementById("toast"),
};

document.querySelectorAll("[data-nav-target]").forEach((item) => {
  item.addEventListener("click", () => {
    document.querySelectorAll("[data-nav-target]").forEach((candidate) => {
      candidate.classList.toggle("active", candidate === item);
    });
  });
});

function haptic(type = "light") {
  try {
    telegram?.HapticFeedback?.impactOccurred(type);
  } catch {
    // Haptics are optional and platform-dependent.
  }
}

function showToast(message, isError = false) {
  window.clearTimeout(state.toastTimer);
  elements.toast.textContent = String(message);
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("show");
  state.toastTimer = window.setTimeout(() => elements.toast.classList.remove("show"), 3200);
}

function compactNumber(value) {
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 })
    .format(Number(value) || 0);
}

async function api(path, options = {}) {
  // Read this at request time. Telegram can populate `initData` immediately
  // after the WebApp bridge becomes ready; capturing it during script startup
  // can otherwise retain an empty or stale value.
  const initData = String(window.Telegram?.WebApp?.initData || "").trim();
  if (!initData) {
    throw new Error("Open this dashboard from the bot inside Telegram.");
  }
  const headers = new Headers(options.headers || {});
  headers.set("X-Telegram-Init-Data", initData);
  // Keep a standard authorization transport as a fallback for proxies that
  // strip custom X-* headers on state-changing requests.
  headers.set("Authorization", `Bearer ${initData}`);
  headers.set("Accept", "application/json");
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    ...options,
    headers,
    credentials: "same-origin",
    cache: "no-store",
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const detail = typeof payload?.detail === "string"
      ? payload.detail
      : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return payload;
}

function setHealth(element, label, kind = "ok") {
  element.textContent = label;
  element.classList.toggle("warn", kind === "warn");
  element.classList.toggle("down", kind === "down");
}

function renderProfile(payload) {
  const user = payload.user || {};
  const fullName = [user.first_name, user.last_name].filter(Boolean).join(" ") || "Administrator";
  elements.profileName.textContent = fullName;
  elements.profileHandle.textContent = user.username
    ? `@${user.username} · Telegram verified`
    : `ID ${user.id ?? "—"} · Telegram verified`;
  elements.profileFallback.textContent = fullName.slice(0, 1).toUpperCase();

  let photoUrl = null;
  try {
    const candidate = new URL(user.photo_url || "");
    if (candidate.protocol === "https:") photoUrl = candidate.href;
  } catch {
    photoUrl = null;
  }
  if (photoUrl) {
    elements.profilePhoto.src = photoUrl;
    elements.profilePhoto.alt = `${fullName} profile photo`;
    elements.profilePhoto.referrerPolicy = "no-referrer";
    elements.profilePhoto.classList.remove("is-hidden");
    elements.profileFallback.classList.add("is-hidden");
  }
}

function renderStats(payload) {
  const bot = payload.bot || {};
  const usage = payload.usage || {};
  const redis = payload.redis || {};
  const database = payload.database || {};

  elements.totalUsers.textContent = compactNumber(usage.total_users);
  elements.messageCount.textContent = compactNumber(usage.message_count);
  elements.redisLatency.textContent = redis.ok && redis.latency_ms != null
    ? `${redis.latency_ms} ms`
    : "Offline";
  elements.uptime.textContent = bot.uptime || "Starting";
  elements.botMode.textContent = bot.mode || "—";
  elements.lastUpdated.textContent = `Updated ${new Date(payload.generated_at).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  })}`;

  setHealth(elements.botHealth, bot.active ? "Operational" : "Starting", bot.active ? "ok" : "warn");
  setHealth(elements.redisHealth, redis.ok ? "Healthy" : "Unavailable", redis.ok ? "ok" : "down");
  setHealth(
    elements.databaseHealth,
    database.ok ? "Connected" : (database.memory_fallback ? "Memory fallback" : "Unavailable"),
    database.ok ? "ok" : "warn",
  );

  const overallHealthy = Boolean(bot.active && redis.ok);
  elements.systemBadge.classList.toggle("down", !overallHealthy);
  elements.systemBadge.lastElementChild.textContent = overallHealthy ? "System operational" : "Attention required";
  renderMaintenance(Boolean(bot.maintenance_mode));
}

function renderMaintenance(enabled) {
  elements.maintenanceToggle.checked = enabled;
  elements.maintenanceNotice.textContent = enabled
    ? "Maintenance is ON. Normal-user features are paused."
    : "Normal bot service is active.";
  elements.maintenanceNotice.classList.toggle("warning", enabled);
}

function makeRuntimeControl(key, spec) {
  const wrapper = document.createElement("div");
  wrapper.className = "setting-field";

  const copy = document.createElement("div");
  const label = document.createElement("label");
  label.htmlFor = `runtime-${key}`;
  label.textContent = spec.label || key;
  const help = document.createElement("small");
  help.textContent = spec.help || "Runtime setting";
  copy.append(label, help);

  const control = document.createElement("div");
  control.className = "setting-control";
  let input;
  if (spec.kind === "bool") {
    const switchLabel = document.createElement("label");
    switchLabel.className = "switch";
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(spec.value);
    const track = document.createElement("span");
    track.className = "switch-track";
    track.setAttribute("aria-hidden", "true");
    switchLabel.append(input, track);
    control.append(switchLabel);
  } else {
    input = document.createElement("input");
    input.type = "number";
    input.value = String(spec.value ?? "");
    input.step = spec.kind === "float" ? "0.1" : "1";
    if (spec.min != null) input.min = String(spec.min);
    if (spec.max != null) input.max = String(spec.max);
    control.append(input);
  }
  input.id = `runtime-${key}`;
  input.name = key;
  input.dataset.kind = spec.kind || "str";
  wrapper.append(copy, control);
  return wrapper;
}

function renderSettings(payload) {
  state.settings = payload;
  renderMaintenance(Boolean(payload.maintenance_mode));
  elements.runtimeFields.replaceChildren();
  Object.entries(payload.runtime || {}).forEach(([key, spec]) => {
    elements.runtimeFields.append(makeRuntimeControl(key, spec));
  });
}

function renderCors(origins) {
  state.corsOrigins = Array.isArray(origins) ? origins : [];
  elements.corsList.replaceChildren();
  elements.corsEmpty.classList.toggle("is-hidden", state.corsOrigins.length > 0);
  state.corsOrigins.forEach((origin) => {
    const row = document.createElement("div");
    row.className = "origin-chip";
    const value = document.createElement("span");
    value.textContent = origin;
    value.title = origin;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "button danger";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => removeOrigin(origin, remove));
    row.append(value, remove);
    elements.corsList.append(row);
  });
}

async function refreshAll({ initial = false, silent = false } = {}) {
  if (state.refreshing) return;
  state.refreshing = true;
  elements.refreshButton.disabled = true;
  try {
    const [profile, stats, settings, cors] = await Promise.all([
      api("/api/admin/me"),
      api("/api/admin/stats"),
      api("/api/admin/settings"),
      api("/api/admin/cors"),
    ]);
    renderProfile(profile);
    renderStats(stats);
    renderSettings(settings);
    renderCors(cors.origins);
    elements.authState.classList.add("is-hidden");
    elements.dashboard.classList.remove("is-hidden");
    if (!initial && !silent) {
      haptic("light");
      showToast("Dashboard refreshed.");
    }
  } catch (error) {
    if (initial) {
      elements.authState.classList.add("error");
      elements.authState.replaceChildren();
      const copy = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = "Admin access denied";
      const detail = document.createElement("p");
      detail.textContent = error.message;
      copy.append(title, detail);
      elements.authState.append(copy);
    } else {
      showToast(error.message, true);
    }
  } finally {
    state.refreshing = false;
    elements.refreshButton.disabled = false;
  }
}

async function updateMaintenance(nextValue) {
  elements.maintenanceToggle.disabled = true;
  try {
    const payload = await api("/api/admin/settings", {
      method: "POST",
      body: JSON.stringify({ maintenance_mode: nextValue }),
    });
    renderMaintenance(Boolean(payload.maintenance_mode));
    haptic(nextValue ? "medium" : "light");
    showToast(nextValue ? "Maintenance mode enabled." : "Maintenance mode disabled.");
  } catch (error) {
    renderMaintenance(!nextValue);
    showToast(error.message, true);
  } finally {
    elements.maintenanceToggle.disabled = false;
  }
}

async function confirmMaintenance(nextValue) {
  const message = nextValue
    ? "Pause bot features for normal users?"
    : "Resume bot features for normal users?";
  if (typeof telegram?.showConfirm === "function") {
    telegram.showConfirm(message, (confirmed) => {
      if (confirmed) updateMaintenance(nextValue);
      else renderMaintenance(!nextValue);
    });
    return;
  }
  if (window.confirm(message)) await updateMaintenance(nextValue);
  else renderMaintenance(!nextValue);
}

async function addOrigin(event) {
  event.preventDefault();
  const origin = elements.corsOrigin.value.trim();
  if (!origin) return;
  const submit = elements.corsForm.querySelector("button[type='submit']");
  submit.disabled = true;
  try {
    const payload = await api("/api/admin/cors", {
      method: "POST",
      body: JSON.stringify({ origin }),
    });
    elements.corsOrigin.value = "";
    renderCors(payload.origins);
    haptic("light");
    showToast(payload.changed ? "Origin added." : "Origin was already allowed.");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    submit.disabled = false;
  }
}

async function removeOrigin(origin, button) {
  button.disabled = true;
  try {
    const payload = await api("/api/admin/cors", {
      method: "DELETE",
      body: JSON.stringify({ origin }),
    });
    renderCors(payload.origins);
    haptic("light");
    showToast(payload.changed ? "Origin removed." : "Origin was not present.");
  } catch (error) {
    button.disabled = false;
    showToast(error.message, true);
  }
}

async function saveRuntime(event) {
  event.preventDefault();
  const runtime = {};
  const inputs = elements.runtimeFields.querySelectorAll("input[name]");
  inputs.forEach((input) => {
    if (input.dataset.kind === "bool") {
      runtime[input.name] = input.checked;
    } else {
      runtime[input.name] = input.dataset.kind === "int"
        ? Number.parseInt(input.value, 10)
        : Number.parseFloat(input.value);
    }
  });
  if (Object.values(runtime).some((value) => Number.isNaN(value))) {
    showToast("Enter a valid value for every runtime setting.", true);
    return;
  }

  elements.saveRuntimeButton.disabled = true;
  try {
    const payload = await api("/api/admin/settings", {
      method: "POST",
      body: JSON.stringify({ runtime }),
    });
    renderSettings(payload);
    haptic("medium");
    showToast(payload.changed.length ? "Runtime settings saved." : "Settings are already up to date.");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.saveRuntimeButton.disabled = false;
  }
}

function initializeTelegram() {
  if (!telegram) return;
  telegram.ready();
  telegram.expand();
  applyTelegramTheme();
  bindTelegramEvents();
  try {
    telegram.setHeaderColor("secondary_bg_color");
    telegram.setBackgroundColor("bg_color");
  } catch {
    // Older Telegram clients may not support these methods.
  }
}

function applyTelegramTheme() {
  const root = document.documentElement;
  const params = telegram?.themeParams || {};
  const colors = {
    "--tg-bg": params.bg_color,
    "--tg-secondary-bg": params.secondary_bg_color,
    "--tg-text": params.text_color,
    "--tg-hint": params.hint_color,
    "--tg-link": params.link_color,
    "--tg-button": params.button_color,
    "--tg-button-text": params.button_text_color,
  };
  Object.entries(colors).forEach(([name, value]) => {
    if (value) root.style.setProperty(name, value);
  });
  if (telegram?.colorScheme) root.dataset.telegramTheme = telegram.colorScheme;
  updateViewportVars();
}

function updateViewportVars() {
  const root = document.documentElement;
  const viewportHeight = telegram?.viewportHeight || window.innerHeight;
  const stableHeight = telegram?.viewportStableHeight || viewportHeight;
  root.style.setProperty("--tg-viewport-height", `${viewportHeight}px`);
  root.style.setProperty("--tg-viewport-stable-height", `${stableHeight}px`);
}

function bindTelegramEvents() {
  if (!telegram?.onEvent) return;
  telegram.onEvent("themeChanged", applyTelegramTheme);
  telegram.onEvent("viewportChanged", updateViewportVars);
  telegram.onEvent("fullscreenChanged", updateViewportVars);
  window.addEventListener("resize", updateViewportVars, { passive: true });
}

elements.refreshButton.addEventListener("click", () => refreshAll());
elements.maintenanceToggle.addEventListener("change", (event) => confirmMaintenance(event.target.checked));
elements.corsForm.addEventListener("submit", addOrigin);
elements.runtimeForm.addEventListener("submit", saveRuntime);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && !state.refreshing) refreshAll({ silent: true });
});

elements.profilePhoto.addEventListener("error", () => {
  elements.profilePhoto.classList.add("is-hidden");
  elements.profileFallback.classList.remove("is-hidden");
});

initializeTelegram();
refreshAll({ initial: true });
window.setInterval(() => {
  if (!document.hidden && !state.refreshing) refreshAll({ silent: true });
}, 20_000);
