"use strict";

const $ = (id) => document.getElementById(id);
const tg = window.Telegram?.WebApp;
const savedLanguage = (() => {
  try { return localStorage.getItem("bot-admin-language"); } catch (_) { return null; }
})();
const state = {
  language: savedLanguage === "km" ? "km" : "en",
  initData: "",
  refreshing: false,
  runtime: {},
  lastOk: 0,
  timer: null,
  botStatusTimer: null,
  analytics: null,
  userUsage: {},
};

const I18N = {
  en: {
    miniApp: "Telegram Mini App", controlCenter: "Bot Control Center",
    verifying: "Verifying Telegram account", secureSession: "Establishing a secure admin session…",
    authorizedAdmin: "Authorized administrator", totalUsers: "Total users", messages: "Messages handled",
    storage: "Settings storage", uptime: "Uptime", liveStatus: "Live status", systemHealth: "System health",
    botService: "Bot service", database: "Settings database", architecture: "Architecture", botMode: "Bot mode",
    quickControl: "Quick control", maintenance: "Maintenance mode",
    maintenanceHelp: "Temporarily pause normal-user features while keeping admin controls available.",
    simplifiedRuntime: "Simplified runtime", runtimeArchitecture: "Single-process architecture", activeJobs: "active", waitingJobs: "waiting", rejectedJobs: "rejected", webhookReplay: "Webhook replay",
    processHealth: "This-instance health", providers: "AI & speech providers", accessControl: "Access control",
    administrators: "Administrators", supabaseBacked: "Supabase-backed", addAdmin: "Add administrator",
    auditLog: "Audit log", networkPolicy: "Network policy", corsOrigins: "Allowed CORS origins",
    exactOrigins: "Exact origins only", addOrigin: "Add origin", noOrigins: "No browser origins are currently allowed.",
    supabaseControls: "Supabase-backed controls", runtimeSettings: "Runtime settings", validatedLimits: "Validated limits",
    settingsApply: "Changes apply immediately and persist across restarts.", saveSettings: "Save settings",
    operational: "Operational", degraded: "Degraded", healthy: "Healthy", unavailable: "Unavailable",
    connected: "Connected", memory: "Memory fallback", active: "Active", paused: "Paused",
    loading: "Starting…", standby: "Standby", stopping: "Stopping…", error: "Error",
    single: "Single process", saved: "Saved", removed: "Removed", added: "Added", reset: "Reset", remove: "Remove",
    noAdmins: "No administrators found.", noAudit: "No audit entries yet.", confirmRemove: "Remove this administrator?",
    refreshFailed: "Some dashboard sections could not refresh.", requestTimeout: "Request timed out. Please retry.",
    online: "Online", offline: "Offline", maintenanceActive: "Normal-user features are paused.",
    normalActive: "Normal bot service is active.", telegramVerified: "Telegram verified",
  },
  km: {
    miniApp: "Telegram Mini App", controlCenter: "មជ្ឈមណ្ឌលគ្រប់គ្រងបូត",
    verifying: "កំពុងផ្ទៀងផ្ទាត់គណនី Telegram", secureSession: "កំពុងបង្កើតសម័យ Admin ដែលមានសុវត្ថិភាព…",
    authorizedAdmin: "អ្នកគ្រប់គ្រងដែលបានផ្ទៀងផ្ទាត់", totalUsers: "អ្នកប្រើសរុប", messages: "សារដែលបានដំណើរការ",
    storage: "ទីតាំងរក្សាទុក Settings", uptime: "រយៈពេលដំណើរការ", liveStatus: "ស្ថានភាពបច្ចុប្បន្ន", systemHealth: "សុខភាពប្រព័ន្ធ",
    botService: "សេវាបូត", database: "មូលដ្ឋានទិន្នន័យ Settings", architecture: "ស្ថាបត្យកម្ម", botMode: "របៀបបូត",
    quickControl: "គ្រប់គ្រងរហ័ស", maintenance: "របៀបថែទាំ",
    maintenanceHelp: "ផ្អាកមុខងារអ្នកប្រើធម្មតាបណ្តោះអាសន្ន ខណៈ Admin នៅតែអាចគ្រប់គ្រងបាន។",
    simplifiedRuntime: "Runtime សាមញ្ញ", runtimeArchitecture: "ស្ថាបត្យកម្មដំណើរការតែមួយ", activeJobs: "កំពុងដំណើរការ", waitingJobs: "កំពុងរង់ចាំ", rejectedJobs: "បានបដិសេធ", webhookReplay: "ការពារ Webhook ស្ទួន",
    processHealth: "សុខភាព Process នេះ", providers: "AI និង Speech Providers", accessControl: "ការគ្រប់គ្រងសិទ្ធិ",
    administrators: "អ្នកគ្រប់គ្រង", supabaseBacked: "រក្សាទុកក្នុង Supabase", addAdmin: "បន្ថែម Admin",
    auditLog: "ប្រវត្តិសកម្មភាព", networkPolicy: "គោលការណ៍បណ្តាញ", corsOrigins: "CORS origins ដែលអនុញ្ញាត",
    exactOrigins: "អនុញ្ញាត origin ជាក់លាក់", addOrigin: "បន្ថែម origin", noOrigins: "មិនទាន់មាន browser origin ដែលបានអនុញ្ញាត។",
    supabaseControls: "Settings ក្នុង Supabase", runtimeSettings: "ការកំណត់ Runtime", validatedLimits: "តម្លៃមានដែនកំណត់",
    settingsApply: "ការផ្លាស់ប្តូរអនុវត្តភ្លាមៗ និងរក្សាទុកក្រោយ restart។", saveSettings: "រក្សាទុក Settings",
    operational: "ដំណើរការល្អ", degraded: "ដំណើរការមិនពេញលេញ", healthy: "ល្អ", unavailable: "មិនអាចប្រើបាន",
    connected: "បានភ្ជាប់", memory: "ប្រើ Memory បម្រុង", active: "សកម្ម", paused: "បានផ្អាក",
    single: "ដំណើរការតែមួយ", saved: "បានរក្សាទុក", removed: "បានដកចេញ", added: "បានបន្ថែម", reset: "កំណត់ឡើងវិញ", remove: "ដកចេញ",
    noAdmins: "មិនមានអ្នកគ្រប់គ្រងទេ។", noAudit: "មិនទាន់មានប្រវត្តិសកម្មភាពទេ។", confirmRemove: "ដកអ្នកគ្រប់គ្រងនេះមែនទេ?",
    refreshFailed: "ផ្នែកខ្លះមិនអាចធ្វើបច្ចុប្បន្នភាពបាន។", requestTimeout: "សំណើអស់ពេល។ សូមព្យាយាមម្តងទៀត។",
    online: "អនឡាញ", offline: "ក្រៅបណ្តាញ", maintenanceActive: "មុខងារអ្នកប្រើធម្មតាត្រូវបានផ្អាក។",
    normalActive: "សេវាបូតកំពុងដំណើរការធម្មតា។", telegramVerified: "បានផ្ទៀងផ្ទាត់ដោយ Telegram",
  },
};
const t = (key) => I18N[state.language]?.[key] || I18N.en[key] || key;

function applyTranslations() {
  document.documentElement.lang = state.language === "km" ? "km" : "en";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const value = t(el.dataset.i18n);
    if (value) el.textContent = value;
  });
  $("languageButton").textContent = state.language === "en" ? "ខ្មែរ" : "EN";
  $("refreshButton").setAttribute("aria-label", state.language === "km" ? "ធ្វើបច្ចុប្បន្នភាព" : "Refresh dashboard");
}

function haptic(type = "selection") {
  try {
    if (type === "success") tg?.HapticFeedback?.notificationOccurred("success");
    else if (type === "error") tg?.HapticFeedback?.notificationOccurred("error");
    else tg?.HapticFeedback?.selectionChanged();
  } catch (_) {}
}

function initTelegram() {
  if (!tg) return;
  tg.ready();
  tg.expand();
  state.initData = tg.initData || "";
  try {
    tg.setHeaderColor("secondary_bg_color");
    tg.setBackgroundColor("bg_color");
    tg.enableClosingConfirmation?.();
  } catch (_) {}
}

function headers() {
  const h = { "Content-Type": "application/json", Accept: "application/json" };
  if (state.initData) {
    h["X-Telegram-Init-Data"] = state.initData;
    h.Authorization = `Bearer ${state.initData}`;
  }
  return h;
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(path, {
      credentials: "same-origin",
      cache: "no-store",
      ...options,
      headers: { ...headers(), ...(options.headers || {}) },
      signal: controller.signal,
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
    return body;
  } catch (error) {
    if (error?.name === "AbortError") throw new Error(t("requestTimeout"));
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function showToast(message, error = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("error", error);
  el.classList.add("show");
  haptic(error ? "error" : "success");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => el.classList.remove("show"), 3200);
}

function setHealth(el, text, kind = "") {
  el.textContent = text;
  el.classList.toggle("down", kind === "down");
  el.classList.toggle("warn", kind === "warn");
}
function compact(value) {
  return new Intl.NumberFormat(state.language === "km" ? "km-KH" : "en", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(Number(value || 0));
}
function setConnection(ok) {
  const el = $("connectionBadge");
  const offline = !navigator.onLine;
  el.textContent = ok ? t("online") : (offline ? t("offline") : t("degraded"));
  el.classList.toggle("down", !ok);
  el.classList.toggle("warn", !ok && !offline);
}

function renderProfile(payload) {
  const user = payload.user || {};
  const name = [user.first_name, user.last_name].filter(Boolean).join(" ") || "Administrator";
  $("profileName").textContent = name;
  $("profileHandle").textContent = user.username
    ? `@${user.username} · ${payload.auth_method}`
    : `${payload.auth_method || t("telegramVerified")} · ID ${user.id || "—"}`;
  $("profileFallback").textContent = (name[0] || "A").toUpperCase();
  if (user.photo_url) {
    $("profilePhoto").src = user.photo_url;
    $("profilePhoto").classList.remove("is-hidden");
    $("profileFallback").classList.add("is-hidden");
  } else {
    $("profilePhoto").removeAttribute("src");
    $("profilePhoto").classList.add("is-hidden");
    $("profileFallback").classList.remove("is-hidden");
  }
}

function renderStats(payload) {
  const bot = payload.bot || {};
  const usage = payload.usage || {};
  const db = payload.database || {};
  const storage = payload.storage || {};
  $("totalUsers").textContent = compact(usage.total_users);
  $("messageCount").textContent = compact(usage.message_count);
  $("uptime").textContent = bot.uptime || "—";
  $("storageBackend").textContent = storage.persistent ? "Supabase" : "Memory";
  const botStatus = bot.status || (bot.active ? "online" : "offline");
  const botStatusLabels = {
    online: t("active"), loading: t("loading"), standby: t("standby"),
    stopping: t("stopping"), offline: t("unavailable"), error: t("error"),
  };
  const botHealthKind = botStatus === "online" ? "" : (["loading", "standby", "stopping"].includes(botStatus) ? "warn" : "down");
  setHealth($("botHealth"), botStatusLabels[botStatus] || botStatus, botHealthKind);
  setHealth($("databaseHealth"), db.ok ? t("connected") : t("memory"), db.ok ? "" : "warn");
  setHealth($("architectureHealth"), t("single"));
  $("botMode").textContent = bot.mode || "—";
  $("maintenanceToggle").checked = Boolean(bot.maintenance_mode);
  $("maintenanceNotice").textContent = bot.maintenance_mode ? t("maintenanceActive") : t("normalActive");
  $("maintenanceNotice").classList.toggle("warning", Boolean(bot.maintenance_mode));
  const healthy = Boolean(bot.active && (db.ok || db.memory_fallback));
  $("systemBadge").classList.toggle("down", !healthy);
  $("systemBadge").classList.toggle("warn", !healthy && bot.active);
  $("systemBadge").lastElementChild.textContent = healthy ? t("operational") : t("degraded");
  $("lastUpdated").textContent = new Date(payload.generated_at || Date.now()).toLocaleTimeString(
    state.language === "km" ? "km-KH" : undefined,
  );
  clearTimeout(state.botStatusTimer);
  state.botStatusTimer = null;
  if (["loading", "stopping"].includes(botStatus)) {
    state.botStatusTimer = setTimeout(() => refreshAll({ silent: true }), 2500);
  }
}

function renderAnalytics(payload, usersPayload = {}) {
  state.analytics = payload || {};
  state.userUsage = usersPayload || {};
  const period = $("analyticsPeriod")?.value || "daily";
  const rows = payload?.[period] || [];
  const canvas = $("usageChart");
  if (canvas) {
    const context = canvas.getContext("2d");
    const width = canvas.clientWidth || 520;
    const height = 170;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const values = rows.map((row) => Number(row.requests || 0));
    const max = Math.max(1, ...values);
    const gap = 3;
    const barWidth = Math.max(2, (width - gap * Math.max(0, values.length - 1)) / Math.max(1, values.length));
    context.fillStyle = "#5b7cfa";
    values.forEach((value, index) => {
      const barHeight = Math.max(2, (value / max) * (height - 28));
      const x = index * (barWidth + gap);
      context.fillRect(x, height - barHeight - 18, barWidth, barHeight);
    });
    context.fillStyle = "#8290a8";
    context.font = "11px system-ui";
    context.fillText(`0`, 2, height - 3);
    context.fillText(`${max}`, 2, 12);
  }
  const total = rows.reduce((sum, row) => sum + Number(row.requests || 0), 0);
  const audioMs = rows.reduce((sum, row) => sum + Number(row.audio_generation_ms || 0), 0);
  $("usageSummary").textContent = `${compact(total)} requests · ${(audioMs / 1000).toFixed(1)}s audio generation · source: ${payload?.source || "process"}`;
  renderUserUsage(usersPayload);
}

function renderUserUsage(payload = {}) {
  const root = $("userUsageList");
  if (!root) return;
  root.replaceChildren();
  const users = payload.users || [];
  if (!users.length) {
    root.append(simpleRow("No usage recorded yet."));
    return;
  }
  users.slice(0, 12).forEach((user) => {
    const label = `${user.username ? `@${user.username} · ` : ""}${user.user_id}`;
    const detail = `${compact(user.request_count)} requests · ${(Number(user.audio_generation_ms || 0) / 1000).toFixed(1)}s audio`;
    root.append(simpleRow(`${label} — ${detail}`));
  });
}

function renderScheduleFailures(payload = {}) {
  const root = $("scheduleFailures");
  if (!root) return;
  root.replaceChildren();
  const failures = payload.failures || [];
  if (!failures.length) {
    root.append(simpleRow("No failed schedules."));
    return;
  }
  failures.forEach((row) => {
    const detail = `${row.id} · ${row.error_msg || "delivery failed"}`;
    root.append(simpleRow(detail, "Retry", async () => {
      try {
        await api(`/api/admin/schedules/${Number(row.id)}/retry`, { method: "POST" });
        showToast("Retry queued");
        await refreshV2();
      } catch (error) { showToast(error.message, true); }
    }));
  });
}

function renderDailySchedules(payload = {}) {
  const root = $("dailyScheduleList");
  if (!root) return;
  root.replaceChildren();
  const schedules = payload.schedules || [];
  if (!schedules.length) {
    root.append(simpleRow("No daily broadcast configured."));
    return;
  }
  schedules.forEach((row) => {
    const time = row.time || "--:--";
    const status = row.status || "pending";
    const content = row.content || "Daily message";
    root.append(simpleRow(`${time} · ${content} · ${status}`, "Cancel", async () => {
      if (!window.confirm("Cancel this daily broadcast?")) return;
      try {
        await api(`/api/admin/schedules/daily/${Number(row.id)}`, { method: "DELETE" });
        showToast("Daily broadcast cancelled");
        await refreshV2();
      } catch (error) { showToast(error.message, true); }
    }));
  });
}

function renderRuntime(payload) {
  const store = payload.settings_store || {};
  const backend = store.backend === "supabase" ? "Supabase" : "Memory";
  const replay = payload.webhook_replay || {};
  $("runtimeSummary").textContent = `${t("single")} · Settings: ${backend} · No Redis · No dedicated worker · ${t("webhookReplay")}: ${compact(replay.entries || 0)}`;

  const root = $("workloadGrid");
  root.replaceChildren();
  const workloads = payload.telegram_workloads || {};
  [
    ["OCR", workloads.ocr || {}],
    ["Transcribe", workloads.transcribe || {}],
    ["Audio", workloads.audio || {}],
  ].forEach(([label, item]) => {
    const card = document.createElement("article");
    card.className = "workload-card";
    const heading = document.createElement("div");
    heading.className = "workload-heading";
    const title = document.createElement("strong");
    title.textContent = label;
    const capacity = document.createElement("span");
    capacity.className = "secure-label";
    capacity.textContent = `max ${Number(item.capacity || 0)}`;
    heading.append(title, capacity);
    const detail = document.createElement("small");
    detail.className = "muted";
    detail.textContent = `${Number(item.in_use || 0)} ${t("activeJobs")} · ${Number(item.waiting || 0)} ${t("waitingJobs")} · ${Number(item.rejected || 0)} ${t("rejectedJobs")}`;
    card.append(heading, detail);
    root.append(card);
  });
}

function renderSettings(payload) {
  state.runtime = payload.runtime || {};
  if ($("maintenanceMessage")) $("maintenanceMessage").value = payload.maintenance_message || "";
  const root = $("runtimeFields");
  root.replaceChildren();
  Object.entries(state.runtime).forEach(([key, spec]) => {
    const wrap = document.createElement("label");
    wrap.className = "setting-field";
    const copy = document.createElement("span");
    copy.className = "setting-copy";
    const title = document.createElement("strong");
    title.textContent = spec.label || key;
    const help = document.createElement("small");
    help.className = "muted";
    help.textContent = spec.help || key;
    copy.append(title, help);
    const control = document.createElement("span");
    control.className = "setting-control";
    let input;
    if (spec.kind === "bool") {
      const switchLabel = document.createElement("span");
      switchLabel.className = "switch";
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(spec.value);
      const track = document.createElement("span");
      track.className = "switch-track";
      switchLabel.append(input, track);
      control.append(switchLabel);
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.value = spec.value ?? "";
      if (spec.min != null) input.min = spec.min;
      if (spec.max != null) input.max = spec.max;
      input.step = String(spec.kind).includes("float") ? "0.1" : "1";
      input.className = "text-input";
      control.append(input);
    }
    input.dataset.runtimeKey = key;
    input.setAttribute("aria-label", spec.label || key);
    wrap.append(copy, control);
    root.append(wrap);
  });
}

function renderCors(origins = []) {
  const root = $("corsList");
  root.replaceChildren();
  origins.forEach((origin) => {
    const chip = document.createElement("span");
    chip.className = "origin-chip";
    const text = document.createElement("span");
    text.textContent = origin;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button small danger";
    button.textContent = t("remove");
    button.setAttribute("aria-label", `${t("remove")} ${origin}`);
    button.addEventListener("click", () => removeCors(origin));
    chip.append(text, button);
    root.append(chip);
  });
  $("corsEmpty").classList.toggle("is-hidden", origins.length > 0);
}

function renderProviders(payload) {
  const root = $("providersList");
  root.replaceChildren();
  $("providerScope").textContent = `Process-local · ${payload.count || 0}`;
  Object.entries(payload.providers || {}).forEach(([name, p]) => {
    const card = document.createElement("article");
    card.className = "provider-card";
    const heading = document.createElement("div");
    heading.className = "provider-heading";
    const strong = document.createElement("strong");
    strong.textContent = name;
    const badge = document.createElement("span");
    badge.className = `health-value ${p.available ? "" : "down"}`;
    badge.textContent = p.available ? t("healthy") : `${Math.ceil(p.cooldown_remaining_seconds || 0)}s`;
    heading.append(strong, badge);
    const meta = document.createElement("p");
    meta.className = "muted";
    meta.textContent = `Score ${p.health_score ?? "—"} · ${p.latency_ewma_ms ?? "—"} ms · ${p.successes || 0}/${p.failures || 0}`;
    const reset = document.createElement("button");
    reset.className = "button small";
    reset.textContent = t("reset");
    reset.onclick = async () => {
      reset.disabled = true;
      try {
        await api("/api/admin/runtime/providers/reset", { method: "POST", body: JSON.stringify({ provider: name }) });
        await refreshProviders();
        showToast(t("saved"));
      } catch (error) {
        showToast(error.message, true);
      } finally {
        reset.disabled = false;
      }
    };
    card.append(heading, meta, reset);
    root.append(card);
  });
}

function simpleRow(text, actionText, action) {
  const row = document.createElement("div");
  row.className = "data-row";
  const label = document.createElement("span");
  label.className = "data-copy";
  label.textContent = text;
  row.append(label);
  if (action) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button small danger";
    button.textContent = actionText;
    button.onclick = action;
    row.append(button);
  }
  return row;
}

function renderAdmins(payload, audit) {
  const root = $("adminList");
  root.replaceChildren();
  const ids = payload.administrators || [];
  if (!ids.length) root.append(simpleRow(t("noAdmins")));
  ids.forEach((id) => root.append(simpleRow(`Telegram ID ${id}`, t("remove"), () => removeAdmin(id))));
  const auditRoot = $("auditList");
  auditRoot.replaceChildren();
  const entries = audit.entries || [];
  if (!entries.length) auditRoot.append(simpleRow(t("noAudit")));
  entries.slice(0, 20).forEach((entry) => {
    const when = entry.timestamp ? new Date(entry.timestamp * 1000).toLocaleString() : "—";
    auditRoot.append(simpleRow(`${entry.action} · ${entry.target_id} · by ${entry.actor_id} · ${when}`));
  });
}

async function refreshProviders() {
  renderProviders(await api("/api/admin/providers/health"));
}
async function refreshAdmins() {
  const [admins, audit] = await Promise.all([
    api("/api/admin/administrators"),
    api("/api/admin/administrators/audit?limit=50"),
  ]);
  renderAdmins(admins, audit);
}

async function refreshAll({ silent = false } = {}) {
  if (state.refreshing) return;
  state.refreshing = true;
  $("refreshButton").disabled = true;
  try {
    const calls = [
      api("/api/admin/me"), api("/api/admin/stats"), api("/api/admin/settings"), api("/api/admin/cors"),
      api("/api/admin/runtime/status"), api("/api/admin/providers/health"), api("/api/admin/administrators"),
      api("/api/admin/administrators/audit?limit=50"), api("/api/admin/analytics?days=30"),
      api("/api/admin/usage/users?limit=50"), api("/api/admin/schedules/failures?limit=50"),
      api("/api/admin/schedules/daily"),
    ];
    const results = await Promise.allSettled(calls);
    const good = results.filter((item) => item.status === "fulfilled").length;
    if (results[0].status === "fulfilled") renderProfile(results[0].value);
    if (results[1].status === "fulfilled") renderStats(results[1].value);
    if (results[2].status === "fulfilled") renderSettings(results[2].value);
    if (results[3].status === "fulfilled") renderCors(results[3].value.origins || []);
    if (results[4].status === "fulfilled") renderRuntime(results[4].value);
    if (results[5].status === "fulfilled") renderProviders(results[5].value);
    if (results[6].status === "fulfilled" && results[7].status === "fulfilled") {
      renderAdmins(results[6].value, results[7].value);
    }
    if (results[8].status === "fulfilled" || results[9].status === "fulfilled") {
      renderAnalytics(results[8].status === "fulfilled" ? results[8].value : {}, results[9].status === "fulfilled" ? results[9].value : {});
    }
    if (results[10].status === "fulfilled") renderScheduleFailures(results[10].value);
    if (results[11].status === "fulfilled") renderDailySchedules(results[11].value);
    if (good === 0) throw (results[0].reason || new Error("Dashboard unavailable"));
    state.lastOk = Date.now();
    setConnection(good === results.length);
    $("authState").classList.add("is-hidden");
    $("authState").classList.remove("error");
    $("dashboard").classList.remove("is-hidden");
    if (good < results.length && !silent) showToast(t("refreshFailed"), true);
  } catch (error) {
    setConnection(false);
    if (!state.lastOk) {
      $("authState").classList.add("error");
      $("authDetail").textContent = error.message;
    }
    if (!silent) showToast(error.message, true);
  } finally {
    state.refreshing = false;
    $("refreshButton").disabled = false;
  }
}

async function confirmMutation(action, userId) {
  return api("/api/admin/administrators/confirmations", {
    method: "POST",
    body: JSON.stringify({ action, user_id: Number(userId) }),
  });
}
async function addAdmin(userId) {
  const confirmation = await confirmMutation("add", userId);
  await api("/api/admin/administrators", {
    method: "POST",
    body: JSON.stringify({ user_id: Number(userId), confirmation_token: confirmation.confirmation_token }),
  });
  showToast(t("added"));
  await refreshAdmins();
}
async function removeAdmin(userId) {
  if (!window.confirm(t("confirmRemove"))) return;
  try {
    const confirmation = await confirmMutation("remove", userId);
    await api("/api/admin/administrators", {
      method: "DELETE",
      body: JSON.stringify({ user_id: Number(userId), confirmation_token: confirmation.confirmation_token }),
    });
    showToast(t("removed"));
    await refreshAdmins();
  } catch (error) {
    showToast(error.message, true);
  }
}
async function removeCors(origin) {
  try {
    const payload = await api("/api/admin/cors", { method: "DELETE", body: JSON.stringify({ origin }) });
    renderCors(payload.origins || []);
    showToast(t("removed"));
  } catch (error) {
    showToast(error.message, true);
  }
}

async function refreshV2() {
  const days = Number($("analyticsRange")?.value || 30);
  const [analytics, users, failures, daily] = await Promise.all([
    api(`/api/admin/analytics?days=${days}`),
    api("/api/admin/usage/users?limit=50"),
    api("/api/admin/schedules/failures?limit=50"),
    api("/api/admin/schedules/daily"),
  ]);
  renderAnalytics(analytics, users);
  renderScheduleFailures(failures);
  renderDailySchedules(daily);
}

async function downloadAdminFile(path, filename) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store", headers: headers() });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

async function downloadBackup() {
  const payload = await api("/api/admin/backup");
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "bot-settings-backup.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

$("refreshButton").onclick = () => refreshAll();
$("languageButton").onclick = () => {
  state.language = state.language === "en" ? "km" : "en";
  try { localStorage.setItem("bot-admin-language", state.language); } catch (_) {}
  applyTranslations();
  haptic();
  refreshAll({ silent: true });
};
$("maintenanceToggle").onchange = async (event) => {
  event.target.disabled = true;
  try {
    await api("/api/admin/settings", { method: "POST", body: JSON.stringify({ maintenance_mode: event.target.checked }) });
    await refreshAll({ silent: true });
    showToast(t("saved"));
  } catch (error) {
    event.target.checked = !event.target.checked;
    showToast(error.message, true);
  } finally {
    event.target.disabled = false;
  }
};
$("saveMaintenanceMessage").onclick = async () => {
  const button = $("saveMaintenanceMessage");
  button.disabled = true;
  try {
    await api("/api/admin/settings", { method: "POST", body: JSON.stringify({ maintenance_message: $("maintenanceMessage").value }) });
    showToast(t("saved"));
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; }
};
$("analyticsRange").onchange = async () => {
  try { await refreshV2(); } catch (error) { showToast(error.message, true); }
};
$("analyticsPeriod").onchange = () => renderAnalytics(state.analytics || {}, state.userUsage || {});
$("clearCacheButton").onclick = async () => {
  const button = $("clearCacheButton");
  button.disabled = true;
  try {
    const payload = await api("/api/admin/cache/clear", { method: "POST", body: JSON.stringify({}) });
    showToast(payload.message || t("saved"));
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; }
};
$("downloadBackupButton").onclick = async () => {
  try { await downloadBackup(); showToast("Backup downloaded"); } catch (error) { showToast(error.message, true); }
};
$("restoreBackupButton").onclick = async () => {
  try {
    const value = JSON.parse($("backupPayload").value || "{}");
    await api("/api/admin/backup/restore", { method: "POST", body: JSON.stringify({ settings: value.settings || {}, runtime: value.runtime || {} }) });
    showToast("Backup restored");
    await refreshAll({ silent: true });
  } catch (error) { showToast(error.message, true); }
};
document.querySelectorAll("[data-export]").forEach((button) => button.addEventListener("click", async () => {
  try {
    const dataset = button.dataset.export;
    const format = button.dataset.format;
    await downloadAdminFile(`/api/admin/export/${dataset}?format=${format}`, `bot-${dataset}.${format}`);
    showToast("Export downloaded");
  } catch (error) { showToast(error.message, true); }
}));
$("broadcastTestButton").onclick = async () => {
  const button = $("broadcastTestButton");
  button.disabled = true;
  try {
    await api("/api/admin/broadcast/test", { method: "POST", body: JSON.stringify({
      text: $("broadcastTestText").value,
      photo_file_id: $("broadcastPhotoId").value || null,
      parse_mode: $("broadcastParseMode").value,
    }) });
    showToast("Test sent to your Telegram account");
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; }
};
$("dailyBroadcastForm").onsubmit = async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const payload = await api("/api/admin/schedules/daily", {
      method: "POST",
      body: JSON.stringify({
        time: $("dailyBroadcastTime").value,
        text: $("dailyBroadcastText").value,
        photo_file_id: $("dailyBroadcastPhotoId").value || null,
        caption: $("dailyBroadcastCaption").value,
        parse_mode: $("dailyBroadcastParseMode").value,
      }),
    });
    showToast(`Daily broadcast scheduled for ${payload.schedule?.broadcast_at || "the next run"}`);
    $("dailyBroadcastText").value = "";
    $("dailyBroadcastPhotoId").value = "";
    $("dailyBroadcastCaption").value = "";
    await refreshV2();
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; }
};
$("runtimeForm").onsubmit = async (event) => {
  event.preventDefault();
  const runtime = {};
  document.querySelectorAll("[data-runtime-key]").forEach((input) => {
    runtime[input.dataset.runtimeKey] = input.type === "checkbox" ? input.checked : Number(input.value);
  });
  $("saveRuntimeButton").disabled = true;
  try {
    await api("/api/admin/settings", { method: "POST", body: JSON.stringify({ runtime }) });
    showToast(t("saved"));
    await refreshAll({ silent: true });
  } catch (error) {
    showToast(error.message, true);
  } finally {
    $("saveRuntimeButton").disabled = false;
  }
};
$("corsForm").onsubmit = async (event) => {
  event.preventDefault();
  const input = $("corsOrigin");
  try {
    const payload = await api("/api/admin/cors", { method: "POST", body: JSON.stringify({ origin: input.value }) });
    input.value = "";
    renderCors(payload.origins || []);
    showToast(t("added"));
  } catch (error) {
    showToast(error.message, true);
  }
};
$("adminForm").onsubmit = async (event) => {
  event.preventDefault();
  const input = $("adminUserId");
  try {
    await addAdmin(input.value);
    input.value = "";
  } catch (error) {
    showToast(error.message, true);
  }
};
document.querySelectorAll("[data-nav-target]").forEach((item) => item.addEventListener("click", () => {
  document.querySelectorAll("[data-nav-target]").forEach((candidate) => candidate.classList.toggle("active", candidate === item));
}));
window.addEventListener("online", () => refreshAll({ silent: true }));
window.addEventListener("offline", () => setConnection(false));
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && Date.now() - state.lastOk > 30000) refreshAll({ silent: true });
});

initTelegram();
applyTranslations();
refreshAll();
state.timer = setInterval(() => { if (!document.hidden) refreshAll({ silent: true }); }, 30000);
