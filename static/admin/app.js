"use strict";

const telegram = window.Telegram?.WebApp ?? null;
const state = {
  language: window.localStorage.getItem("admin-language") || "en",
  settings: null,
  corsOrigins: [],
  refreshing: false,
  toastTimer: null,
  jobCursor: null,
  selectedJobs: new Set(),
};

const translations = {
  en: {
    miniApp: "Telegram Mini App", controlCenter: "Bot Control Center", verifying: "Verifying Telegram account",
    secureSession: "Establishing a secure admin session…", authorizedAdmin: "Authorized administrator",
    checkingSystem: "Checking system", totalUsers: "Total users", messages: "Messages handled",
    queuedJobs: "Queued jobs", deadJobs: "Dead jobs", redisLatency: "Redis latency", uptime: "Uptime",
    liveStatus: "Live status", systemHealth: "System health", botService: "Bot service", redisCache: "Redis cache",
    database: "Settings database", workers: "Durable workers", botMode: "Bot mode", quickControl: "Quick control",
    maintenance: "Maintenance mode", maintenanceHelp: "Temporarily pause normal-user features while keeping admin controls available.",
    durableRuntime: "Durable runtime", jobQueue: "Job queue", refresh: "Refresh", noJobs: "No jobs in this state.",
    retrySelected: "Retry selected", loadMore: "Load more", processHealth: "This-instance health",
    providers: "AI & speech providers", accessControl: "Access control", administrators: "Administrators",
    confirmationRequired: "Confirmation required", addAdmin: "Add administrator", auditLog: "Audit log",
    networkPolicy: "Network policy", corsOrigins: "Allowed CORS origins", exactOrigins: "Exact origins only",
    addOrigin: "Add origin", noOrigins: "No browser origins are currently allowed.", redisControls: "Redis-backed controls",
    runtimeSettings: "Runtime settings", validatedLimits: "Validated limits", settingsApply: "Changes apply immediately and persist across restarts.",
    saveSettings: "Save settings", overview: "Overview", health: "Health", jobs: "Jobs", providersShort: "Providers",
    admins: "Admins", settings: "Settings", operational: "Operational", starting: "Starting", healthy: "Healthy",
    unavailable: "Unavailable", connected: "Connected", memoryFallback: "Memory fallback", active: "Active",
    paused: "Paused", drainWorkers: "Drain workers", resumeWorkers: "Resume workers", accepting: "Accepting jobs", drained: "Drained", processLocal: "Process-local", reset: "Reset", retry: "Retry", cancel: "Cancel", remove: "Remove",
  },
  km: {
    miniApp: "Telegram Mini App", controlCenter: "មជ្ឈមណ្ឌលគ្រប់គ្រងបូត", verifying: "កំពុងផ្ទៀងផ្ទាត់គណនី Telegram",
    secureSession: "កំពុងបង្កើតសម័យ Admin ដែលមានសុវត្ថិភាព…", authorizedAdmin: "អ្នកគ្រប់គ្រងដែលបានអនុញ្ញាត",
    checkingSystem: "កំពុងពិនិត្យប្រព័ន្ធ", totalUsers: "អ្នកប្រើប្រាស់សរុប", messages: "សារដែលបានដំណើរការ",
    queuedJobs: "ការងារកំពុងរង់ចាំ", deadJobs: "ការងារបរាជ័យ", redisLatency: "ល្បឿនឆ្លើយតប Redis", uptime: "រយៈពេលដំណើរការ",
    liveStatus: "ស្ថានភាពផ្ទាល់", systemHealth: "សុខភាពប្រព័ន្ធ", botService: "សេវាបូត", redisCache: "Redis cache",
    database: "មូលដ្ឋានទិន្នន័យការកំណត់", workers: "Durable workers", botMode: "របៀបបូត", quickControl: "ការគ្រប់គ្រងរហ័ស",
    maintenance: "របៀបថែទាំ", maintenanceHelp: "ផ្អាកមុខងារសម្រាប់អ្នកប្រើធម្មតាជាបណ្ដោះអាសន្ន ខណៈ Admin នៅតែអាចគ្រប់គ្រងបាន។",
    durableRuntime: "Durable runtime", jobQueue: "ជួរការងារ", refresh: "ផ្ទុកឡើងវិញ", noJobs: "មិនមានការងារនៅក្នុងស្ថានភាពនេះទេ។",
    retrySelected: "សាកឡើងវិញដែលបានជ្រើស", loadMore: "បង្ហាញបន្ថែម", processHealth: "សុខភាព instance នេះ",
    providers: "AI និង Speech providers", accessControl: "ការគ្រប់គ្រងសិទ្ធិ", administrators: "អ្នកគ្រប់គ្រង",
    confirmationRequired: "ត្រូវការការបញ្ជាក់", addAdmin: "បន្ថែម Admin", auditLog: "កំណត់ហេតុសកម្មភាព",
    networkPolicy: "គោលការណ៍បណ្ដាញ", corsOrigins: "CORS origins ដែលបានអនុញ្ញាត", exactOrigins: "ទទួលតែ origin ពេញលេញ",
    addOrigin: "បន្ថែម origin", noOrigins: "មិនមាន browser origin ដែលបានអនុញ្ញាតទេ។", redisControls: "ការគ្រប់គ្រងតាម Redis",
    runtimeSettings: "ការកំណត់ Runtime", validatedLimits: "ដែនកំណត់ដែលបានផ្ទៀងផ្ទាត់", settingsApply: "ការផ្លាស់ប្ដូរអនុវត្តភ្លាម និងរក្សាទុកក្រោយ restart។",
    saveSettings: "រក្សាទុក", overview: "ទិដ្ឋភាព", health: "សុខភាព", jobs: "ការងារ", providersShort: "Providers",
    admins: "Admin", settings: "ការកំណត់", operational: "ដំណើរការល្អ", starting: "កំពុងចាប់ផ្ដើម", healthy: "ល្អ",
    unavailable: "មិនអាចប្រើបាន", connected: "បានភ្ជាប់", memoryFallback: "ប្រើ Memory បម្រុង", active: "សកម្ម",
    paused: "បានផ្អាក", drainWorkers: "ផ្អាកទទួលការងារ", resumeWorkers: "បន្តទទួលការងារ", accepting: "កំពុងទទួលការងារ", drained: "បានផ្អាកទទួលការងារ", processLocal: "តាម process", reset: "កំណត់ឡើងវិញ", retry: "សាកឡើងវិញ", cancel: "បោះបង់", remove: "ដកចេញ",
  },
};

const $ = (id) => document.getElementById(id);
const elements = {
  authState: $("authState"), dashboard: $("dashboard"), refreshButton: $("refreshButton"), languageButton: $("languageButton"),
  profilePhoto: $("profilePhoto"), profileFallback: $("profileFallback"), profileName: $("profileName"), profileHandle: $("profileHandle"),
  systemBadge: $("systemBadge"), totalUsers: $("totalUsers"), messageCount: $("messageCount"), queuedJobs: $("queuedJobs"),
  deadJobs: $("deadJobs"), redisLatency: $("redisLatency"), uptime: $("uptime"), lastUpdated: $("lastUpdated"),
  botHealth: $("botHealth"), redisHealth: $("redisHealth"), databaseHealth: $("databaseHealth"), workerHealth: $("workerHealth"),
  botMode: $("botMode"), maintenanceToggle: $("maintenanceToggle"), maintenanceNotice: $("maintenanceNotice"),
  corsForm: $("corsForm"), corsOrigin: $("corsOrigin"), corsList: $("corsList"), corsEmpty: $("corsEmpty"),
  runtimeForm: $("runtimeForm"), runtimeFields: $("runtimeFields"), saveRuntimeButton: $("saveRuntimeButton"),
  jobState: $("jobState"), refreshJobsButton: $("refreshJobsButton"), jobsList: $("jobsList"), jobsEmpty: $("jobsEmpty"),
  jobSummary: $("jobSummary"), drainWorkersButton: $("drainWorkersButton"), resumeWorkersButton: $("resumeWorkersButton"), retrySelectedButton: $("retrySelectedButton"), loadMoreJobsButton: $("loadMoreJobsButton"),
  providersList: $("providersList"), providerScope: $("providerScope"), adminForm: $("adminForm"), adminUserId: $("adminUserId"),
  adminList: $("adminList"), auditList: $("auditList"), toast: $("toast"),
};

function t(key) { return translations[state.language]?.[key] || translations.en[key] || key; }
function applyLanguage() {
  document.documentElement.lang = state.language;
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  elements.languageButton.textContent = state.language === "en" ? "ខ្មែរ" : "English";
  window.localStorage.setItem("admin-language", state.language);
}

function haptic(type = "light") { try { telegram?.HapticFeedback?.impactOccurred(type); } catch {} }
function showToast(message, isError = false) {
  window.clearTimeout(state.toastTimer);
  elements.toast.textContent = String(message);
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("show");
  state.toastTimer = window.setTimeout(() => elements.toast.classList.remove("show"), 3200);
}
function compactNumber(value) { return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(Number(value) || 0); }
function formatDate(value) { return value ? new Date(Number(value) * 1000).toLocaleString() : "—"; }

async function api(path, options = {}) {
  const initData = String(window.Telegram?.WebApp?.initData || "").trim();
  if (!initData) throw new Error("Open this dashboard from the bot inside Telegram.");
  const headers = new Headers(options.headers || {});
  headers.set("X-Telegram-Init-Data", initData);
  headers.set("Authorization", `Bearer ${initData}`);
  headers.set("Accept", "application/json");
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers, credentials: "same-origin", cache: "no-store" });
  let payload = null;
  try { payload = await response.json(); } catch {}
  if (!response.ok) throw new Error(typeof payload?.detail === "string" ? payload.detail : `Request failed (${response.status})`);
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
  elements.profileHandle.textContent = user.username ? `@${user.username} · Telegram verified` : `ID ${user.id ?? "—"} · Telegram verified`;
  elements.profileFallback.textContent = fullName.slice(0, 1).toUpperCase();
  try {
    const candidate = new URL(user.photo_url || "");
    if (candidate.protocol === "https:") {
      elements.profilePhoto.src = candidate.href;
      elements.profilePhoto.alt = `${fullName} profile photo`;
      elements.profilePhoto.referrerPolicy = "no-referrer";
      elements.profilePhoto.classList.remove("is-hidden");
      elements.profileFallback.classList.add("is-hidden");
    }
  } catch {}
}
function renderStats(payload, jobsPayload, workersPayload) {
  const bot = payload.bot || {}, usage = payload.usage || {}, redis = payload.redis || {}, database = payload.database || {};
  const jobs = jobsPayload?.jobs || {};
  const workers = workersPayload || jobsPayload?.workers || {};
  elements.totalUsers.textContent = compactNumber(usage.total_users);
  elements.messageCount.textContent = compactNumber(usage.message_count);
  elements.queuedJobs.textContent = compactNumber(jobs.queued);
  elements.deadJobs.textContent = compactNumber(jobs.dead);
  elements.redisLatency.textContent = redis.ok && redis.latency_ms != null ? `${redis.latency_ms} ms` : "Offline";
  elements.uptime.textContent = bot.uptime || t("starting");
  elements.botMode.textContent = bot.mode || "—";
  elements.lastUpdated.textContent = new Date(payload.generated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  setHealth(elements.botHealth, bot.active ? t("operational") : t("starting"), bot.active ? "ok" : "warn");
  setHealth(elements.redisHealth, redis.ok ? t("healthy") : t("unavailable"), redis.ok ? "ok" : "down");
  setHealth(elements.databaseHealth, database.ok ? t("connected") : (database.memory_fallback ? t("memoryFallback") : t("unavailable")), database.ok ? "ok" : "warn");
  setHealth(elements.workerHealth, workers.healthy ? `${workers.alive}/${workers.count} · ${workers.accepting ? t("accepting") : t("drained")}` : `${workers.alive || 0}/${workers.count || 0} ${t("unavailable")}`, workers.healthy ? (workers.accepting ? "ok" : "warn") : "down");
  elements.drainWorkersButton.disabled = !workers.accepting;
  elements.resumeWorkersButton.disabled = Boolean(workers.accepting);
  const overallHealthy = Boolean(bot.active && redis.ok && workers.healthy);
  elements.systemBadge.classList.toggle("down", !overallHealthy);
  elements.systemBadge.lastElementChild.textContent = overallHealthy ? t("operational") : t("unavailable");
  renderMaintenance(Boolean(bot.maintenance_mode));
  elements.jobSummary.textContent = `Queued ${jobs.queued || 0} · Running ${jobs.running || 0} · Dead ${jobs.dead || 0} · Succeeded ${jobs.succeeded || 0} · Cancelled ${jobs.cancelled || 0} · Capacity ${jobs.queue_available ?? "—"}`;
}
function renderMaintenance(enabled) {
  elements.maintenanceToggle.checked = enabled;
  elements.maintenanceNotice.textContent = enabled ? "Maintenance is ON. Normal-user features are paused." : "Normal bot service is active.";
  elements.maintenanceNotice.classList.toggle("warning", enabled);
}
function makeRuntimeControl(key, spec) {
  const wrapper = document.createElement("div"); wrapper.className = "setting-field";
  const copy = document.createElement("div"), label = document.createElement("label"), help = document.createElement("small");
  label.htmlFor = `runtime-${key}`; label.textContent = spec.label || key; help.textContent = spec.help || "Runtime setting"; copy.append(label, help);
  const control = document.createElement("div"); control.className = "setting-control"; let input;
  if (spec.kind === "bool") {
    const switchLabel = document.createElement("label"); switchLabel.className = "switch"; input = document.createElement("input"); input.type = "checkbox"; input.checked = Boolean(spec.value);
    const track = document.createElement("span"); track.className = "switch-track"; switchLabel.append(input, track); control.append(switchLabel);
  } else {
    input = document.createElement("input"); input.type = "number"; input.value = String(spec.value ?? ""); input.step = spec.kind === "float" ? "0.1" : "1";
    if (spec.min != null) input.min = String(spec.min); if (spec.max != null) input.max = String(spec.max); control.append(input);
  }
  input.id = `runtime-${key}`; input.name = key; input.dataset.kind = spec.kind || "str"; wrapper.append(copy, control); return wrapper;
}
function renderSettings(payload) {
  state.settings = payload; renderMaintenance(Boolean(payload.maintenance_mode)); elements.runtimeFields.replaceChildren();
  Object.entries(payload.runtime || {}).forEach(([key, spec]) => elements.runtimeFields.append(makeRuntimeControl(key, spec)));
}
function renderCors(origins) {
  state.corsOrigins = Array.isArray(origins) ? origins : []; elements.corsList.replaceChildren();
  elements.corsEmpty.classList.toggle("is-hidden", state.corsOrigins.length > 0);
  state.corsOrigins.forEach((origin) => {
    const row = document.createElement("div"); row.className = "origin-chip"; const value = document.createElement("span"); value.textContent = origin; value.title = origin;
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "button danger"; remove.textContent = t("remove"); remove.addEventListener("click", () => removeOrigin(origin, remove)); row.append(value, remove); elements.corsList.append(row);
  });
}

async function refreshJobs({ append = false } = {}) {
  const selectedState = elements.jobState.value;
  const cursor = append && state.jobCursor ? `&cursor=${encodeURIComponent(state.jobCursor)}` : "";
  const payload = await api(`/api/admin/runtime/jobs/list?state=${selectedState}&limit=50${cursor}`);
  if (!append) { elements.jobsList.replaceChildren(); state.selectedJobs.clear(); }
  payload.jobs.forEach((job) => {
    const row = document.createElement("article"); row.className = "data-row job-row";
    if (["dead", "cancelled"].includes(job.state)) {
      const check = document.createElement("input"); check.type = "checkbox"; check.className = "job-check"; check.addEventListener("change", () => { check.checked ? state.selectedJobs.add(job.id) : state.selectedJobs.delete(job.id); elements.retrySelectedButton.disabled = state.selectedJobs.size === 0; }); row.append(check);
    }
    const copy = document.createElement("div"); copy.className = "data-copy"; const title = document.createElement("strong"); title.textContent = `${job.type} · ${job.id.slice(0, 12)}`;
    const meta = document.createElement("small"); meta.textContent = `${job.state} · attempt ${job.attempts}/${job.max_attempts} · ${formatDate(job.created_at)}`;
    copy.append(title, meta);
    if (job.progress_stage) { const progress = document.createElement("div"); progress.className = "job-progress"; const bar = document.createElement("span"); bar.style.width = `${Math.max(0, Math.min(100, Number(job.progress_percent) || 0))}%`; const label = document.createElement("small"); label.textContent = `${job.progress_percent || 0}% · ${job.progress_stage}${job.progress_detail ? ` · ${job.progress_detail}` : ""}`; progress.append(bar, label); copy.append(progress); }
    if (job.last_error) { const error = document.createElement("p"); error.className = "row-error"; error.textContent = job.last_error; copy.append(error); } row.append(copy);
    const actions = document.createElement("div"); actions.className = "row-actions";
    if (["dead", "cancelled"].includes(job.state)) { const retry = document.createElement("button"); retry.className = "button small"; retry.textContent = t("retry"); retry.addEventListener("click", () => mutateJob(job.id, "retry", retry)); actions.append(retry); }
    if (["queued", "running"].includes(job.state)) { const cancel = document.createElement("button"); cancel.className = "button danger"; cancel.textContent = t("cancel"); cancel.addEventListener("click", () => mutateJob(job.id, "cancel", cancel)); actions.append(cancel); }
    row.append(actions); elements.jobsList.append(row);
  });
  state.jobCursor = payload.next_cursor; elements.loadMoreJobsButton.classList.toggle("is-hidden", !state.jobCursor);
  elements.jobsEmpty.classList.toggle("is-hidden", elements.jobsList.children.length > 0);
  elements.retrySelectedButton.disabled = state.selectedJobs.size === 0;
}

async function setWorkerAcceptance(action, button) {
  button.disabled = true;
  try {
    const payload = await api(`/api/admin/runtime/workers/${action}`, { method: "POST" });
    showToast(action === "drain" ? "Workers are draining." : "Workers resumed.");
    elements.drainWorkersButton.disabled = !payload.accepting;
    elements.resumeWorkersButton.disabled = Boolean(payload.accepting);
    await refreshAll({ silent: true });
  } catch (error) {
    showToast(error.message, true);
    button.disabled = false;
  }
}

async function mutateJob(jobId, action, button) {
  button.disabled = true;
  try { await api(`/api/admin/runtime/jobs/${encodeURIComponent(jobId)}/${action}`, { method: "POST" }); showToast(`Job ${action} accepted.`); await refreshJobs(); }
  catch (error) { showToast(error.message, true); button.disabled = false; }
}
async function retrySelected() {
  const ids = [...state.selectedJobs]; if (!ids.length) return; elements.retrySelectedButton.disabled = true;
  try { const payload = await api("/api/admin/runtime/jobs/retry-selected", { method: "POST", body: JSON.stringify({ job_ids: ids }) }); showToast(`${payload.retried} job(s) queued.`); await refreshJobs(); }
  catch (error) { showToast(error.message, true); }
}
function renderProviders(payload) {
  elements.providersList.replaceChildren(); elements.providerScope.textContent = `${t("processLocal")} · ${payload.instance_id || "instance"}`;
  Object.entries(payload.providers || {}).forEach(([name, provider]) => {
    const card = document.createElement("article"); card.className = "provider-card"; const heading = document.createElement("div"); heading.className = "provider-heading";
    const title = document.createElement("strong"); title.textContent = name; const badge = document.createElement("span"); badge.className = `health-value ${provider.available ? "" : "down"}`; badge.textContent = provider.available ? t("healthy") : `${provider.cooldown_remaining_seconds}s`; heading.append(title, badge);
    const meta = document.createElement("p"); meta.className = "muted"; meta.textContent = `Score ${provider.health_score} · ${provider.latency_ewma_ms ?? "—"} ms · ${provider.successes}/${provider.failures}`;
    const reset = document.createElement("button"); reset.className = "button small"; reset.textContent = t("reset"); reset.addEventListener("click", () => resetProvider(name, reset)); card.append(heading, meta, reset); elements.providersList.append(card);
  });
}
async function resetProvider(name, button) { button.disabled = true; try { await api("/api/admin/runtime/providers/reset", { method: "POST", body: JSON.stringify({ provider: name }) }); showToast(`${name} reset.`); await refreshProviders(); } catch (error) { showToast(error.message, true); button.disabled = false; } }
async function refreshProviders() { renderProviders(await api("/api/admin/runtime/providers")); }

function renderAdministrators(payload) {
  elements.adminList.replaceChildren(); (payload.administrators || []).forEach((userId) => {
    const row = document.createElement("div"); row.className = "data-row"; const copy = document.createElement("div"); copy.className = "data-copy"; const title = document.createElement("strong"); title.textContent = `Telegram ID ${userId}`; copy.append(title);
    const remove = document.createElement("button"); remove.className = "button danger"; remove.textContent = t("remove"); remove.addEventListener("click", () => mutateAdministrator("remove", Number(userId), remove)); row.append(copy, remove); elements.adminList.append(row);
  });
}
function renderAudit(payload) {
  elements.auditList.replaceChildren(); (payload.entries || []).forEach((entry) => {
    const row = document.createElement("div"); row.className = "data-row"; const copy = document.createElement("div"); copy.className = "data-copy"; const title = document.createElement("strong"); title.textContent = `${entry.action || "action"} · ${entry.target_id || "—"}`;
    const meta = document.createElement("small"); meta.textContent = `${entry.timestamp || ""} · actor ${entry.actor_id || "—"}`; copy.append(title, meta); row.append(copy); elements.auditList.append(row);
  });
}
async function refreshAdministrators() { const [admins, audit] = await Promise.all([api("/api/admin/administrators"), api("/api/admin/administrators/audit?limit=50")]); renderAdministrators(admins); renderAudit(audit); }
async function mutateAdministrator(action, userId, button) {
  button.disabled = true;
  try {
    const confirmation = await api("/api/admin/administrators/confirmations", { method: "POST", body: JSON.stringify({ action, user_id: userId }) });
    await api("/api/admin/administrators", { method: action === "remove" ? "DELETE" : "POST", body: JSON.stringify({ user_id: userId, confirmation_token: confirmation.confirmation_token }) });
    showToast(`Administrator ${action} completed.`); await refreshAdministrators();
  } catch (error) { showToast(error.message, true); button.disabled = false; }
}

async function refreshAll({ initial = false, silent = false } = {}) {
  if (state.refreshing) return; state.refreshing = true; elements.refreshButton.disabled = true;
  try {
    const [profile, stats, settings, cors, jobs, workers] = await Promise.all([api("/api/admin/me"), api("/api/admin/stats"), api("/api/admin/settings"), api("/api/admin/cors"), api("/api/admin/runtime/jobs"), api("/api/admin/runtime/workers")]);
    renderProfile(profile); renderStats(stats, jobs, workers); renderSettings(settings); renderCors(cors.origins);
    const optional = await Promise.allSettled([refreshJobs(), refreshProviders(), refreshAdministrators()]);
    optional.filter((item) => item.status === "rejected").forEach((item) => console.warn(item.reason));
    elements.authState.classList.add("is-hidden"); elements.dashboard.classList.remove("is-hidden"); if (!initial && !silent) { haptic(); showToast("Dashboard refreshed."); }
  } catch (error) {
    if (initial) { elements.authState.classList.add("error"); elements.authState.replaceChildren(); const copy = document.createElement("div"), title = document.createElement("strong"), detail = document.createElement("p"); title.textContent = "Admin access denied"; detail.textContent = error.message; copy.append(title, detail); elements.authState.append(copy); }
    else showToast(error.message, true);
  } finally { state.refreshing = false; elements.refreshButton.disabled = false; }
}

async function updateMaintenance(nextValue) { elements.maintenanceToggle.disabled = true; try { const payload = await api("/api/admin/settings", { method: "POST", body: JSON.stringify({ maintenance_mode: nextValue }) }); renderMaintenance(Boolean(payload.maintenance_mode)); haptic("medium"); showToast("Maintenance setting updated."); } catch (error) { renderMaintenance(!nextValue); showToast(error.message, true); } finally { elements.maintenanceToggle.disabled = false; } }
async function confirmMaintenance(nextValue) { const message = nextValue ? "Pause bot features for normal users?" : "Resume bot features for normal users?"; if (typeof telegram?.showConfirm === "function") { telegram.showConfirm(message, (confirmed) => confirmed ? updateMaintenance(nextValue) : renderMaintenance(!nextValue)); return; } if (window.confirm(message)) await updateMaintenance(nextValue); else renderMaintenance(!nextValue); }
async function addOrigin(event) { event.preventDefault(); const origin = elements.corsOrigin.value.trim(); if (!origin) return; const submit = elements.corsForm.querySelector("button[type='submit']"); submit.disabled = true; try { const payload = await api("/api/admin/cors", { method: "POST", body: JSON.stringify({ origin }) }); elements.corsOrigin.value = ""; renderCors(payload.origins); showToast(payload.changed ? "Origin added." : "Origin already allowed."); } catch (error) { showToast(error.message, true); } finally { submit.disabled = false; } }
async function removeOrigin(origin, button) { button.disabled = true; try { const payload = await api("/api/admin/cors", { method: "DELETE", body: JSON.stringify({ origin }) }); renderCors(payload.origins); showToast("Origin removed."); } catch (error) { button.disabled = false; showToast(error.message, true); } }
async function saveRuntime(event) { event.preventDefault(); const runtime = {}; elements.runtimeFields.querySelectorAll("input[name]").forEach((input) => { runtime[input.name] = input.dataset.kind === "bool" ? input.checked : (input.dataset.kind === "int" ? Number.parseInt(input.value, 10) : Number.parseFloat(input.value)); }); if (Object.values(runtime).some(Number.isNaN)) return showToast("Enter valid values.", true); elements.saveRuntimeButton.disabled = true; try { const payload = await api("/api/admin/settings", { method: "POST", body: JSON.stringify({ runtime }) }); renderSettings(payload); showToast("Runtime settings saved."); } catch (error) { showToast(error.message, true); } finally { elements.saveRuntimeButton.disabled = false; } }

function applyTelegramTheme() { const root = document.documentElement, params = telegram?.themeParams || {}; Object.entries({ "--tg-bg": params.bg_color, "--tg-text": params.text_color }).forEach(([name, value]) => { if (value) root.style.setProperty(name, value); }); updateViewportVars(); }
function updateViewportVars() { const root = document.documentElement, viewportHeight = telegram?.viewportHeight || window.innerHeight, stableHeight = telegram?.viewportStableHeight || viewportHeight; root.style.setProperty("--tg-viewport-height", `${viewportHeight}px`); root.style.setProperty("--tg-viewport-stable-height", `${stableHeight}px`); }
function initializeTelegram() { if (!telegram) return; telegram.ready(); telegram.expand(); applyTelegramTheme(); telegram.onEvent?.("themeChanged", applyTelegramTheme); telegram.onEvent?.("viewportChanged", updateViewportVars); }

document.querySelectorAll("[data-nav-target]").forEach((item) => item.addEventListener("click", () => document.querySelectorAll("[data-nav-target]").forEach((candidate) => candidate.classList.toggle("active", candidate === item))));
elements.refreshButton.addEventListener("click", () => refreshAll());
elements.languageButton.addEventListener("click", () => { state.language = state.language === "en" ? "km" : "en"; applyLanguage(); refreshAll({ silent: true }); });
elements.maintenanceToggle.addEventListener("change", (event) => confirmMaintenance(event.target.checked));
elements.corsForm.addEventListener("submit", addOrigin); elements.runtimeForm.addEventListener("submit", saveRuntime);
elements.jobState.addEventListener("change", () => refreshJobs()); elements.refreshJobsButton.addEventListener("click", () => refreshJobs());
elements.drainWorkersButton.addEventListener("click", () => setWorkerAcceptance("drain", elements.drainWorkersButton));
elements.resumeWorkersButton.addEventListener("click", () => setWorkerAcceptance("resume", elements.resumeWorkersButton));
elements.loadMoreJobsButton.addEventListener("click", () => refreshJobs({ append: true })); elements.retrySelectedButton.addEventListener("click", retrySelected);
elements.adminForm.addEventListener("submit", async (event) => { event.preventDefault(); const userId = Number.parseInt(elements.adminUserId.value, 10); if (!Number.isSafeInteger(userId) || userId <= 0) return showToast("Enter a valid Telegram user ID.", true); const button = elements.adminForm.querySelector("button"); await mutateAdministrator("add", userId, button); elements.adminUserId.value = ""; });

applyLanguage(); initializeTelegram(); refreshAll({ initial: true });
