"use strict";

const telegram = window.Telegram?.WebApp ?? null;
const API_TIMEOUT_MS = 15_000;
const DASHBOARD_REFRESH_MS = 30_000;
const MONITOR_INTERVALS = new Set([3000, 5000, 10000]);
const storedMonitorInterval = Number.parseInt(window.localStorage.getItem("admin-monitor-interval") || "5000", 10);
const state = {
  language: window.localStorage.getItem("admin-language") || "en",
  settings: null,
  corsOrigins: [],
  refreshing: false,
  overviewRefreshing: false,
  toastTimer: null,
  jobCursor: null,
  selectedJobs: new Set(),
  jobRequestId: 0,
  jobAppendPending: false,
  queueHistory: [],
  monitorRefreshing: false,
  monitorPaused: false,
  monitorTimer: null,
  monitorFullscreen: false,
  monitorVisible: false,
  monitorIntervalMs: MONITOR_INTERVALS.has(storedMonitorInterval) ? storedMonitorInterval : 5000,
  monitorHistory: { cpu: [], pressure: [], requests: [], tts: [], age: [], failures: [] },
  monitorLastSample: null,
  monitorTtsFilter: "all",
  monitorWorkloadType: "all",
  monitorTtsPayload: null,
  monitorLogs: [],
  dashboardTimer: null,
  connected: navigator.onLine,
  connectionState: navigator.onLine ? "connecting" : "offline",
  currentJobId: "",
  lastDashboardRefresh: 0,
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
    unavailable: "Unavailable", disabled: "Disabled", connected: "Connected", memoryFallback: "Memory fallback", active: "Active",
    paused: "Paused", drainWorkers: "Drain workers", resumeWorkers: "Resume workers", accepting: "Accepting jobs", drained: "Drained", processLocal: "Process-local", reset: "Reset", retry: "Retry", cancel: "Cancel", remove: "Remove",
    allTypes: "All types", searchJobs: "Search jobs", queueAge: "Oldest queued", throughput: "Throughput/min", failureRate: "Failure rate",
    liveMonitor: "Live operations", botMonitor: "Bot Monitor", live: "Live", fullScreen: "Full screen", exitFullScreen: "Exit full screen",
    processState: "Bot process", workerProcesses: "Worker processes", ttsProcesses: "TTS processes", queueActivity: "Queue activity",
    processDetails: "Process details", durableWorkers: "Durable workers", ttsProcessQueue: "TTS process queue",
    ttsMonitorHelp: "Running and waiting voice jobs refresh automatically.", noTtsProcesses: "No active or queued TTS processes.",
    runtimeLogs: "Runtime logs", logsSafe: "Recent logs are bounded and secrets are automatically hidden.", allLogs: "All logs",
    searchLogs: "Search logs", pauseLive: "Pause", resumeLive: "Resume", noLogs: "No matching runtime logs.", monitor: "Monitor",
    online: "Online", running: "Running", waiting: "Waiting", noWorkers: "No workers are running in this process.",
    secureRealtime: "Secure real-time control", lastSync: "Last sync", monitorConnecting: "Connecting to runtime",
    monitorConnectingHelp: "Waiting for the first secure process snapshot…", cpuActivity: "CPU activity", queuePressure: "Queue pressure",
    webRequests: "Web requests", ttsLoad: "Media workload", allProcesses: "All", copyLogs: "Copy logs", logsCopied: "Logs copied.", downloadLogs: "Download", logsDownloaded: "Redacted logs downloaded.",
    mediaProcessQueue: "TTS, OCR & transcription", mediaMonitorHelp: "Real-time progress for running and waiting media jobs.", noMediaProcesses: "No active or queued media processes.",
    incidentRecovery: "Incident recovery", incidentRecoveryHelp: "Component restarts, backoff, recovery, and open configuration circuits.", noIncidents: "No component incidents since startup.", publicUrl: "Public URL",
    monitorHealthy: "Runtime healthy", monitorHealthyHelp: "Workers and queue are operating within normal limits.",
    monitorWarning: "Runtime needs attention", monitorWarningHelp: "Queue pressure or recent failure rate is elevated.",
    monitorCritical: "Worker interruption detected", monitorCriticalHelp: "One or more durable workers are not healthy.",
    monitorOffline: "Monitor unavailable", monitorOfflineHelp: "The secure runtime snapshot could not be refreshed.",
    connecting: "Connecting", offline: "Offline", reconnecting: "Reconnecting",
    details: "Details", jobDetails: "Job details", copyJobId: "Copy job ID", close: "Close", copied: "Copied",
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
    unavailable: "មិនអាចប្រើបាន", disabled: "Disabled", connected: "បានភ្ជាប់", memoryFallback: "ប្រើ Memory បម្រុង", active: "សកម្ម",
    paused: "បានផ្អាក", drainWorkers: "ផ្អាកទទួលការងារ", resumeWorkers: "បន្តទទួលការងារ", accepting: "កំពុងទទួលការងារ", drained: "បានផ្អាកទទួលការងារ", processLocal: "តាម process", reset: "កំណត់ឡើងវិញ", retry: "សាកឡើងវិញ", cancel: "បោះបង់", remove: "ដកចេញ",
    allTypes: "គ្រប់ប្រភេទ", searchJobs: "ស្វែងរកការងារ", queueAge: "ការងារចាស់បំផុត", throughput: "បានបញ្ចប់/នាទី", failureRate: "អត្រាបរាជ័យ",
    liveMonitor: "ប្រតិបត្តិការផ្ទាល់", botMonitor: "តាមដានបូត", live: "ផ្ទាល់", fullScreen: "ពេញអេក្រង់", exitFullScreen: "ចាកចេញពីពេញអេក្រង់",
    processState: "Process បូត", workerProcesses: "Worker processes", ttsProcesses: "TTS processes", queueActivity: "សកម្មភាពជួរ",
    processDetails: "ព័ត៌មាន Process", durableWorkers: "Durable workers", ttsProcessQueue: "ជួរ TTS process",
    ttsMonitorHelp: "ការងារសំឡេងដែលកំពុងដំណើរការ និងរង់ចាំ ផ្ទុកឡើងវិញដោយស្វ័យប្រវត្តិ។", noTtsProcesses: "មិនមាន TTS process កំពុងដំណើរការ ឬរង់ចាំទេ។",
    runtimeLogs: "Runtime logs", logsSafe: "Logs ថ្មីៗត្រូវបានកំណត់ចំនួន ហើយ secrets ត្រូវបានលាក់ដោយស្វ័យប្រវត្តិ។", allLogs: "Logs ទាំងអស់",
    searchLogs: "ស្វែងរក logs", pauseLive: "ផ្អាក", resumeLive: "បន្ត", noLogs: "មិនមាន runtime log ដែលត្រូវគ្នាទេ។", monitor: "តាមដាន",
    online: "អនឡាញ", running: "កំពុងដំណើរការ", waiting: "កំពុងរង់ចាំ", noWorkers: "មិនមាន worker ដំណើរការនៅក្នុង process នេះទេ។",
    secureRealtime: "ការគ្រប់គ្រងផ្ទាល់ដែលមានសុវត្ថិភាព", lastSync: "ធ្វើសមកាលកម្មចុងក្រោយ", monitorConnecting: "កំពុងភ្ជាប់ Runtime",
    monitorConnectingHelp: "កំពុងរង់ចាំទិន្នន័យ process ដែលមានសុវត្ថិភាព…", cpuActivity: "សកម្មភាព CPU", queuePressure: "សម្ពាធជួរ",
    webRequests: "សំណើ Web", ttsLoad: "បន្ទុក Media", allProcesses: "ទាំងអស់", copyLogs: "ចម្លង Logs", logsCopied: "បានចម្លង Logs។", downloadLogs: "ទាញយក", logsDownloaded: "បានទាញយក Logs ដែលបានលាក់ព័ត៌មានសម្ងាត់។",
    mediaProcessQueue: "ដំណើរការ TTS, OCR និងបម្លែងសំឡេង", mediaMonitorHelp: "វឌ្ឍនភាពផ្ទាល់សម្រាប់ការងារ Media។", noMediaProcesses: "មិនមានការងារ Media កំពុងដំណើរការ ឬរង់ចាំទេ។",
    incidentRecovery: "ការស្ដារបញ្ហាឡើងវិញ", incidentRecoveryHelp: "ប្រវត្តិ Restart, Backoff, Recovery និង Configuration Circuit។", noIncidents: "មិនមានបញ្ហា Component ចាប់តាំងពីចាប់ផ្ដើម។", publicUrl: "URL សាធារណៈ",
    monitorHealthy: "Runtime ដំណើរការល្អ", monitorHealthyHelp: "Workers និងជួរការងារដំណើរការក្នុងកម្រិតធម្មតា។",
    monitorWarning: "Runtime ត្រូវការការពិនិត្យ", monitorWarningHelp: "សម្ពាធជួរ ឬអត្រាបរាជ័យថ្មីៗកំពុងកើនឡើង។",
    monitorCritical: "រកឃើញ Worker ផ្អាក", monitorCriticalHelp: "Durable worker មួយ ឬច្រើនមិនមានសុខភាពល្អ។",
    monitorOffline: "មិនអាចប្រើ Monitor", monitorOfflineHelp: "មិនអាចផ្ទុកទិន្នន័យ Runtime ដែលមានសុវត្ថិភាពបានទេ។",
    connecting: "កំពុងភ្ជាប់", offline: "ក្រៅបណ្ដាញ", reconnecting: "កំពុងភ្ជាប់ឡើងវិញ",
    details: "ព័ត៌មានលម្អិត", jobDetails: "ព័ត៌មានការងារ", copyJobId: "ចម្លង Job ID", close: "បិទ", copied: "បានចម្លង",
  },
};

const $ = (id) => document.getElementById(id);
const elements = {
  authState: $("authState"), dashboard: $("dashboard"), refreshButton: $("refreshButton"), languageButton: $("languageButton"),
  connectionBadge: $("connectionBadge"), connectionText: $("connectionText"),
  profilePhoto: $("profilePhoto"), profileFallback: $("profileFallback"), profileName: $("profileName"), profileHandle: $("profileHandle"),
  systemBadge: $("systemBadge"), heroSyncState: $("heroSyncState"), totalUsers: $("totalUsers"), messageCount: $("messageCount"), queuedJobs: $("queuedJobs"),
  deadJobs: $("deadJobs"), redisLatency: $("redisLatency"), uptime: $("uptime"), lastUpdated: $("lastUpdated"),
  botHealth: $("botHealth"), redisHealth: $("redisHealth"), databaseHealth: $("databaseHealth"), workerHealth: $("workerHealth"),
  botMode: $("botMode"), maintenanceToggle: $("maintenanceToggle"), maintenanceNotice: $("maintenanceNotice"),
  corsForm: $("corsForm"), corsOrigin: $("corsOrigin"), corsList: $("corsList"), corsEmpty: $("corsEmpty"),
  runtimeForm: $("runtimeForm"), runtimeFields: $("runtimeFields"), saveRuntimeButton: $("saveRuntimeButton"),
  jobState: $("jobState"), jobType: $("jobType"), jobSearch: $("jobSearch"), refreshJobsButton: $("refreshJobsButton"), jobsList: $("jobsList"), jobsEmpty: $("jobsEmpty"),
  jobSummary: $("jobSummary"), drainWorkersButton: $("drainWorkersButton"), resumeWorkersButton: $("resumeWorkersButton"), retrySelectedButton: $("retrySelectedButton"), loadMoreJobsButton: $("loadMoreJobsButton"),
  queueCharts: $("queueCharts"),
  monitor: $("monitor"), monitorLiveState: $("monitorLiveState"), monitorInterval: $("monitorInterval"), refreshMonitorButton: $("refreshMonitorButton"), monitorFullscreenButton: $("monitorFullscreenButton"),
  monitorAlert: $("monitorAlert"), monitorAlertTitle: $("monitorAlertTitle"), monitorAlertDetail: $("monitorAlertDetail"), monitorLatency: $("monitorLatency"),
  monitorProcessState: $("monitorProcessState"), monitorProcessMeta: $("monitorProcessMeta"), monitorWorkerState: $("monitorWorkerState"), monitorWorkerMeta: $("monitorWorkerMeta"),
  monitorTtsState: $("monitorTtsState"), monitorTtsMeta: $("monitorTtsMeta"), monitorQueueState: $("monitorQueueState"), monitorQueueMeta: $("monitorQueueMeta"),
  monitorCpuValue: $("monitorCpuValue"), monitorCpuChart: $("monitorCpuChart"), monitorPressureValue: $("monitorPressureValue"), monitorPressureChart: $("monitorPressureChart"),
  monitorRequestsValue: $("monitorRequestsValue"), monitorRequestsChart: $("monitorRequestsChart"), monitorTtsTrendValue: $("monitorTtsTrendValue"), monitorTtsTrendChart: $("monitorTtsTrendChart"),
  monitorQueueAgeValue: $("monitorQueueAgeValue"), monitorQueueAgeChart: $("monitorQueueAgeChart"), monitorFailureValue: $("monitorFailureValue"), monitorFailureChart: $("monitorFailureChart"),
  monitorUpdated: $("monitorUpdated"), monitorProcessDetails: $("monitorProcessDetails"), monitorWorkerCount: $("monitorWorkerCount"), monitorWorkerList: $("monitorWorkerList"),
  monitorIncidentCount: $("monitorIncidentCount"), monitorIncidentList: $("monitorIncidentList"), monitorIncidentsEmpty: $("monitorIncidentsEmpty"),
  monitorTtsCount: $("monitorTtsCount"), monitorTtsList: $("monitorTtsList"), monitorTtsEmpty: $("monitorTtsEmpty"), monitorWorkloadType: $("monitorWorkloadType"), monitorLogLevel: $("monitorLogLevel"),
  monitorLogSearch: $("monitorLogSearch"), monitorLogCount: $("monitorLogCount"), monitorCopyLogsButton: $("monitorCopyLogsButton"), monitorDownloadLogsButton: $("monitorDownloadLogsButton"), monitorPauseButton: $("monitorPauseButton"), monitorLogList: $("monitorLogList"), monitorLogsEmpty: $("monitorLogsEmpty"),
  providersList: $("providersList"), providerScope: $("providerScope"), adminForm: $("adminForm"), adminUserId: $("adminUserId"),
  adminList: $("adminList"), auditList: $("auditList"), toast: $("toast"),
  jobDetailDialog: $("jobDetailDialog"), jobDetailTitle: $("jobDetailTitle"), jobDetailContent: $("jobDetailContent"),
  closeJobDetailButton: $("closeJobDetailButton"), copyJobIdButton: $("copyJobIdButton"),
};

function t(key) { return translations[state.language]?.[key] || translations.en[key] || key; }
function applyLanguage() {
  document.documentElement.lang = state.language;
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  elements.languageButton.textContent = state.language === "en" ? "ខ្មែរ" : "English";
  elements.jobType.options[0].textContent = t("allTypes");
  elements.jobSearch.placeholder = t("searchJobs");
  elements.monitorPauseButton.textContent = state.monitorPaused ? t("resumeLive") : t("pauseLive");
  elements.monitorFullscreenButton.textContent = state.monitorFullscreen ? t("exitFullScreen") : t("fullScreen");
  elements.monitorInterval.value = String(state.monitorIntervalMs);
  renderConnectionState(state.connectionState);
  window.localStorage.setItem("admin-language", state.language);
}

function haptic(type = "light") { try { telegram?.HapticFeedback?.impactOccurred(type); } catch {} }
function renderConnectionState(status) {
  state.connectionState = status;
  state.connected = status === "online";
  elements.connectionText.textContent = t(status);
  elements.connectionBadge.classList.toggle("offline", status === "offline");
  elements.connectionBadge.classList.toggle("reconnecting", ["connecting", "reconnecting"].includes(status));
}
function confirmAction(message) {
  return new Promise((resolve) => {
    if (typeof telegram?.showConfirm === "function") {
      telegram.showConfirm(message, (confirmed) => resolve(Boolean(confirmed)));
      return;
    }
    resolve(window.confirm(message));
  });
}
async function copyText(value) {
  if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(value);
  else {
    const area = document.createElement("textarea"); area.value = value; area.setAttribute("readonly", ""); area.style.position = "fixed"; area.style.opacity = "0";
    document.body.append(area); area.select(); document.execCommand("copy"); area.remove();
  }
}
function showToast(message, isError = false) {
  window.clearTimeout(state.toastTimer);
  elements.toast.textContent = String(message);
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("show");
  state.toastTimer = window.setTimeout(() => elements.toast.classList.remove("show"), 3200);
}
function runAsync(action) {
  Promise.resolve()
    .then(action)
    .catch((error) => showToast(error instanceof Error ? error.message : String(error), true));
}
function compactNumber(value) { return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(Number(value) || 0); }
function formatDate(value) { return value ? new Date(Number(value) * 1000).toLocaleString() : "—"; }
function formatIsoDate(value) { const date = value ? new Date(value) : null; return date && !Number.isNaN(date.getTime()) ? date.toLocaleString() : "—"; }
function formatDuration(value) { const seconds = Math.max(0, Number(value) || 0); if (seconds < 60) return `${Math.round(seconds)}s`; if (seconds < 3600) return `${Math.round(seconds / 60)}m`; return `${(seconds / 3600).toFixed(1)}h`; }
function formatMemory(value) { const kilobytes = Number(value); return Number.isFinite(kilobytes) && kilobytes >= 0 ? `${(kilobytes / 1024).toFixed(1)} MiB` : "—"; }

async function api(path, options = {}) {
  const initData = String(window.Telegram?.WebApp?.initData || "").trim();
  if (!initData) throw new Error("Open this dashboard from the bot inside Telegram.");
  const headers = new Headers(options.headers || {});
  headers.set("X-Telegram-Init-Data", initData);
  headers.set("Accept", "application/json");
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const controller = new AbortController();
  let timedOut = false;
  const timeout = window.setTimeout(() => { timedOut = true; controller.abort(); }, API_TIMEOUT_MS);
  let response;
  let payload = null;
  try {
    response = await fetch(path, { ...options, headers, signal: controller.signal, credentials: "same-origin", cache: "no-store" });
    try { payload = await response.json(); } catch (error) { if (timedOut) throw error; }
  } catch (error) {
    renderConnectionState(navigator.onLine ? "reconnecting" : "offline");
    if (timedOut) throw new Error("The server took too long to respond. Please retry.");
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
  renderConnectionState("online");
  if (!response.ok) throw new Error(typeof payload?.detail === "string" ? payload.detail : `Request failed (${response.status})`);
  return payload;
}

function setHealth(element, label, kind = "ok") {
  element.textContent = label;
  element.classList.toggle("warn", kind === "warn");
  element.classList.toggle("down", kind === "down");
}
function sparkline(values, color) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 120 36"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Recent samples");
  const clean = values.map((value) => Math.max(0, Number(value) || 0)); const maximum = Math.max(1, ...clean);
  const chartValues = clean.length === 1 ? [clean[0], clean[0]] : clean;
  const points = chartValues.map((value, index) => `${(index / (chartValues.length - 1)) * 120},${34 - (value / maximum) * 30}`).join(" ");
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline"); line.setAttribute("points", points); line.setAttribute("fill", "none"); line.setAttribute("stroke", color); line.setAttribute("stroke-width", "3"); line.setAttribute("stroke-linecap", "round"); line.setAttribute("stroke-linejoin", "round"); svg.append(line); return svg;
}
function renderQueueCharts(jobs) {
  const sample = { age: Number(jobs.oldest_queued_age_seconds) || 0, throughput: Number(jobs.throughput_per_minute) || 0, failures: Number(jobs.failure_rate_percent) || 0 };
  state.queueHistory.push(sample); if (state.queueHistory.length > 20) state.queueHistory.shift();
  const specs = [
    ["age", t("queueAge"), formatDuration(sample.age), "#f7b955"],
    ["throughput", t("throughput"), sample.throughput.toFixed(2), "#50d890"],
    ["failures", t("failureRate"), `${sample.failures.toFixed(1)}%`, "#ff6b7a"],
  ];
  elements.queueCharts.replaceChildren();
  specs.forEach(([key, labelText, valueText, color]) => { const card = document.createElement("article"); card.className = "queue-chart"; const copy = document.createElement("div"), label = document.createElement("span"), value = document.createElement("strong"); label.textContent = labelText; value.textContent = valueText; copy.append(label, value); card.append(copy, sparkline(state.queueHistory.map((item) => item[key]), color)); elements.queueCharts.append(card); });
}

function appendMonitorDetail(container, labelText, valueText) {
  const row = document.createElement("div"); row.className = "monitor-detail-row";
  const label = document.createElement("span"); label.textContent = labelText;
  const value = document.createElement("strong"); value.textContent = valueText;
  row.append(label, value); container.append(row);
}

function pushMonitorSample(series, value) {
  series.push(Math.max(0, Number(value) || 0));
  if (series.length > 30) series.shift();
}

function renderMonitorTrend(container, values, color) {
  container.replaceChildren(sparkline(values.length ? values : [0], color));
}

function updateMonitorTrends(payload) {
  const process = payload.process || {}, health = payload.health || {}, tts = payload.tts || {}, workloads = payload.workloads || {}, queue = payload.queue || {};
  const sampledAt = Number(process.sampled_at), cpuSeconds = Number(process.cpu_seconds);
  let cpuPercent = 0;
  const previous = state.monitorLastSample;
  if (previous && previous.pid === process.pid && sampledAt > previous.sampledAt && cpuSeconds >= previous.cpuSeconds) {
    cpuPercent = Math.min(999, ((cpuSeconds - previous.cpuSeconds) / (sampledAt - previous.sampledAt)) * 100);
  }
  if (Number.isFinite(sampledAt) && Number.isFinite(cpuSeconds)) state.monitorLastSample = { pid: process.pid, sampledAt, cpuSeconds };
  const pressure = Number(health.queue_pressure_percent) || 0;
  const requests = Number(process.active_requests) || 0;
  const ttsLoad = (Number(workloads.running_count ?? tts.running_count) || 0) + (Number(workloads.queued_count ?? tts.queued_count) || 0);
  const queueAge = Number(queue.oldest_queued_age_seconds) || 0;
  const failureRate = Number(health.failure_rate_percent ?? queue.failure_rate_percent) || 0;
  pushMonitorSample(state.monitorHistory.cpu, cpuPercent);
  pushMonitorSample(state.monitorHistory.pressure, pressure);
  pushMonitorSample(state.monitorHistory.requests, requests);
  pushMonitorSample(state.monitorHistory.tts, ttsLoad);
  pushMonitorSample(state.monitorHistory.age, queueAge);
  pushMonitorSample(state.monitorHistory.failures, failureRate);
  elements.monitorCpuValue.textContent = `${cpuPercent.toFixed(1)}%`;
  elements.monitorPressureValue.textContent = `${pressure.toFixed(1)}%`;
  elements.monitorRequestsValue.textContent = compactNumber(requests);
  elements.monitorTtsTrendValue.textContent = compactNumber(ttsLoad);
  elements.monitorQueueAgeValue.textContent = formatDuration(queueAge);
  elements.monitorFailureValue.textContent = `${failureRate.toFixed(1)}%`;
  renderMonitorTrend(elements.monitorCpuChart, state.monitorHistory.cpu, "#65b5ff");
  renderMonitorTrend(elements.monitorPressureChart, state.monitorHistory.pressure, pressure >= 80 ? "#ff6b79" : "#a58bff");
  renderMonitorTrend(elements.monitorRequestsChart, state.monitorHistory.requests, "#50d890");
  renderMonitorTrend(elements.monitorTtsTrendChart, state.monitorHistory.tts, "#ffbd59");
  renderMonitorTrend(elements.monitorQueueAgeChart, state.monitorHistory.age, "#f7b955");
  renderMonitorTrend(elements.monitorFailureChart, state.monitorHistory.failures, failureRate >= 20 ? "#ff6b79" : "#50d890");
}

function renderMonitorAlert(payload, latencyMs) {
  const healthState = ["healthy", "warning", "critical"].includes(payload.health?.state) ? payload.health.state : "warning";
  const copy = {
    healthy: [t("monitorHealthy"), t("monitorHealthyHelp")],
    warning: [t("monitorWarning"), t("monitorWarningHelp")],
    critical: [t("monitorCritical"), t("monitorCriticalHelp")],
  }[healthState];
  elements.monitorAlert.classList.remove("healthy", "warning", "critical", "down");
  elements.monitorAlert.classList.add(healthState);
  elements.monitorAlertTitle.textContent = copy[0];
  elements.monitorAlertDetail.textContent = copy[1];
  elements.monitorLatency.textContent = `${Math.max(0, Math.round(latencyMs))} ms`;
}

function renderMonitorWorkers(workers) {
  const rows = Array.isArray(workers.workers) ? workers.workers : [];
  const restartHistory = Array.isArray(workers.restart_history) ? workers.restart_history : [];
  elements.monitorWorkerList.replaceChildren();
  elements.monitorWorkerCount.textContent = `${workers.alive || 0}/${workers.count || 0} · ${workers.restart_total || 0} restarts`;
  if (!rows.length) {
    const empty = document.createElement("p"); empty.className = "empty-state"; empty.textContent = t("noWorkers"); elements.monitorWorkerList.append(empty);
  }
  rows.forEach((worker) => {
    const row = document.createElement("div"); row.className = "monitor-worker-row";
    const status = document.createElement("span"); status.className = `monitor-state-dot ${worker.alive ? "" : "down"}`; status.setAttribute("aria-label", worker.alive ? t("healthy") : t("unavailable"));
    const copy = document.createElement("div"); copy.className = "data-copy";
    const title = document.createElement("strong"); title.textContent = worker.worker_id || "worker";
    const meta = document.createElement("small"); meta.textContent = `${worker.alive ? t("running") : t("unavailable")} · heartbeat ${formatDate(worker.last_heartbeat_at)} · restarts ${worker.restart_count || 0}`;
    copy.append(title, meta);
    if (worker.last_error) { const error = document.createElement("p"); error.className = "row-error"; error.textContent = worker.last_error; copy.append(error); }
    row.append(status, copy); elements.monitorWorkerList.append(row);
  });
  restartHistory.slice(0, 5).forEach((restart) => {
    const row = document.createElement("div"); row.className = "monitor-worker-row restart-history";
    const status = document.createElement("span"); status.className = "monitor-state-dot down";
    const copy = document.createElement("div"); copy.className = "data-copy";
    const title = document.createElement("strong"); title.textContent = `${restart.worker_id || "worker"} · restart #${restart.restart_count || 0}`;
    const meta = document.createElement("small"); meta.textContent = `${formatDate(restart.timestamp)} · backoff ${Number(restart.delay_seconds || 0).toFixed(1)}s`;
    copy.append(title, meta);
    if (restart.error) { const error = document.createElement("p"); error.className = "row-error"; error.textContent = restart.error; copy.append(error); }
    row.append(status, copy); elements.monitorWorkerList.append(row);
  });
}

function renderMonitorTts(tts, workloads = null) {
  const source = workloads && (Array.isArray(workloads.running) || Array.isArray(workloads.queued)) ? workloads : tts;
  const running = Array.isArray(source.running) ? source.running : [], queued = Array.isArray(source.queued) ? source.queued : [];
  state.monitorTtsPayload = { tts, workloads: source };
  const allJobs = [...running.map((job) => ({ ...job, monitorState: "running" })), ...queued.map((job) => ({ ...job, monitorState: "queued" }))];
  const stateFiltered = state.monitorTtsFilter === "all" ? allJobs : allJobs.filter((job) => job.monitorState === state.monitorTtsFilter);
  const jobs = state.monitorWorkloadType === "all" ? stateFiltered : stateFiltered.filter((job) => String(job.type || "tts") === state.monitorWorkloadType);
  elements.monitorTtsList.replaceChildren();
  const truncated = source.running_truncated || source.queued_truncated ? "+" : "";
  elements.monitorTtsCount.textContent = `${running.length} ${t("running")} · ${queued.length}${truncated} ${t("waiting")}`;
  elements.monitorTtsEmpty.classList.toggle("is-hidden", jobs.length > 0);
  document.querySelectorAll("[data-tts-filter]").forEach((button) => button.classList.toggle("active", button.dataset.ttsFilter === state.monitorTtsFilter));
  jobs.forEach((job) => {
    const row = document.createElement("div"); row.className = "monitor-job-row";
    const badge = document.createElement("span"); badge.className = `monitor-job-badge ${job.monitorState}`; badge.textContent = job.monitorState === "running" ? t("running") : t("waiting");
    const copy = document.createElement("div"); copy.className = "data-copy";
    const jobType = String(job.type || "tts").toUpperCase();
    const title = document.createElement("strong"); title.textContent = `${jobType} · ${String(job.id || "").slice(0, 18)}`;
    const meta = document.createElement("small"); meta.textContent = `${job.progress_percent || 0}% · ${job.progress_stage || job.monitorState} · attempt ${job.attempts || 0}/${job.max_attempts || 0}`;
    copy.append(title, meta);
    const progress = document.createElement("div"); progress.className = "monitor-job-progress"; const progressBar = document.createElement("span"); progressBar.style.width = `${Math.max(0, Math.min(100, Number(job.progress_percent) || 0))}%`; progress.append(progressBar); copy.append(progress);
    if (job.progress_detail) { const detail = document.createElement("small"); detail.textContent = job.progress_detail; copy.append(detail); }
    if (job.last_error) { const error = document.createElement("p"); error.className = "row-error"; error.textContent = job.last_error; copy.append(error); }
    row.append(badge, copy); elements.monitorTtsList.append(row);
  });
}

function renderMonitorIncidents(incidents) {
  const events = Array.isArray(incidents?.events) ? incidents.events : [];
  elements.monitorIncidentList.replaceChildren();
  elements.monitorIncidentCount.textContent = `${events.length}/${incidents?.captured || events.length}`;
  elements.monitorIncidentsEmpty.classList.toggle("is-hidden", events.length > 0);
  events.slice(0, 20).forEach((event) => {
    const row = document.createElement("div"); row.className = `monitor-incident-row severity-${event.severity || "info"}`;
    const heading = document.createElement("div"); heading.className = "monitor-incident-heading";
    const title = document.createElement("strong"); title.textContent = `${event.component || "runtime"} · ${event.event || "status"}`;
    const time = document.createElement("time"); time.dateTime = event.ts || ""; time.textContent = formatIsoDate(event.ts);
    heading.append(title, time);
    const meta = document.createElement("small"); meta.textContent = `${event.state || "unknown"} · restarts ${event.restart_count || 0}${event.next_retry_seconds == null ? "" : ` · retry ${event.next_retry_seconds}s`}`;
    row.append(heading, meta);
    if (event.message) { const detail = document.createElement("p"); detail.textContent = event.message; row.append(detail); }
    elements.monitorIncidentList.append(row);
  });
}

function renderMonitorLogs(logs) {
  const entries = Array.isArray(logs.entries) ? logs.entries : [];
  state.monitorLogs = entries;
  elements.monitorLogList.replaceChildren();
  elements.monitorLogCount.textContent = `${entries.length}/${logs.captured || entries.length}`;
  elements.monitorLogsEmpty.classList.toggle("is-hidden", entries.length > 0);
  entries.forEach((entry) => {
    const row = document.createElement("div"); row.className = `monitor-log-row level-${String(entry.level || "INFO").toLowerCase()}`;
    const meta = document.createElement("div"); meta.className = "monitor-log-meta";
    const level = document.createElement("span"); level.className = "monitor-log-level"; level.textContent = entry.level || "INFO";
    const time = document.createElement("time"); time.dateTime = entry.ts || ""; time.textContent = formatIsoDate(entry.ts);
    const source = document.createElement("span"); source.textContent = entry.source || "runtime"; meta.append(level, time, source);
    const message = document.createElement("p"); message.textContent = entry.message || "";
    row.append(meta, message); elements.monitorLogList.append(row);
  });
}

function renderMonitor(payload, latencyMs = 0) {
  const process = payload.process || {}, workers = payload.workers || {}, queue = payload.queue || {}, tts = payload.tts || {}, workloads = payload.workloads || {};
  elements.monitorLiveState.classList.remove("down", "paused");
  elements.monitorLiveState.lastElementChild.textContent = state.monitorPaused ? t("paused") : t("live");
  elements.monitorLiveState.classList.toggle("paused", state.monitorPaused);
  elements.monitorProcessState.textContent = payload.health?.state === "critical" ? t("unavailable") : t("online");
  elements.monitorProcessMeta.textContent = `PID ${process.pid ?? "—"} · ${process.uptime || formatDuration(process.uptime_seconds)}`;
  elements.monitorWorkerState.textContent = `${workers.alive || 0}/${workers.count || 0}`;
  elements.monitorWorkerMeta.textContent = workers.accepting ? t("accepting") : t("drained");
  const ttsInUse = tts.in_use == null ? tts.running_count || 0 : tts.in_use;
  elements.monitorTtsState.textContent = `${ttsInUse}/${tts.configured || 0}`;
  elements.monitorTtsMeta.textContent = `${tts.running_count || 0} ${t("running")} · ${tts.queued_count || 0} ${t("waiting")}`;
  elements.monitorQueueState.textContent = `${queue.running || 0} ${t("running")}`;
  elements.monitorQueueMeta.textContent = `${queue.queued || 0} ${t("waiting")} · ${Number(queue.throughput_per_minute || 0).toFixed(1)}/min`;
  elements.monitorUpdated.textContent = formatIsoDate(payload.generated_at);
  elements.heroSyncState.textContent = `${t("lastSync")} · ${new Date(payload.generated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
  elements.monitorProcessDetails.replaceChildren();
  [
    ["Instance", process.instance_id || "—"], ["PID", String(process.pid ?? "—")], [t("uptime"), process.uptime || formatDuration(process.uptime_seconds)],
    ["Threads", String(process.threads ?? "—")], ["Max RSS", formatMemory(process.max_rss_kb)], ["Load average", Array.isArray(process.load_average) && process.load_average.length ? process.load_average.join(" · ") : "—"],
    ["Active web requests", String(process.active_requests ?? 0)], ["DB executor queue", String(process.db_queue_size ?? 0)], ["TTS completed", compactNumber(process.metrics?.tts || 0)],
    [t("publicUrl"), payload.public_url?.url || "Not detected"], ["URL source", payload.public_url?.source || "—"],
  ].forEach(([label, value]) => appendMonitorDetail(elements.monitorProcessDetails, label, value));
  renderMonitorAlert(payload, latencyMs); updateMonitorTrends(payload); renderMonitorWorkers(workers); renderMonitorIncidents(payload.incidents || {}); renderMonitorTts(tts, workloads); renderMonitorLogs(payload.logs || {});
}

async function refreshMonitor({ force = false, silent = true } = {}) {
  if (state.monitorRefreshing || (state.monitorPaused && !force) || (!state.monitorVisible && !force) || document.hidden) return;
  state.monitorRefreshing = true; elements.refreshMonitorButton.disabled = true;
  const params = new URLSearchParams({ log_limit: "120" });
  if (elements.monitorLogLevel.value) params.set("log_level", elements.monitorLogLevel.value);
  if (elements.monitorLogSearch.value.trim()) params.set("log_query", elements.monitorLogSearch.value.trim());
  try {
    const requestStarted = performance.now();
    const payload = await api(`/api/admin/runtime/monitor?${params}`);
    renderMonitor(payload, performance.now() - requestStarted);
  } catch (error) {
    elements.monitorLiveState.classList.add("down"); elements.monitorLiveState.lastElementChild.textContent = t("unavailable");
    elements.monitorAlert.classList.remove("healthy", "warning", "critical"); elements.monitorAlert.classList.add("down");
    elements.monitorAlertTitle.textContent = t("monitorOffline"); elements.monitorAlertDetail.textContent = t("monitorOfflineHelp"); elements.monitorLatency.textContent = "— ms";
    if (!silent) showToast(error.message, true);
  } finally { state.monitorRefreshing = false; elements.refreshMonitorButton.disabled = false; }
}

function toggleMonitorPause() {
  state.monitorPaused = !state.monitorPaused;
  elements.monitorPauseButton.textContent = state.monitorPaused ? t("resumeLive") : t("pauseLive");
  elements.monitorLiveState.classList.toggle("paused", state.monitorPaused);
  elements.monitorLiveState.lastElementChild.textContent = state.monitorPaused ? t("paused") : t("live");
  if (!state.monitorPaused) runAsync(() => refreshMonitor({ force: true }));
}

function toggleMonitorFullscreen() {
  state.monitorFullscreen = !state.monitorFullscreen;
  elements.monitor.classList.toggle("is-monitor-fullscreen", state.monitorFullscreen);
  document.body.classList.toggle("monitor-fullscreen-open", state.monitorFullscreen);
  elements.monitorFullscreenButton.textContent = state.monitorFullscreen ? t("exitFullScreen") : t("fullScreen");
  if (state.monitorFullscreen) { telegram?.expand?.(); elements.monitor.scrollTop = 0; runAsync(() => refreshMonitor({ force: true })); }
}

function scheduleMonitorPolling() {
  window.clearInterval(state.monitorTimer);
  state.monitorTimer = window.setInterval(() => runAsync(() => refreshMonitor()), state.monitorIntervalMs);
}

async function copyMonitorLogs() {
  if (!state.monitorLogs.length) return showToast(t("noLogs"), true);
  const text = state.monitorLogs.map((entry) => `[${entry.ts || ""}] ${entry.level || "INFO"} ${entry.source || "runtime"} — ${entry.message || ""}`).join("\n");
  try {
    await navigator.clipboard.writeText(text);
    haptic(); showToast(t("logsCopied"));
  } catch { showToast("Could not copy logs.", true); }
}

async function downloadMonitorLogs() {
  const initData = String(window.Telegram?.WebApp?.initData || "").trim();
  if (!initData) throw new Error("Open this dashboard from the bot inside Telegram.");
  const params = new URLSearchParams({ log_limit: "400" });
  if (elements.monitorLogLevel.value) params.set("log_level", elements.monitorLogLevel.value);
  if (elements.monitorLogSearch.value.trim()) params.set("log_query", elements.monitorLogSearch.value.trim());
  elements.monitorDownloadLogsButton.disabled = true;
  try {
    const response = await fetch(`/api/admin/runtime/monitor/logs/download?${params}`, {
      headers: { "X-Telegram-Init-Data": initData, "Accept": "text/plain" },
      credentials: "same-origin",
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`Log download failed (${response.status})`);
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1] || "bot-runtime.log";
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; document.body.append(anchor); anchor.click(); anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    haptic(); showToast(t("logsDownloaded"));
  } finally {
    elements.monitorDownloadLogsButton.disabled = false;
  }
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
  const queueMode = jobsPayload?.queue || payload.runtime?.queue || {};
  const redisDisabled = redis.status === "disabled";
  elements.totalUsers.textContent = compactNumber(usage.total_users);
  elements.messageCount.textContent = compactNumber(usage.message_count);
  elements.queuedJobs.textContent = compactNumber(jobs.queued);
  elements.deadJobs.textContent = compactNumber(jobs.dead);
  elements.redisLatency.textContent = redisDisabled ? t("disabled") : (redis.ok && redis.latency_ms != null ? `${redis.latency_ms} ms` : "Offline");
  elements.uptime.textContent = bot.uptime || t("starting");
  elements.botMode.textContent = bot.mode || "—";
  elements.lastUpdated.textContent = new Date(payload.generated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  setHealth(elements.botHealth, bot.active ? t("operational") : t("starting"), bot.active ? "ok" : "warn");
  setHealth(elements.redisHealth, redisDisabled ? t("disabled") : (redis.ok ? t("healthy") : t("unavailable")), redisDisabled ? "warn" : (redis.ok ? "ok" : "down"));
  setHealth(elements.databaseHealth, database.ok ? t("connected") : (database.memory_fallback ? t("memoryFallback") : t("unavailable")), database.ok ? "ok" : "warn");
  setHealth(elements.workerHealth, workers.healthy ? `${workers.alive}/${workers.count} · ${workers.accepting ? t("accepting") : t("drained")}` : `${workers.alive || 0}/${workers.count || 0} ${t("unavailable")}`, workers.healthy ? (workers.accepting ? "ok" : "warn") : "down");
  elements.drainWorkersButton.disabled = !workers.accepting;
  elements.resumeWorkersButton.disabled = Boolean(workers.accepting);
  const overallHealthy = Boolean(bot.active && (redis.ok || redisDisabled) && workers.healthy);
  elements.systemBadge.classList.toggle("down", !overallHealthy);
  elements.systemBadge.lastElementChild.textContent = overallHealthy ? t("operational") : t("unavailable");
  renderMaintenance(Boolean(bot.maintenance_mode));
  const queueLabel = queueMode.durable === false ? `${t("processLocal")} · ` : "";
  elements.jobSummary.textContent = `${queueLabel}Queued ${jobs.queued || 0} · Running ${jobs.running || 0} · Dead ${jobs.dead || 0} · Succeeded ${jobs.succeeded || 0} · Cancelled ${jobs.cancelled || 0} · Capacity ${jobs.queue_available ?? "—"}`;
  renderQueueCharts(jobs);
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

function appendJobDetail(labelText, valueText) {
  const row = document.createElement("div"); row.className = "job-detail-row";
  const label = document.createElement("span"); label.textContent = labelText;
  const value = document.createElement("strong"); value.textContent = valueText == null || valueText === "" ? "—" : String(valueText);
  row.append(label, value); elements.jobDetailContent.append(row);
}
function renderJobDetail(job) {
  elements.jobDetailTitle.textContent = `${job.type} · ${job.id.slice(0, 12)}`;
  elements.jobDetailContent.replaceChildren();
  [
    ["Job ID", job.id], ["State", job.state], ["Priority", job.priority],
    ["Attempts", `${job.attempts}/${job.max_attempts}`], ["Worker", job.worker_id],
    ["Created", formatDate(job.created_at)], ["Started", formatDate(job.started_at)],
    ["Completed", formatDate(job.completed_at)], ["Updated", formatDate(job.updated_at)],
    ["Progress", `${job.progress_percent || 0}% · ${job.progress_stage || "—"}`],
    ["Detail", job.progress_detail], ["Last error", job.last_error],
  ].forEach(([label, value]) => appendJobDetail(label, value));
  if (job.result != null) {
    const result = document.createElement("div"); result.className = "job-detail-result";
    const label = document.createElement("span"); label.textContent = "Result";
    const content = document.createElement("pre"); content.textContent = JSON.stringify(job.result, null, 2);
    result.append(label, content); elements.jobDetailContent.append(result);
  }
}
async function openJobDetail(jobId, button) {
  button.disabled = true;
  try {
    const payload = await api(`/api/admin/runtime/jobs/${encodeURIComponent(jobId)}`);
    state.currentJobId = payload.job.id;
    renderJobDetail(payload.job);
    elements.jobDetailDialog.classList.remove("is-hidden");
    document.body.classList.add("dialog-open");
    elements.closeJobDetailButton.focus();
  } catch (error) {
    showToast(error.message, true);
  } finally { button.disabled = false; }
}
function closeJobDetail() {
  elements.jobDetailDialog.classList.add("is-hidden");
  document.body.classList.remove("dialog-open");
  state.currentJobId = "";
}

async function refreshJobs({ append = false } = {}) {
  if (append && (!state.jobCursor || state.jobAppendPending)) return;
  if (append) state.jobAppendPending = true;
  const selectedState = elements.jobState.value;
  const requestId = ++state.jobRequestId;
  elements.refreshJobsButton.disabled = true;
  elements.loadMoreJobsButton.disabled = true;
  const params = new URLSearchParams({ state: selectedState, limit: "50" });
  if (elements.jobType.value) params.set("job_type", elements.jobType.value);
  if (elements.jobSearch.value.trim()) params.set("query", elements.jobSearch.value.trim());
  if (append && state.jobCursor) params.set("cursor", state.jobCursor);
  try {
    const payload = await api(`/api/admin/runtime/jobs/list?${params}`);
    if (requestId !== state.jobRequestId) return;
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
      const details = document.createElement("button"); details.className = "button small"; details.textContent = t("details"); details.addEventListener("click", () => openJobDetail(job.id, details)); actions.append(details);
      if (["dead", "cancelled"].includes(job.state)) { const retry = document.createElement("button"); retry.className = "button small"; retry.textContent = t("retry"); retry.addEventListener("click", () => mutateJob(job.id, "retry", retry)); actions.append(retry); }
      if (["queued", "running"].includes(job.state)) { const cancel = document.createElement("button"); cancel.className = "button danger"; cancel.textContent = t("cancel"); cancel.addEventListener("click", () => mutateJob(job.id, "cancel", cancel)); actions.append(cancel); }
      row.append(actions); elements.jobsList.append(row);
    });
    state.jobCursor = payload.next_cursor; elements.loadMoreJobsButton.classList.toggle("is-hidden", !state.jobCursor);
    elements.jobsEmpty.classList.toggle("is-hidden", elements.jobsList.children.length > 0);
    elements.retrySelectedButton.disabled = state.selectedJobs.size === 0;
  } finally {
    if (append) state.jobAppendPending = false;
    if (requestId === state.jobRequestId) {
      elements.refreshJobsButton.disabled = false;
      elements.loadMoreJobsButton.disabled = !state.jobCursor;
    }
  }
}

async function setWorkerAcceptance(action, button) {
  if (action === "drain" && !await confirmAction("Stop workers from accepting new jobs? Running jobs will continue.")) return;
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
  if (action === "cancel" && !await confirmAction(`Cancel job ${jobId.slice(0, 12)}?`)) return;
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
async function resetProvider(name, button) { if (!await confirmAction(`Reset health and cooldown state for ${name}?`)) return; button.disabled = true; try { await api("/api/admin/runtime/providers/reset", { method: "POST", body: JSON.stringify({ provider: name }) }); showToast(`${name} reset.`); await refreshProviders(); } catch (error) { showToast(error.message, true); button.disabled = false; } }
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
  if (action === "remove" && !await confirmAction(`Remove Telegram administrator ${userId}?`)) return;
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
    const [profile, stats, settings, cors, jobs] = await Promise.all([api("/api/admin/me"), api("/api/admin/stats"), api("/api/admin/settings"), api("/api/admin/cors"), api("/api/admin/runtime/jobs")]);
    renderProfile(profile); renderStats(stats, jobs, jobs.workers); renderSettings(settings); renderCors(cors.origins);
    const optional = await Promise.allSettled([refreshJobs(), refreshProviders(), refreshAdministrators(), refreshMonitor({ force: true })]);
    optional.filter((item) => item.status === "rejected").forEach((item) => console.warn(item.reason));
    state.lastDashboardRefresh = Date.now();
    elements.authState.classList.add("is-hidden"); elements.dashboard.classList.remove("is-hidden"); if (!initial && !silent) { haptic(); showToast("Dashboard refreshed."); }
  } catch (error) {
    if (initial) { elements.authState.classList.add("error"); elements.authState.replaceChildren(); const copy = document.createElement("div"), title = document.createElement("strong"), detail = document.createElement("p"), retry = document.createElement("button"); title.textContent = "Unable to open dashboard"; detail.textContent = error.message; retry.type = "button"; retry.className = "button primary auth-retry"; retry.textContent = "Retry"; retry.addEventListener("click", () => { elements.authState.classList.remove("error"); retry.disabled = true; runAsync(() => refreshAll({ initial: true })); }); copy.append(title, detail, retry); elements.authState.append(copy); }
    else showToast(error.message, true);
  } finally { state.refreshing = false; elements.refreshButton.disabled = false; }
}

async function refreshOverview({ silent = true } = {}) {
  if (state.refreshing || state.overviewRefreshing) return;
  state.overviewRefreshing = true;
  try {
    const [stats, jobs] = await Promise.all([api("/api/admin/stats"), api("/api/admin/runtime/jobs")]);
    renderStats(stats, jobs, jobs.workers);
    state.lastDashboardRefresh = Date.now();
  } catch (error) {
    if (!silent) showToast(error.message, true);
  } finally { state.overviewRefreshing = false; }
}

function scheduleDashboardRefresh() {
  window.clearInterval(state.dashboardTimer);
  state.dashboardTimer = window.setInterval(() => {
    if (!document.hidden && navigator.onLine && !elements.dashboard.classList.contains("is-hidden")) runAsync(() => refreshOverview());
  }, DASHBOARD_REFRESH_MS);
}

async function updateMaintenance(nextValue) { elements.maintenanceToggle.disabled = true; try { const payload = await api("/api/admin/settings", { method: "POST", body: JSON.stringify({ maintenance_mode: nextValue }) }); renderMaintenance(Boolean(payload.maintenance_mode)); haptic("medium"); showToast("Maintenance setting updated."); } catch (error) { renderMaintenance(!nextValue); showToast(error.message, true); } finally { elements.maintenanceToggle.disabled = false; } }
async function confirmMaintenance(nextValue) { const message = nextValue ? "Pause bot features for normal users?" : "Resume bot features for normal users?"; if (await confirmAction(message)) await updateMaintenance(nextValue); else renderMaintenance(!nextValue); }
async function addOrigin(event) { event.preventDefault(); const origin = elements.corsOrigin.value.trim(); if (!origin) return; const submit = elements.corsForm.querySelector("button[type='submit']"); submit.disabled = true; try { const payload = await api("/api/admin/cors", { method: "POST", body: JSON.stringify({ origin }) }); elements.corsOrigin.value = ""; renderCors(payload.origins); showToast(payload.changed ? "Origin added." : "Origin already allowed."); } catch (error) { showToast(error.message, true); } finally { submit.disabled = false; } }
async function removeOrigin(origin, button) { if (!await confirmAction(`Remove allowed origin ${origin}?`)) return; button.disabled = true; try { const payload = await api("/api/admin/cors", { method: "DELETE", body: JSON.stringify({ origin }) }); renderCors(payload.origins); showToast("Origin removed."); } catch (error) { button.disabled = false; showToast(error.message, true); } }
async function saveRuntime(event) { event.preventDefault(); const runtime = {}; elements.runtimeFields.querySelectorAll("input[name]").forEach((input) => { runtime[input.name] = input.dataset.kind === "bool" ? input.checked : (input.dataset.kind === "int" ? Number.parseInt(input.value, 10) : Number.parseFloat(input.value)); }); if (Object.values(runtime).some(Number.isNaN)) return showToast("Enter valid values.", true); elements.saveRuntimeButton.disabled = true; try { const payload = await api("/api/admin/settings", { method: "POST", body: JSON.stringify({ runtime }) }); renderSettings(payload); showToast("Runtime settings saved."); } catch (error) { showToast(error.message, true); } finally { elements.saveRuntimeButton.disabled = false; } }

function applyTelegramTheme() { const root = document.documentElement, params = telegram?.themeParams || {}; Object.entries({ "--tg-bg": params.bg_color, "--tg-text": params.text_color }).forEach(([name, value]) => { if (value) root.style.setProperty(name, value); }); updateViewportVars(); }
function updateViewportVars() { const root = document.documentElement, viewportHeight = telegram?.viewportHeight || window.innerHeight, stableHeight = telegram?.viewportStableHeight || viewportHeight; root.style.setProperty("--tg-viewport-height", `${viewportHeight}px`); root.style.setProperty("--tg-viewport-stable-height", `${stableHeight}px`); }
function initializeTelegram() { if (!telegram) return; telegram.ready(); telegram.expand(); applyTelegramTheme(); telegram.onEvent?.("themeChanged", applyTelegramTheme); telegram.onEvent?.("viewportChanged", updateViewportVars); }

const navigationItems = [...document.querySelectorAll("[data-nav-target]")];
function activateNavigation(target) { navigationItems.forEach((item) => item.classList.toggle("active", item.dataset.navTarget === target)); }
navigationItems.forEach((item) => item.addEventListener("click", () => { activateNavigation(item.dataset.navTarget); haptic(); }));
elements.refreshButton.addEventListener("click", () => refreshAll());
elements.languageButton.addEventListener("click", () => { state.language = state.language === "en" ? "km" : "en"; applyLanguage(); refreshAll({ silent: true }); });
elements.maintenanceToggle.addEventListener("change", (event) => confirmMaintenance(event.target.checked));
elements.corsForm.addEventListener("submit", addOrigin); elements.runtimeForm.addEventListener("submit", saveRuntime);
elements.jobState.addEventListener("change", () => runAsync(() => refreshJobs())); elements.jobType.addEventListener("change", () => runAsync(() => refreshJobs())); elements.refreshJobsButton.addEventListener("click", () => runAsync(() => refreshJobs()));
let jobSearchTimer = null; elements.jobSearch.addEventListener("input", () => { window.clearTimeout(jobSearchTimer); jobSearchTimer = window.setTimeout(() => runAsync(() => refreshJobs()), 250); });
elements.drainWorkersButton.addEventListener("click", () => setWorkerAcceptance("drain", elements.drainWorkersButton));
elements.resumeWorkersButton.addEventListener("click", () => setWorkerAcceptance("resume", elements.resumeWorkersButton));
elements.loadMoreJobsButton.addEventListener("click", () => runAsync(() => refreshJobs({ append: true }))); elements.retrySelectedButton.addEventListener("click", retrySelected);
elements.refreshMonitorButton.addEventListener("click", () => runAsync(() => refreshMonitor({ force: true, silent: false })));
elements.monitorPauseButton.addEventListener("click", toggleMonitorPause);
elements.monitorFullscreenButton.addEventListener("click", toggleMonitorFullscreen);
elements.monitorCopyLogsButton.addEventListener("click", () => runAsync(copyMonitorLogs));
elements.monitorDownloadLogsButton.addEventListener("click", () => runAsync(downloadMonitorLogs));
elements.monitorInterval.addEventListener("change", () => {
  const interval = Number.parseInt(elements.monitorInterval.value, 10);
  if (!MONITOR_INTERVALS.has(interval)) return;
  state.monitorIntervalMs = interval; window.localStorage.setItem("admin-monitor-interval", String(interval)); scheduleMonitorPolling();
  runAsync(() => refreshMonitor({ force: true }));
});
document.querySelectorAll("[data-tts-filter]").forEach((button) => button.addEventListener("click", () => {
  state.monitorTtsFilter = button.dataset.ttsFilter || "all";
  if (state.monitorTtsPayload) renderMonitorTts(state.monitorTtsPayload.tts, state.monitorTtsPayload.workloads);
}));
elements.monitorWorkloadType.addEventListener("change", () => {
  state.monitorWorkloadType = elements.monitorWorkloadType.value || "all";
  if (state.monitorTtsPayload) renderMonitorTts(state.monitorTtsPayload.tts, state.monitorTtsPayload.workloads);
});
elements.monitorLogLevel.addEventListener("change", () => runAsync(() => refreshMonitor({ force: true, silent: false })));
let monitorSearchTimer = null; elements.monitorLogSearch.addEventListener("input", () => { window.clearTimeout(monitorSearchTimer); monitorSearchTimer = window.setTimeout(() => runAsync(() => refreshMonitor({ force: true })), 300); });
document.addEventListener("keydown", (event) => { if (event.key !== "Escape") return; if (!elements.jobDetailDialog.classList.contains("is-hidden")) closeJobDetail(); else if (state.monitorFullscreen) toggleMonitorFullscreen(); });
document.addEventListener("visibilitychange", () => { if (document.hidden) return; if (!state.monitorPaused) runAsync(() => refreshMonitor()); if (navigator.onLine && Date.now() - state.lastDashboardRefresh >= DASHBOARD_REFRESH_MS) runAsync(() => refreshOverview()); });
window.addEventListener("offline", () => renderConnectionState("offline"));
window.addEventListener("online", () => { renderConnectionState("reconnecting"); runAsync(() => elements.dashboard.classList.contains("is-hidden") ? refreshAll({ initial: true }) : refreshOverview({ silent: false })); });
document.querySelectorAll("[data-close-job-detail]").forEach((button) => button.addEventListener("click", closeJobDetail));
elements.closeJobDetailButton.addEventListener("click", closeJobDetail);
elements.copyJobIdButton.addEventListener("click", () => runAsync(async () => { if (!state.currentJobId) return; await copyText(state.currentJobId); haptic(); showToast(t("copied")); }));
elements.adminForm.addEventListener("submit", async (event) => { event.preventDefault(); const userId = Number.parseInt(elements.adminUserId.value, 10); if (!Number.isSafeInteger(userId) || userId <= 0) return showToast("Enter a valid Telegram user ID.", true); const button = elements.adminForm.querySelector("button"); await mutateAdministrator("add", userId, button); elements.adminUserId.value = ""; });

applyLanguage(); initializeTelegram(); scheduleDashboardRefresh(); refreshAll({ initial: true });
if ("IntersectionObserver" in window) {
  const visibleSections = new Map();
  const navigationObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) visibleSections.set(entry.target.id, entry.intersectionRatio);
      else visibleSections.delete(entry.target.id);
    });
    const active = [...visibleSections.entries()].sort((left, right) => right[1] - left[1])[0]?.[0];
    if (active) activateNavigation(active);
  }, { rootMargin: "-18% 0px -62% 0px", threshold: [0.05, 0.25, 0.5] });
  navigationItems.map((item) => document.getElementById(item.dataset.navTarget)).filter(Boolean).forEach((section) => navigationObserver.observe(section));
  const monitorObserver = new IntersectionObserver((entries) => {
    state.monitorVisible = entries.some((entry) => entry.isIntersecting);
    if (state.monitorVisible && !state.monitorPaused) runAsync(() => refreshMonitor());
  }, { rootMargin: "200px 0px" });
  monitorObserver.observe(elements.monitor);
} else state.monitorVisible = true;
scheduleMonitorPolling();
