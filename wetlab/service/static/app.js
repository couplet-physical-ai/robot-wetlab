/* Virtual wet lab console: vanilla JS, no build step. API shapes: wetlab/service/app.py. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const KEY_STORE = "wetlab.key";
const SLOT_S = 900;
const TERMINAL = ["SUCCEEDED", "FAILED", "ABORTED", "CANCELLED", "COMPILE_FAILED", "DRY_RUN_FAILED", "DRY_RUN_OK"];
const JOINTS = ["base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"];

const PY_SAMPLE = `# Python protocol using the wetlab SDK. It only RECORDS steps; the service
# compiles it in a no-network sandbox, dry-runs it in the twin, then runs it.
from wetlab import protocol

p = protocol("demo: shake plate-1, image it, put it back")
p.home()
p.pick("plate-1")
p.place("plate-1", "shaker-1")
p.shake("shaker-1", rpm=600, seconds=3)
p.capture("after-shake")
p.pick("plate-1")
p.place("plate-1", "nest-A1")
p.move_to(labware="plate-2", well="A1", z_above=0.03)
p.home()
`;
const IR_SAMPLE = JSON.stringify({
  ir_version: "wetlab.ir/1", name: "demo: shake plate-1, image it, put it back",
  steps: [
    { op: "home" }, { op: "pick", labware: "plate-1" }, { op: "place", labware: "plate-1", at: "shaker-1" },
    { op: "instrument", instrument: "shaker-1", action: "shake", params: { seconds: 3, rpm: 600 } },
    { op: "capture", tag: "after-shake" }, { op: "pick", labware: "plate-1" },
    { op: "place", labware: "plate-1", at: "nest-A1" },
    { op: "move_to", target: { labware: "plate-2", well: "A1", z_above: 0.03 } }, { op: "home" },
  ],
}, null, 2);

const S = {
  key: null, me: null, cell: null, snap: null, bookings: [], jobs: [], avail: null, sel: null,
  tab: "book", kind: "python", sources: { python: PY_SAMPLE, ir: IR_SAMPLE },
  token: null, tokenExp: 0, ws: null, wsLive: false, streaming: false,
  jobId: null, jobSig: "", blobs: new Map(), timers: [],
};

// ---------- helpers -------------------------------------------------------------------
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const now = () => Date.now() / 1000;
const pad = (n) => String(n).padStart(2, "0");
const hm = (ts) => { const d = new Date(ts * 1000); return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`; };
const dayLabel = (ts) => new Date(ts * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
const ago = (ts) => { const s = Math.max(0, now() - ts); return s < 60 ? `${s | 0}s ago` : s < 3600 ? `${(s / 60) | 0}m ago` : s < 86400 ? `${(s / 3600) | 0}h ago` : dayLabel(ts); };
const money = (x) => `$${(+x || 0).toFixed(2)}`;
const secs = (s) => (s == null ? "—" : s < 120 ? `${(+s).toFixed(1)} s` : `${(s / 60).toFixed(1)} min`);
const bytes = (n) => (n < 1024 ? `${n} B` : n < 1 << 20 ? `${(n / 1024).toFixed(0)} KB` : `${(n / (1 << 20)).toFixed(1)} MB`);
const shortId = (id) => (id ? id.replace(/^(job|bk)_/, "") : "—");
const pill = (status) => `<span class="pill s-${esc(String(status).toLowerCase())}">${esc(String(status).replaceAll("_", " "))}</span>`;
const isStaff = () => S.me && ["operator", "admin"].includes(S.me.role);

function toast(msg, kind = "info") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  el.textContent = msg;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 6000 : 3500);
}

async function api(path, opts = {}) {
  const r = await fetch(path, {
    ...opts,
    headers: { Authorization: `Bearer ${S.key}`, ...(opts.body ? { "Content-Type": "application/json" } : {}), ...(opts.headers || {}) },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (r.status === 401 && path !== "/v1/me") { signOut("Your API key was rejected."); throw new Error("unauthorized"); }
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : r;
  if (!r.ok) {
    const d = data && data.detail;
    const e = new Error(typeof d === "string" ? d : d && d.message ? `${d.message}: ${(d.faults || []).join(", ")}` : `HTTP ${r.status}`);
    e.status = r.status; e.detail = d;
    throw e;
  }
  return data;
}

async function blobUrl(url, size) {   // artifacts need the Authorization header, so <video>/<img> get blob: URLs
  const k = `${url}#${size}`;
  if (!S.blobs.has(k)) S.blobs.set(k, fetch(url, { headers: { Authorization: `Bearer ${S.key}` } })
    .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.blob(); }).then((b) => URL.createObjectURL(b)));
  return S.blobs.get(k);
}

async function download(url, name) {
  try {
    const href = await blobUrl(url, "dl");
    const a = Object.assign(document.createElement("a"), { href, download: name });
    document.body.append(a); a.click(); a.remove();
  } catch (e) { toast(`Download failed: ${e.message}`, "error"); }
}

// ---------- boot / auth -----------------------------------------------------------------
function readHash() {   // #key=…&tab=…&job=… (read once, then stripped from the URL)
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("key")) localStorage.setItem(KEY_STORE, h.get("key"));
  if (h.get("tab")) S.tab = h.get("tab");
  if (h.get("job")) S.jobId = h.get("job");
  if (location.hash) history.replaceState(null, "", location.pathname + location.search);
}

async function boot() {
  readHash();
  S.key = localStorage.getItem(KEY_STORE);
  if (!S.key) return showSignin();
  try {
    S.me = await api("/v1/me");
  } catch (e) {
    return showSignin(e.status === 401 ? "That API key is not valid." : `Could not reach the service: ${e.message}`);
  }
  const cells = await api("/v1/cells");
  S.cell = cells[0];
  $("#signin").hidden = true;
  $("#app").hidden = false;
  $("#cell-name").textContent = `${S.cell.name}`;
  $("#cell-name").title = `${S.cell.id} · spec ${S.cell.lab_sha.slice(0, 10)}`;
  $("#who").innerHTML = `${esc(S.me.name)} <span class="tag">${esc(S.me.role)}</span>`;
  $("#operator").hidden = !isStaff();
  $("#cam-meta").textContent = `${S.cell.camera.id} · ${S.cell.camera.width}×${S.cell.camera.height} · ${S.cell.camera.fps} fps`;
  $("#cam-frame").style.aspectRatio = `${S.cell.camera.width} / ${S.cell.camera.height}`;
  $("#book-date").value = new Date().toISOString().slice(0, 10);
  $("#source").value = S.sources[S.kind];
  setTab(S.tab);
  await refreshBookings();
  refreshJobs();
  liveLoop();
  S.timers.push(setInterval(liveLoop, 15000));
  S.timers.push(setInterval(pollState, 1000));
  S.timers.push(setInterval(tickTab, 2500));
}

function showSignin(msg) {
  $("#app").hidden = true;
  $("#signin").hidden = false;
  $("#signin-error").hidden = !msg;
  $("#signin-error").textContent = msg || "";
  $("#key-input").focus();
}

function signOut(msg) {
  localStorage.removeItem(KEY_STORE);
  S.timers.forEach(clearInterval); S.timers = [];
  if (S.ws) S.ws.close();
  S.ws = null; S.token = null; S.streaming = false;
  $("#cam-img").removeAttribute("src");
  showSignin(msg);
}

$("#signin-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  localStorage.setItem(KEY_STORE, $("#key-input").value.trim());
  $("#key-input").value = "";
  boot();
});
$("#signout").addEventListener("click", () => signOut());

// ---------- live: stream token, MJPEG, WebSocket ------------------------------------------
async function liveLoop() {
  if (S.token && S.tokenExp - now() > 60) return;   // still fresh; refresh a minute before expiry
  try {
    const t = await api("/v1/stream-tokens", { method: "POST", body: { cell: S.cell.id } });
    S.token = t.token; S.tokenExp = t.expires;
    startStream();
    connectWs();
  } catch (e) {
    if (e.status === 403) stopStream(); else if (e.message !== "unauthorized") stopStream("Camera unavailable", e.message);
  }
}

function startStream() {
  const img = $("#cam-img");
  img.onerror = () => { S.token = null; stopStream("Stream ended", "Reconnecting…"); setTimeout(liveLoop, 2000); };
  img.onload = () => { $("#cam-placeholder").hidden = true; $("#cam-live").hidden = false; };
  img.src = `/v1/cells/${encodeURIComponent(S.cell.id)}/stream.mjpg?token=${encodeURIComponent(S.token)}`;
  img.hidden = false;
  $("#cam-placeholder").hidden = true;
  $("#cam-live").hidden = false;
  $("#cam-note").textContent = `MJPEG · token renews before ${hm(S.tokenExp)} UTC`;
  S.streaming = true;
}

function stopStream(title = "No active booking", sub = "The live camera opens during your booked slot.") {
  S.streaming = false; S.token = null;
  const img = $("#cam-img");
  img.onerror = null;
  img.removeAttribute("src");
  img.hidden = true;
  $("#cam-live").hidden = true;
  $("#cam-note").textContent = "offline";
  $("#ph-title").textContent = title;
  $("#ph-sub").textContent = sub;
  $("#cam-placeholder").hidden = false;
  if (S.ws) { S.ws.onclose = null; S.ws.close(); S.ws = null; S.wsLive = false; }
}

function connectWs() {
  if (S.ws) { S.ws.onclose = null; S.ws.close(); }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/v1/cells/${encodeURIComponent(S.cell.id)}/ws?token=${encodeURIComponent(S.token)}`);
  S.ws = ws;
  ws.onmessage = (m) => { S.wsLive = true; S.snap = JSON.parse(m.data); renderState("live · WebSocket 10 Hz"); };
  ws.onclose = () => { S.wsLive = false; if (S.ws === ws) { S.ws = null; setTimeout(liveLoop, 2000); } };
}

async function pollState() {   // REST fallback when there is no WebSocket (no active booking)
  if (S.wsLive || !S.cell) return;
  try { S.snap = await api(`/v1/cells/${S.cell.id}/state`); renderState("polled · 1 Hz"); } catch { /* transient */ }
}

// ---------- state rendering ---------------------------------------------------------------
function chip(label, cls, title = "") { return `<span class="chip ${cls}" title="${esc(title)}"><i></i>${esc(label)}</span>`; }

function renderState(src) {
  const s = S.snap;
  if (!s) return;
  const d = s.door, il = s.interlock;
  const mine = S.bookings.find((b) => b.id === s.booking_active && b.user_id === S.me.id);
  const chips = [
    !d.closed ? chip(`Door open ${d.angle_deg.toFixed(0)}°`, "bad") : d.locked ? chip("Door closed · locked", "ok") : chip("Door closed · unlocked", "warn"),
    il.tripped ? chip(`Interlock tripped · ${il.reason}`, "bad", "Latched until an operator resets it")
      : il.faults.length ? chip(`Interlock · ${il.faults.join(", ")}`, "warn", "Blocks job start") : chip("Interlock clear", "ok"),
    s.arm_safety === "normal" ? chip("Arm normal", "ok") : chip(`Arm ${s.arm_safety.replaceAll("_", " ")}`, "bad"),
    s.booking_active ? (mine || isStaff() ? chip(mine ? "Your booking active" : "Booking active", "ok") : chip("Cell booked", "neutral"))
      : chip("No active booking", "neutral"),
  ];
  $("#chips").innerHTML = chips.join("");
  $("#state-src").textContent = src;

  const j = s.job;
  $("#job-live").innerHTML = j
    ? `<div class="row between"><span>${pill("RUNNING")} ${j.id ? `<a href="#" data-job="${esc(j.id)}">${esc(shortId(j.id))}</a> · step ${j.step ?? "—"}` : "another renter's job"}</span><b>${Math.round(j.progress * 100)}%</b></div>
       <div class="bar"><span style="width:${(j.progress * 100).toFixed(1)}%"></span></div>`
    : `<div class="row between"><span>${il.tripped ? pill("TRIPPED") : pill("IDLE")} <span class="muted">${il.tripped ? "arm stopped; operator reset needed" : "no job running"}</span></span></div>`;

  $("#joints").innerHTML = `<thead><tr>${JOINTS.map((n) => `<th>${n}</th>`).join("")}</tr></thead>
    <tbody><tr>${s.q.map((r) => `<td>${((r * 180) / Math.PI).toFixed(1)}°</td>`).join("")}</tr></tbody>`;
  const labware = Object.entries(s.labware_at || {}).map(([k, v]) => `${k} @ ${v}`).join(" · ") || "—";
  $("#state-kv").innerHTML = [
    ["Gripper", `${(+s.gripper_mm).toFixed(1)} mm`], ["Held", s.held || "—"], ["Door", `${d.angle_deg.toFixed(1)}°`],
    ["Labware", labware], ["Cell time", `${hm(s.t)}:${pad(new Date(s.t * 1000).getUTCSeconds())} UTC`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");

  if (isStaff()) {
    $("#reset").disabled = !il.tripped && !il.faults.includes("estop");
    $("#op-msg").textContent = il.tripped ? `Latched: ${il.reason}. Close + lock the door, then reset.${il.faults.length ? ` Open faults: ${il.faults.join(", ")}.` : ""}`
      : il.faults.length ? `Start blocked by: ${il.faults.join(", ")}` : "All clear.";
  }
}

// ---------- tabs ----------------------------------------------------------------------------
function setTab(t) {
  if (!["book", "submit", "jobs", "usage"].includes(t)) t = "book";
  S.tab = t;
  $$(".tabs [data-tab]").forEach((b) => { b.classList.toggle("on", b.dataset.tab === t); b.setAttribute("aria-selected", b.dataset.tab === t); });
  $$(".tabpane").forEach((p) => { p.hidden = p.id !== `tab-${t}`; });
  tickTab(true);
}
$$(".tabs [data-tab]").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
document.addEventListener("click", (ev) => {
  const g = ev.target.closest("[data-goto]");
  if (g) { ev.preventDefault(); setTab(g.dataset.goto); }
  const j = ev.target.closest("[data-job]");
  if (j) { ev.preventDefault(); openJob(j.dataset.job); }
});

function tickTab(first = false) {
  if (!S.me) return;
  if (S.tab === "book") { if (first) refreshAvail(); }
  else if (S.tab === "jobs") { refreshJobs(); if (S.jobId) refreshJob(); }
  else if (S.tab === "usage") { if (first) refreshUsage(); }
  else if (S.tab === "submit" && first) renderBookingSelect();
}

// ---------- book ----------------------------------------------------------------------------
$("#book-date").addEventListener("change", () => { S.sel = null; refreshAvail(); });

async function refreshAvail() {
  try {
    S.avail = await api(`/v1/cells/${S.cell.id}/availability?date=${$("#book-date").value}`);
  } catch (e) { return toast(e.message, "error"); }
  renderSlots(true);
}

function renderSlots(scroll = false) {
  const a = S.avail, sel = S.sel;
  const t = now();
  let html = "";
  for (let h = 0; h < 24; h++) {
    html += `<div class="hr"><span class="hl">${pad(h)}:00</span>`;
    for (let q = 0; q < 4; q++) {
      const i = h * 4 + q, s = a.slots[i];
      const cls = s.mine ? "mine" : s.free ? "free" : s.end_ts <= t ? "past" : "taken";
      const on = sel && i >= sel.a && i <= sel.b ? " sel" : "";
      const nowc = s.start_ts <= t && t < s.end_ts ? " now" : "";
      html += `<button type="button" class="slot ${cls}${on}${nowc}" data-i="${i}" ${s.free ? "" : "disabled"} title="${hm(s.start_ts)}–${hm(s.end_ts)} UTC">${pad(h)}:${pad(q * 15)}</button>`;
    }
    html += "</div>";
  }
  const box = $("#slots");
  box.innerHTML = html;
  if (scroll) {
    const cur = $(".slot.now", box) || $(".slot.free", box);
    if (cur) box.scrollTop = Math.max(0, cur.parentElement.offsetTop - box.offsetTop - 4);
  }
  const rate = (S.cell.billing && S.cell.billing.slot_minute) || 0;
  if (sel) {
    const mins = ((sel.b - sel.a + 1) * SLOT_S) / 60;
    $("#book-sel").textContent = `${hm(a.slots[sel.a].start_ts)}–${hm(a.slots[sel.b].end_ts)} UTC · ${mins} min · ${money(mins * rate)}`;
  } else $("#book-sel").textContent = "Click a start slot, then an end slot.";
  $("#book-btn").disabled = !sel;
}

$("#slots").addEventListener("click", (ev) => {
  const b = ev.target.closest(".slot");
  if (!b || b.disabled) return;
  const i = +b.dataset.i, sl = S.avail.slots, sel = S.sel;
  const runFree = (x, y) => sl.slice(x, y + 1).every((s) => s.free);
  if (sel && sel.a === sel.b && i > sel.a && i - sel.a < 16 && runFree(sel.a, i)) S.sel = { a: sel.a, b: i };   // max 4 h
  else if (sel && sel.a === i && sel.b === i) S.sel = null;
  else S.sel = { a: i, b: i };
  renderSlots();
});

$("#book-btn").addEventListener("click", async () => {
  const sl = S.avail.slots, sel = S.sel;
  if (!sel) return;
  $("#book-btn").disabled = true;
  try {
    const b = await api("/v1/bookings", { method: "POST", body: { cell: S.cell.id, start_ts: sl[sel.a].start_ts, end_ts: sl[sel.b].end_ts } });
    toast(`Booked ${hm(b.start_ts)}–${hm(b.end_ts)} UTC`);
    S.sel = null;
    await refreshBookings();
    refreshAvail();
    S.token = null; liveLoop();   // the camera may open right away
  } catch (e) { toast(e.message, "error"); $("#book-btn").disabled = false; }
});

async function refreshBookings() {
  try { S.bookings = await api("/v1/bookings"); } catch { return; }
  const t = now();
  const list = S.bookings.filter((b) => isStaff() || b.status === "confirmed" || b.end_ts > t - 86400).slice(0, 20);
  $("#bookings").innerHTML = list.length ? list.map((b) => {
    const active = b.status === "confirmed" && b.start_ts <= t && t < b.end_ts;
    const st = b.status === "cancelled" ? pill("CANCELLED") : active ? pill("ACTIVE") : b.end_ts <= t ? pill("ENDED") : pill("UPCOMING");
    const who = isStaff() && b.user_id !== S.me.id ? ` <span class="muted sm">${esc(b.user_id)}</span>` : "";
    return `<div class="item"><span>${st}</span><span class="grow">${dayLabel(b.start_ts)} ${hm(b.start_ts)}–${hm(b.end_ts)} UTC · ${(b.end_ts - b.start_ts) / 60} min${who}</span>
      ${b.status === "confirmed" && b.start_ts > t ? `<button class="btn ghost sm" data-cancel-booking="${esc(b.id)}" type="button">Cancel</button>` : ""}</div>`;
  }).join("") : `<p class="empty">No bookings yet. Pick a free slot above.</p>`;
  renderBookingSelect();
}

$("#bookings").addEventListener("click", async (ev) => {
  const b = ev.target.closest("[data-cancel-booking]");
  if (!b) return;
  try {
    const r = await api(`/v1/bookings/${b.dataset.cancelBooking}`, { method: "DELETE" });
    toast(r.refunded ? "Booking cancelled and refunded" : "Booking cancelled (inside the refund window: no refund)");
    await refreshBookings(); refreshAvail();
  } catch (e) { toast(e.message, "error"); }
});

// ---------- submit ----------------------------------------------------------------------------
function usableBookings() {
  const t = now();
  return S.bookings.filter((b) => b.status === "confirmed" && b.end_ts > t && (b.user_id === S.me.id)).sort((x, y) => x.start_ts - y.start_ts);
}

function renderBookingSelect() {
  const sel = $("#sub-booking"), prev = sel.value, t = now();
  const bs = usableBookings();
  sel.innerHTML = bs.map((b) => `<option value="${esc(b.id)}">${b.start_ts <= t ? "now · " : ""}${dayLabel(b.start_ts)} ${hm(b.start_ts)}–${hm(b.end_ts)} UTC</option>`).join("")
    + `<option value="">No booking (dry-run only)</option>`;
  if ([...sel.options].some((o) => o.value === prev) && prev) sel.value = prev;
  $("#run-btn").disabled = !sel.value;
  $("#run-btn").title = sel.value ? "" : "Running on the cell needs a booking";
}
$("#sub-booking").addEventListener("change", () => { $("#run-btn").disabled = !$("#sub-booking").value; });

$$(".seg [data-kind]").forEach((b) => b.addEventListener("click", () => {
  S.sources[S.kind] = $("#source").value;
  S.kind = b.dataset.kind;
  $$(".seg [data-kind]").forEach((x) => x.classList.toggle("on", x === b));
  $("#source").value = S.sources[S.kind];
}));

async function submit(mode) {
  const booking_id = $("#sub-booking").value || null;
  S.sources[S.kind] = $("#source").value;
  $("#dry-btn").disabled = $("#run-btn").disabled = true;
  $("#sub-msg").textContent = S.kind === "python" ? "Compiling in the sandbox and dry-running in the twin…" : "Dry-running in the twin…";
  $("#sub-result").innerHTML = "";
  try {
    const j = await api("/v1/jobs", { method: "POST", body: { booking_id, cell: S.cell.id, kind: S.kind, source: S.sources[S.kind], mode } });
    $("#sub-msg").textContent = "";
    const p = j.plan;
    $("#sub-result").innerHTML = `<div class="result">
      <div class="row between"><span>${pill(j.status)} <a href="#" data-job="${esc(j.id)}">${esc(shortId(j.id))}</a></span>
        <span class="muted">${p ? `${secs(p.duration_s)} · ${p.n_waypoints} waypoints` : ""}</span></div>
      ${j.error ? `<p class="error">${esc(j.error)}</p>` : ""}
      ${violationsHtml(j.violations)}
      <p class="muted sm">${j.status === "APPROVED" ? "Queued: it starts on the cell as soon as the door is locked and the interlock is clear. Watch it on the left."
        : p && p.ok ? "The twin video renders in a few seconds; see the job." : ""}</p>
      <button class="btn sm" type="button" data-job="${esc(j.id)}">Open job</button></div>`;
    refreshJobs();
  } catch (e) {
    $("#sub-msg").textContent = "";
    $("#sub-result").innerHTML = `<p class="error">${esc(e.message)}</p>`;
  } finally {
    $("#dry-btn").disabled = false;
    $("#run-btn").disabled = !$("#sub-booking").value;
  }
}
$("#dry-btn").addEventListener("click", () => submit("dry_run"));
$("#run-btn").addEventListener("click", () => submit("run"));

function violationsHtml(v) {
  if (!v || !v.length) return `<p class="okline">No violations</p>`;
  return `<ul class="viol">${v.map((x) => `<li><b>${esc(x.code)}</b> <span class="muted">step ${esc(x.step ?? "—")}</span> ${esc(x.message)}</li>`).join("")}</ul>`;
}

// ---------- jobs ----------------------------------------------------------------------------
async function refreshJobs() {
  try { S.jobs = await api("/v1/jobs"); } catch { return; }
  $("#jobs").innerHTML = S.jobs.length ? `<div class="item head"><span>Status</span><span class="grow">Job</span><span>Duration</span><span class="hide-sm">Submitted</span></div>`
    + S.jobs.slice(0, 50).map((j) => `<a href="#" class="item${j.id === S.jobId ? " cur" : ""}" data-job="${esc(j.id)}">
      <span>${pill(j.status)}</span><span class="grow"><b>${esc(shortId(j.id))}</b> <span class="tag">${j.mode === "run" ? "run" : "dry-run"}</span> <span class="tag">${esc(j.kind)}</span>${j.error ? ` <span class="muted sm">${esc(j.error.slice(0, 60))}</span>` : ""}</span>
      <span>${j.plan ? secs(j.plan.duration_s) : "—"}</span><span class="muted hide-sm">${ago(j.created_ts)}</span></a>`).join("")
    : `<p class="empty">No jobs yet. Submit a protocol in the Submit tab.</p>`;
}

function openJob(id) {
  S.jobId = id; S.jobSig = "";
  if (S.tab !== "jobs") setTab("jobs"); else { refreshJob(); refreshJobs(); }
  $("#job-detail").scrollIntoView({ block: "nearest" });
}

async function refreshJob() {
  const id = S.jobId;
  let j;
  try { j = await api(`/v1/jobs/${id}`); } catch (e) { $("#job-detail").innerHTML = `<p class="error">${esc(e.message)}</p>`; return; }
  if (id !== S.jobId) return;
  const sig = JSON.stringify([j.status, j.error, j.isaac_status, j.artifacts.map((a) => a.name + a.bytes)]);
  if (sig === S.jobSig) return;   // don't rebuild (and restart) video players on every poll
  S.jobSig = sig;
  renderJob(j);
}

function renderJob(j) {
  const p = j.plan || {}, st = p.stats || {};
  const done = TERMINAL.includes(j.status);
  const art = Object.fromEntries(j.artifacts.map((a) => [a.name, a]));
  const caps = j.artifacts.filter((a) => /^capture_.*\.(jpg|png)$/.test(a.name));
  const armS = j.started_ts && j.finished_ts ? j.finished_ts - j.started_ts : null;
  const stats = [
    ["Planned", secs(p.duration_s)], ["Arm time", armS == null ? "—" : secs(armS)], ["Waypoints", p.n_waypoints ?? "—"],
    ["Path", st.path_length_m != null ? `${(+st.path_length_m).toFixed(2)} m` : "—"],
    ["Max TCP", st.max_tcp_speed_mps != null ? `${(st.max_tcp_speed_mps * 1000).toFixed(0)} mm/s` : "—"],
    ["Min clearance", st.min_clearance_m != null ? `${(st.min_clearance_m * 1000).toFixed(0)} mm` : "—"],
    ["Max deviation", j.max_deviation_rad != null ? `${((j.max_deviation_rad * 180) / Math.PI).toFixed(3)}°` : "—"],
    ["Isaac", j.isaac_status || "—"],
  ];
  const vid = (name, label) => art[name]
    ? `<figure><figcaption>${label}</figcaption><video data-src="${esc(art[name].url)}" data-size="${art[name].bytes}" controls muted playsinline preload="metadata"></video></figure>`
    : "";
  const media = [
    art["dry_run.mp4"] ? vid("dry_run.mp4", "Dry-run · twin camera") : j.plan && j.plan.ok ? `<figure><figcaption>Dry-run · twin camera</figcaption><div class="vph">${art["dry_run.error.txt"] ? "render failed" : "rendering…"}</div></figure>` : "",
    done && art["recording.mp4"] ? vid("recording.mp4", "Recording · cell camera") : j.status === "RUNNING" ? `<figure><figcaption>Recording · cell camera</figcaption><div class="vph">recording… live on the left</div></figure>` : "",
    art["isaac.mp4"] ? vid("isaac.mp4", "Isaac Sim dry-run") : "",
  ].join("");
  const canCancel = ["SUBMITTED", "COMPILING", "DRY_RUN_OK", "APPROVED", "RUNNING"].includes(j.status) && j.status !== "DRY_RUN_OK";
  $("#job-detail").innerHTML = `<div class="detail">
    <div class="row between"><h3>${pill(j.status)} ${esc(shortId(j.id))} <span class="tag">${j.mode === "run" ? "run" : "dry-run"}</span> <span class="tag">${esc(j.kind)}</span></h3>
      <span class="row">${canCancel ? `<button class="btn ghost sm" id="job-cancel" type="button">Cancel</button>` : ""}
      ${done ? `<button class="btn primary sm" id="job-zip" type="button">results.zip</button>` : ""}
      <button class="btn ghost sm" id="job-close" type="button" aria-label="Close job">✕</button></span></div>
    ${j.error ? `<p class="error">${j.status === "ABORTED" ? "Aborted by the interlock: " : ""}${esc(j.error)}</p>` : ""}
    <dl class="stats">${stats.map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
    ${violationsHtml(j.violations)}
    ${media ? `<div class="media">${media}</div>` : ""}
    ${caps.length ? `<div class="caps">${caps.map((c) => `<figure><img data-src="${esc(c.url)}" data-size="${c.bytes}" alt="${esc(c.name)}"><figcaption>${esc(c.name.replace(/^capture_/, ""))}</figcaption></figure>`).join("")}</div>` : ""}
    ${j.artifacts.length ? `<div class="files">${j.artifacts.map((a) => `<button class="file" type="button" data-dl="${esc(a.url)}" data-name="${esc(a.name)}">${esc(a.name)} <span>${bytes(a.bytes)}</span></button>`).join("")}</div>` : ""}
    ${j.compile_logs ? `<details><summary>Compile log</summary><pre>${esc(j.compile_logs)}</pre></details>` : ""}
  </div>`;
  $$("#job-detail [data-src]").forEach(async (el) => { try { el.src = await blobUrl(el.dataset.src, el.dataset.size); } catch { /* shown as empty */ } });
  const zip = $("#job-zip");
  if (zip) zip.onclick = () => download(`/v1/jobs/${j.id}/results.zip`, `${j.id}-results.zip`);
  const c = $("#job-cancel");
  if (c) c.onclick = async () => { try { await api(`/v1/jobs/${j.id}/cancel`, { method: "POST" }); toast("Job cancelled"); S.jobSig = ""; refreshJob(); refreshJobs(); } catch (e) { toast(e.message, "error"); } };
  $("#job-close").onclick = () => { S.jobId = null; S.jobSig = ""; $("#job-detail").innerHTML = ""; refreshJobs(); };
}
$("#job-detail").addEventListener("click", (ev) => {
  const f = ev.target.closest("[data-dl]");
  if (f) download(f.dataset.dl, f.dataset.name);
});

// ---------- usage ----------------------------------------------------------------------------
async function refreshUsage() {
  let u, inv;
  try { [u, inv] = await Promise.all([api("/v1/usage"), api("/v1/invoice")]); } catch (e) { return toast(e.message, "error"); }
  const tot = u.totals || {};
  const qty = (k) => u.events.filter((e) => e.kind === k).reduce((s, e) => s + e.qty, 0);
  $("#usage-tiles").innerHTML = [
    ["Invoice", money(inv.total), `${esc(inv.org)} · ${esc(inv.month)}`, true],
    ["Slot time", `${(qty("slot_minutes") + qty("slot_refund")).toFixed(0)} min`, money((tot.slot_minutes || 0) + (tot.slot_refund || 0))],
    ["Arm time", secs(qty("arm_seconds")), money(tot.arm_seconds)],
    ["Dry-runs", qty("sim_dry_run").toFixed(0), money(tot.sim_dry_run)],
  ].map(([l, v, s, hero]) => `<div class="tile"><div class="tl">${l}</div><div class="tv${hero ? " accentnum" : ""}">${v}</div><div class="ts">${s}</div></div>`).join("");
  $("#invoice").innerHTML = `<thead><tr><th>Item</th><th class="r">Qty</th><th class="r">Events</th><th class="r">Amount</th></tr></thead><tbody>`
    + (inv.lines.length ? inv.lines.map((l) => `<tr><td>${esc(l.kind)}</td><td class="r">${(+l.qty).toFixed(2)} ${esc(l.unit)}</td><td class="r">${l.events}</td><td class="r">${money(l.amount)}</td></tr>`).join("")
      : `<tr><td colspan="4" class="empty">Nothing billed this month.</td></tr>`)
    + `</tbody><tfoot><tr><td colspan="3">Total (${esc(inv.currency.toUpperCase())})</td><td class="r">${money(inv.total)}</td></tr></tfoot>`;
  $("#events").innerHTML = `<thead><tr><th>When</th><th>Kind</th><th class="r">Qty</th><th class="r">Amount</th><th class="hide-sm">Ref</th></tr></thead><tbody>`
    + (u.events.length ? u.events.slice(0, 40).map((e) => `<tr><td>${ago(e.ts)}</td><td>${esc(e.kind)}</td><td class="r">${(+e.qty).toFixed(2)} ${esc(e.unit)}</td><td class="r">${money(e.amount)}</td><td class="muted hide-sm">${esc(shortId(e.job_id || e.booking_id))}</td></tr>`).join("")
      : `<tr><td colspan="5" class="empty">No meter events yet.</td></tr>`) + "</tbody>";
}

// ---------- operator ----------------------------------------------------------------------------
async function op(path, body, okMsg) {
  try {
    S.snap = await api(path, { method: "POST", body });
    renderState(S.wsLive ? "live · WebSocket 10 Hz" : "polled · 1 Hz");
    toast(okMsg);
  } catch (e) { toast(e.message, "error"); }
}
$$("[data-door]").forEach((b) => b.addEventListener("click", () => op(`/v1/cells/${S.cell.id}/door`, { action: b.dataset.door }, `Door: ${b.dataset.door}`)));
$("#estop").addEventListener("click", () => op(`/v1/cells/${S.cell.id}/estop`, null, "E-stop: arm stopped, interlock latched"));
$("#reset").addEventListener("click", () => op(`/v1/cells/${S.cell.id}/reset`, null, "Interlock reset"));

boot();
