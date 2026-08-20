/* Voice Agent Dashboard — vanilla JS, hash-routed single page.
   Every view fetches fresh JSON from the local API; nothing is cached beyond
   the visible page, so a refresh always reflects what's on disk. */

"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const main = $("#main");

const esc = (s) => String(s ?? "")
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

/* The one POST helper — controls, saves, and the trading page all use it.
   The parsed body rides on the thrown Error (err.body) so callers can render
   field-error maps ({errors: {...}}), not just the flat message. */
async function apiPost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err = new Error(data.error || r.statusText);
    err.body = data;
    throw err;
  }
  return data;
}

/* "field: message; field: message" from a thrown apiPost error, for the
   save buttons; falls back to the plain message. */
function errorList(e) {
  const errs = e.body && e.body.errors;
  if (errs && Object.keys(errs).length) {
    return Object.entries(errs).map(([k, v]) => `${k}: ${v}`).join("; ");
  }
  return e.message;
}

function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) +
    " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
  return (n / 1048576).toFixed(1) + " MB";
}

/* Minimal markdown renderer for note summaries — headings, bold/italic,
   lists, code fences, paragraphs. Input is escaped first; this is display
   convenience, not a full parser. */
function md(text) {
  const lines = esc(text).split(/\r?\n/);
  const out = [];
  let inList = false, inCode = false;
  const closeList = () => { if (inList) { out.push("</ul>"); inList = false; } };
  for (const raw of lines) {
    const line = raw;
    if (/^```/.test(line.trim())) {
      closeList();
      out.push(inCode ? "</pre>" : "<pre>");
      inCode = !inCode;
      continue;
    }
    if (inCode) { out.push(line); continue; }
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) { closeList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }
    const li = line.match(/^\s*[-*]\s+(.*)$/);
    if (li) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${inline(li[1])}</li>`);
      continue;
    }
    closeList();
    if (line.trim() === "") continue;
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  if (inCode) out.push("</pre>");
  return out.join("\n");

  function inline(s) {
    return s
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|\W)\*(?!\s)(.+?)\*(?=\W|$)/g, "$1<em>$2</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }
}

/* ---------- Agent status (sidebar, polled) ---------- */
async function pollStatus() {
  const el = $("#agent-status");
  try {
    const o = await api("/api/overview");
    el.className = "agent-status " + (o.agent_running ? "running" : "stopped");
    $(".status-text", el).textContent = o.agent_running ? "agent running" : "agent stopped";
  } catch {
    el.className = "agent-status";
    $(".status-text", el).textContent = "dashboard offline";
  }
}
setInterval(pollStatus, 5000);
pollStatus();

/* ---------- Live controls (polled) ----------
   The agent's state, once every 2 s: it drives the sidebar mute button and —
   when the Conversation page is open — its composer. A control POST returns
   the resulting state synchronously, so there's no optimistic UI; the poll
   exists to follow changes made elsewhere (the headset button). The mute
   state is spelled out in words, the button goes red, and the tab title
   carries it, because forgetting the microphone is muted is the exact
   failure this is here to prevent. */
const muteBtn = $("#mute-btn");
let micState = null;   // last /api/control response
let muteNote = "";     // why the last mute request didn't land

function setMute(stateText, actionText, { on = false, live = false } = {}) {
  muteBtn.classList.toggle("muted", on);
  muteBtn.disabled = !live;
  muteBtn.setAttribute("aria-pressed", String(on));
  muteBtn.setAttribute("aria-label", `${stateText}. ${actionText}`);
  $(".mute-state", muteBtn).textContent = stateText;
  $(".mute-action", muteBtn).textContent = actionText;
  document.title = on ? "🔇 MUTED — Voice Agent Dashboard"
                      : "Voice Agent Dashboard";
}

function renderMute() {
  if (!micState || typeof micState.muted !== "boolean") {
    return setMute("Microphone", micState && !micState.agent_running
                   ? "agent not running" : "state unknown");
  }
  // One short line only — a second clause wraps in the narrow sidebar.
  setMute(micState.muted ? "🔇 MUTED" : "🎙 Mic on",
          muteNote ? `${muteNote} · tap to retry`
                   : (micState.muted ? "Tap to unmute" : "Tap to mute"),
          { on: micState.muted, live: true });
}

async function pollControl() {
  try { micState = await api("/api/control"); }
  catch { micState = null; }
  renderMute();
  renderComposer();
}

/* The Conversation page's composer, kept in step by the same poll. A no-op
   whenever that page isn't open, so the view owns its own markup and this
   never has to know which page is showing. */
function renderComposer() {
  const input = $("#chat-text"), btn = $("#chat-send");
  if (!input) return;
  // A real `muted` boolean means the agent is in THIS process and we can call
  // it. agent_running alone isn't enough: on the standalone dashboard it's
  // true whenever an agent lives in its own process, which we cannot reach —
  // the same distinction the server's two 409s draw.
  const live = !!(micState && typeof micState.muted === "boolean");
  input.disabled = btn.disabled = !live;
  // The availability goes in the placeholder, not the status line — that line
  // belongs to the last send, and the two would fight over it.
  input.placeholder =
    live ? "Type to the agent…"
      : micState && micState.agent_running
        ? "Chat lives on the agent's own dashboard — open port 8765"
        : "The agent isn't running — start it to chat";
}
setInterval(pollControl, 2000);
pollControl();

muteBtn.addEventListener("click", async () => {
  if (!micState || typeof micState.muted !== "boolean") return;
  try {
    const body = await apiPost("/api/control/mute", { muted: !micState.muted });
    micState = { ...micState, muted: body.muted, mode: body.mode };
    muteNote = "";
  } catch (e) {
    muteNote = e.message;
  }
  renderMute();
});

/* ---------- Router ---------- */
const views = {};
function currentView() {
  return (location.hash.replace(/^#\//, "") || "overview").split("/")[0];
}
function route() {
  const name = currentView();
  document.querySelectorAll(".nav-links a").forEach(a =>
    a.classList.toggle("active", a.dataset.view === name));
  main.className = "main";   // a view that dressed <main> up owns only its turn
  (views[name] || views.overview)();
}
window.addEventListener("hashchange", route);

function header(title, sub) {
  return `<h1 class="page-title">${esc(title)}</h1><p class="page-sub">${esc(sub)}</p>`;
}

/* ================= Overview ================= */
views.overview = async function () {
  main.innerHTML = header("Overview", "What the agent knows and how it is set up right now.");
  let o;
  try { o = await api("/api/overview"); }
  catch (e) { main.innerHTML += `<div class="card empty">Could not load: ${esc(e.message)}</div>`; return; }

  const maxCount = Math.max(1, ...o.folder_counts.map(f => f.count));
  const folderBars = o.folder_counts
    .slice().sort((a, b) => b.count - a.count)
    .map(f => `
      <div class="bar-row">
        <div class="b-label">${esc(f.display)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${(f.count / maxCount * 100).toFixed(1)}%"></div></div>
        <div class="b-value">${f.count}</div>
      </div>`).join("");

  const maxAct = Math.max(1, ...o.activity.map(a => a.count));
  const spark = o.activity.map(a => `
    <div class="s-col" title="${esc(a.day)}: ${a.count} note${a.count === 1 ? "" : "s"}">
      <div class="s-bar" style="height:${(a.count / maxAct * 100).toFixed(0)}%"></div>
    </div>`).join("");

  const recent = o.recent_notes.map(n => `
    <tr onclick="location.hash='#/notes/${esc(n.id)}'" style="cursor:pointer">
      <td>${esc(n.title)}</td><td>${esc(n.category || "")}</td>
      <td class="num">${esc(fmtDate(n.date))}</td>
    </tr>`).join("");

  main.innerHTML = header("Overview", "What the agent knows and how it is set up right now.") + `
    <div class="tiles">
      <div class="tile"><div class="t-label">Notes</div><div class="t-value">${o.total_notes}</div>
        <div class="t-note">${o.folder_counts.length} folders</div></div>
      <div class="tile"><div class="t-label">History window</div><div class="t-value">${o.history_messages}</div>
        <div class="t-note">messages persisted</div></div>
      <div class="tile"><div class="t-label">Memory staged</div><div class="t-value">${o.memory_pending}</div>
        <div class="t-note">awaiting consolidation</div></div>
      <div class="tile"><div class="t-label">Knowledge docs</div><div class="t-value">${o.knowledge_docs}</div>
        <div class="t-note">ingested sources</div></div>
      <div class="tile small"><div class="t-label">Talking to</div>
        <div class="t-value">${o.agent_running && o.talking_to.name
          ? `<span class="agent-chip agent-${esc(o.talking_to.active)}">${esc(o.talking_to.name)}</span>`
          : `<span class="t-muted">${o.talking_to.name ? esc(o.talking_to.name) + " (agent stopped)" : "—"}</span>`}</div>
        <div class="t-note"><a href="#/agents">manage agents</a></div></div>
      <div class="tile small"><div class="t-label">Conversation model</div><div class="t-value">${esc(o.convo_model)}</div>
        <div class="t-note">summaries: ${esc(o.summary_model)}</div></div>
      <div class="tile small"><div class="t-label">Whisper</div><div class="t-value">${esc(o.whisper_model)}</div>
        <div class="t-note">${o.overrides_active ? o.overrides_active + " override(s) active" : "all defaults"}</div></div>
    </div>
    <div class="two-col">
      <div class="card">
        <h2>Notes per folder</h2>
        <div class="card-sub">where your notes live</div>
        <div class="bars">${folderBars || '<div class="empty">No notes yet</div>'}</div>
      </div>
      <div class="card">
        <h2>Note activity</h2>
        <div class="card-sub">notes saved per day (days with activity)</div>
        <div class="spark">${spark || '<div class="empty">No notes yet</div>'}</div>
        <div class="spark-labels">
          <span>${esc(o.activity[0]?.day || "")}</span>
          <span>${esc(o.activity[o.activity.length - 1]?.day || "")}</span>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>Recent notes</h2>
      <div class="card-sub">latest saved — click to open</div>
      <table class="table">
        <thead><tr><th>Title</th><th>Folder</th><th>Saved</th></tr></thead>
        <tbody>${recent || '<tr><td colspan="3" class="empty">No notes yet</td></tr>'}</tbody>
      </table>
    </div>`;
};

/* ================= Agents ================= */
views.agents = async function () {
  const sub = "The personas you talk to by voice — say a name (\"Bob, …\") or \"switch to Tom\" to change who answers.";
  main.innerHTML = header("Agents", sub);
  let data, voices;
  try {
    data = await api("/api/agents");
    voices = (await api("/api/voices")).voices;
  } catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }

  // pending edits: key -> {field: value}; key -> null means "reset this agent"
  const pending = {};

  function agentVal(a, field) {
    const p = pending[a.key];
    return p && field in p ? p[field] : a[field];
  }

  function voiceOptions(current) {
    const opts = [`<option value="" ${!current ? "selected" : ""}>(system default)</option>`];
    let matched = false;
    for (const v of voices) {
      // The registry stores a substring ("Zira"); match it against full names.
      const sel = current && v.toLowerCase().includes(current.toLowerCase());
      if (sel) matched = true;
      opts.push(`<option value="${esc(v)}" ${sel ? "selected" : ""}>${esc(v)}</option>`);
    }
    if (current && !matched)
      opts.push(`<option value="${esc(current)}" selected>${esc(current)} (not installed)</option>`);
    return opts.join("");
  }

  function cardHtml(a) {
    const isDefault = a.key === data.default_agent;
    const modified = a.modified || a.key in pending;
    return `
      <div class="card agent-card" data-agent="${esc(a.key)}">
        <div class="ac-head">
          <span class="agent-dot agent-${esc(a.key)}"></span>
          <h2>${esc(a.name)}</h2>
          ${isDefault ? '<span class="badge">default</span>' : ""}
          ${modified ? '<span class="badge">modified</span>' : ""}
          ${data.talking_to.active === a.key && data.agent_running
            ? '<span class="badge live">talking now</span>' : ""}
          <button class="btn-ghost ac-reset" data-resetagent="${esc(a.key)}">Reset to defaults</button>
        </div>
        <div class="ac-grid">
          <label class="ac-field"><span>Role (one line)</span>
            <input class="f-text" data-agent-field="role" value="${esc(agentVal(a, "role"))}"></label>
          <label class="ac-field"><span>Model${
            // The dropdown is the configured DEFAULT. A voice "switch to
            // DeepSeek" changes only the running conversation, so say so
            // rather than letting the default read as current fact.
            data.agent_running && a.live_differs
              ? ` <span class="live-model">now on ${esc(a.live_model_label)}</span>`
              : ""}</span>
            <select class="f-select" data-agent-field="model">
              ${data.models.map(m => `<option value="${esc(m.value)}"
                ${m.value === agentVal(a, "model") ? "selected" : ""}>${esc(m.label)}</option>`).join("")}
            </select></label>
          <label class="ac-field"><span>Voice</span>
            ${voices.length
              ? `<select class="f-select" data-agent-field="tts_voice">${voiceOptions(agentVal(a, "tts_voice"))}</select>`
              : `<input class="f-text" data-agent-field="tts_voice" value="${esc(agentVal(a, "tts_voice") || "")}"
                        placeholder="SAPI voice substring">`}</label>
          <label class="ac-field"><span>Speaking rate (wpm, empty = default)</span>
            <input class="f-num" type="number" min="80" max="400" step="5"
                   data-agent-field="tts_rate" value="${agentVal(a, "tts_rate") ?? ""}"></label>
        </div>
        <div class="ac-field"><span>Spoken names (how you address them — include likely mishearings)</span>
          <div class="words-wrap" data-agent-words="${esc(a.key)}">
            ${agentVal(a, "aliases").map(w =>
              `<span class="word-tag">${esc(w)}<button data-word="${esc(w)}" title="remove">✕</button></span>`).join("")}
            <input class="word-add" placeholder="+ add name">
          </div></div>
        <div class="ac-field"><span>Persona (system-prompt block)</span>
          <textarea class="ac-persona" data-agent-field="persona" rows="7">${esc(agentVal(a, "persona"))}</textarea></div>
        <div class="ac-tools">Tools: ${a.tools.map(t => `<code>${esc(t)}</code>`).join(" ")}
          <span class="ac-tools-note">tool access is fixed in code (agents.py)</span></div>
      </div>`;
  }

  function render() {
    main.innerHTML = header("Agents", sub) + `
      <div class="config-banner">
        ⚠ Changes are written to <code>data/agents.json</code> and picked up when the agent
        ${data.agent_running ? "is <strong>restarted</strong> (it is running now)" : "next starts"}.
        Tool access stays in code.
      </div>` +
      data.agents.map(cardHtml).join("") + `
      <div class="save-bar">
        <button class="btn-primary" id="agents-save">Save changes</button>
        <span class="save-msg" id="agents-msg">${Object.keys(pending).length} agent(s) edited</span>
      </div>`;
    wire();
  }

  function edit(key, field, value) {
    if (pending[key] === null) pending[key] = {};
    pending[key] = { ...(pending[key] || {}), [field]: value };
  }

  function wire() {
    main.querySelectorAll(".agent-card").forEach(card => {
      const key = card.dataset.agent;
      card.querySelectorAll("[data-agent-field]").forEach(inp => {
        inp.onchange = () => {
          let v = inp.value;
          if (inp.dataset.agentField === "tts_rate") v = v === "" ? null : Number(v);
          if (inp.dataset.agentField === "tts_voice") v = v || null;
          edit(key, inp.dataset.agentField, v);
          $("#agents-msg").textContent = `${Object.keys(pending).length} agent(s) edited`;
        };
      });
      const wrap = card.querySelector("[data-agent-words]");
      const agent = data.agents.find(a => a.key === key);
      wrap.querySelector(".word-add").onkeydown = (e) => {
        if (e.key !== "Enter") return;
        const w = e.target.value.trim().toLowerCase();
        if (!w) return;
        const cur = [...agentVal(agent, "aliases")];
        if (!cur.includes(w)) cur.push(w);
        edit(key, "aliases", cur);
        render();
      };
      wrap.querySelectorAll(".word-tag button").forEach(b => {
        b.onclick = () => {
          edit(key, "aliases", agentVal(agent, "aliases").filter(w => w !== b.dataset.word));
          render();
        };
      });
      card.querySelector("[data-resetagent]").onclick = () => {
        pending[key] = null;  // null = reset this agent to coded defaults
        render();
      };
    });
    $("#agents-save").onclick = save;
  }

  async function save() {
    const msg = $("#agents-msg");
    msg.className = "save-msg"; msg.textContent = "Saving…";
    try {
      await apiPost("/api/agents", pending);
      for (const k of Object.keys(pending)) delete pending[k];
      data = await api("/api/agents");
      render();  // replaces the DOM — set the message on the fresh node, once
      const fresh = $("#agents-msg");
      fresh.className = "save-msg ok";
      fresh.textContent = "Saved. " + (data.agent_running
        ? "Restart the agent to apply." : "Applied on next agent start.");
    } catch (e) {
      msg.className = "save-msg err";
      msg.textContent = "Not saved: " + errorList(e);
    }
  }

  render();
};

/* ================= Notes ================= */
views.notes = async function () {
  const openId = location.hash.split("/")[2] || null;
  main.innerHTML = header("Notes", "Browse folders, read summaries and full transcripts.");
  let data;
  try { data = await api("/api/notes"); }
  catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }

  let activeFolder = null;
  let notes = data.notes;

  main.innerHTML = header("Notes", "Browse folders, read summaries and full transcripts.") + `
    <div class="folder-chips" id="chips"></div>
    <div class="notes-layout">
      <div>
        <input class="search-box" id="note-search" type="search"
               placeholder="Search titles and note text…" autocomplete="off">
        <div class="note-list" id="note-list"></div>
      </div>
      <div id="note-viewer"></div>
    </div>`;

  const chips = $("#chips"), list = $("#note-list"), viewer = $("#note-viewer");

  function renderChips() {
    const counts = {};
    data.notes.forEach(n => { const c = n.category || "general"; counts[c] = (counts[c] || 0) + 1; });
    chips.innerHTML =
      `<button class="chip ${activeFolder === null ? "active" : ""}" data-slug="">All<span class="chip-n">${data.notes.length}</span></button>` +
      data.folders.map(f =>
        `<button class="chip ${activeFolder === f.slug ? "active" : ""}" data-slug="${esc(f.slug)}"
                 title="${esc(f.description)}">${esc(f.display)}<span class="chip-n">${counts[f.slug] || 0}</span></button>`).join("");
    chips.querySelectorAll(".chip").forEach(b => b.onclick = () => {
      activeFolder = b.dataset.slug || null;
      notes = activeFolder ? data.notes.filter(n => (n.category || "general") === activeFolder) : data.notes;
      renderChips(); renderList(notes);
    });
  }

  function renderList(items, snippets) {
    list.innerHTML = items.length ? items.map(n => `
      <div class="note-item ${n.id === openId ? "active" : ""}" data-id="${esc(n.id)}">
        <div class="n-title">${esc(n.title)}</div>
        <div class="n-meta">${esc(fmtDate(n.date))} · ${esc(n.category || "")}</div>
        ${snippets && n.snippet ? `<div class="n-snippet">${esc(n.snippet)}</div>` : ""}
      </div>`).join("") : '<div class="empty">No notes here</div>';
    list.querySelectorAll(".note-item").forEach(el => el.onclick = () => {
      list.querySelectorAll(".note-item").forEach(x => x.classList.remove("active"));
      el.classList.add("active");
      openNote(el.dataset.id);
    });
  }

  async function openNote(id) {
    history.replaceState(null, "", `#/notes/${id}`);
    viewer.innerHTML = `<div class="card empty">Loading…</div>`;
    let n;
    try { n = await api(`/api/note?id=${encodeURIComponent(id)}`); }
    catch (e) { viewer.innerHTML = `<div class="card empty">${esc(e.message)}</div>`; return; }
    const hasTranscript = !!n.transcript.trim();
    viewer.innerHTML = `
      <div class="card note-viewer">
        <div class="nv-head">
          <h2>${esc(n.title)}</h2>
          <span class="nv-meta">${esc(n.folder_display)} · ${esc(fmtDate(n.date))}</span>
        </div>
        <div class="nv-tabs">
          <button id="tab-sum" class="active">Summary</button>
          <button id="tab-tx" ${hasTranscript ? "" : "disabled"}>Transcript${hasTranscript ? "" : " (none)"}</button>
        </div>
        <div class="md-body" id="nv-body">${md(n.summary || "*No summary file found.*")}</div>
      </div>`;
    $("#tab-sum").onclick = () => { setTab("sum"); };
    $("#tab-tx").onclick = () => { if (hasTranscript) setTab("tx"); };
    function setTab(t) {
      $("#tab-sum").classList.toggle("active", t === "sum");
      $("#tab-tx").classList.toggle("active", t === "tx");
      $("#nv-body").innerHTML = t === "sum"
        ? md(n.summary || "*No summary file found.*")
        : `<pre style="white-space:pre-wrap">${esc(n.transcript)}</pre>`;
    }
  }

  let searchTimer = null;
  $("#note-search").oninput = (e) => {
    clearTimeout(searchTimer);
    const q = e.target.value.trim();
    searchTimer = setTimeout(async () => {
      if (!q) { renderList(notes); return; }
      try {
        const r = await api(`/api/search?q=${encodeURIComponent(q)}`);
        renderList(r.results, true);
      } catch { /* keep current list */ }
    }, 250);
  };

  renderChips();
  renderList(notes);
  if (openId) openNote(openId);
  else viewer.innerHTML = '<div class="card empty">Select a note to read it</div>';
};

/* ================= Config ================= */
views.config = async function () {
  main.innerHTML = header("Config", "Adjust the agent's tunables. Saved values apply the next time the agent starts.");
  let data;
  try { data = await api("/api/config"); }
  catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }

  // pending: key -> value to override (null = reset to default)
  const pending = {};
  for (const t of data.tunables) if (t.override !== null && t.override !== undefined)
    pending[t.key] = t.override;

  const groups = [...new Set(data.tunables.map(t => t.group))];

  function fmtVal(t, v) {
    if (t.type === "words") return Array.isArray(v) ? v.join(", ") : String(v);
    if (t.type === "bool") return v ? "on" : "off";
    if (v === null || v === undefined || v === "") return "(empty)";
    return String(v);
  }

  function effValue(t) {
    return t.key in pending ? pending[t.key] : t.default;
  }

  function controlHtml(t) {
    const v = effValue(t);
    if (t.type === "bool") {
      return `<label class="toggle"><input type="checkbox" data-key="${t.key}" ${v ? "checked" : ""}>
        <span class="track"></span><span class="knob"></span></label>
        <span class="f-unit">${v ? "on" : "off"}</span>`;
    }
    if (t.type === "choice") {
      return `<select class="f-select" data-key="${t.key}">` +
        t.choices.map(c => `<option value="${esc(c.value)}" ${c.value === v ? "selected" : ""}>${esc(c.label)}</option>`).join("") +
        `</select>`;
    }
    if (t.type === "text") {
      return `<input class="f-text" data-key="${t.key}" value="${esc(v ?? "")}"
              placeholder="${t.nullable ? "(system default)" : ""}">`;
    }
    if (t.type === "words") {
      const words = Array.isArray(v) ? v : [];
      return `<div class="words-wrap" data-key="${t.key}">` +
        words.map(w => `<span class="word-tag">${esc(w)}<button data-word="${esc(w)}" title="remove">✕</button></span>`).join("") +
        `<input class="word-add" placeholder="+ add word" data-add="${t.key}"></div>`;
    }
    // int / float → slider + number
    return `
      <input type="range" data-key="${t.key}" data-pair="num"
             min="${t.min}" max="${t.max}" step="${t.step}" value="${v}">
      <input type="number" class="f-num" data-key="${t.key}" data-pair="range"
             min="${t.min}" max="${t.max}" step="${t.step}" value="${v}">
      <span class="f-unit">${esc(t.unit || "")}</span>`;
  }

  function fieldHtml(t) {
    const overridden = t.key in pending;
    // Tall controls (the word-tag editor) get their own full-width stacked
    // layout so they never collide with the centered label / default column.
    const wide = t.type === "words" ? " field-wide" : "";
    return `
      <div class="field${wide}" data-field="${t.key}">
        <div class="f-label">${esc(t.label)}<span class="f-key">${t.key}</span></div>
        <div class="f-control">${controlHtml(t)}</div>
        <div class="f-side">
          ${overridden ? '<span class="badge">modified</span>' : ""}
          <span class="f-default">default: ${esc(fmtVal(t, t.default))}</span>
          ${overridden ? `<button class="reset-btn" data-reset="${t.key}">reset</button>` : ""}
        </div>
        <div class="f-help">${esc(t.help || "")}</div>
      </div>`;
  }

  function render() {
    main.innerHTML = header("Config", "Adjust the agent's tunables. Saved values apply the next time the agent starts.") + `
      <div class="config-banner" id="cfg-banner">
        ⚠ Changes are written to <code>data/config_overrides.json</code> and picked up when the agent
        ${data.agent_running ? "is <strong>restarted</strong> (it is running now)" : "next starts"}.
      </div>` +
      groups.map(g => `
        <div class="card config-group">
          <h2>${esc(g)}</h2>
          ${data.tunables.filter(t => t.group === g).map(fieldHtml).join("")}
        </div>`).join("") + `
      <div class="save-bar">
        <button class="btn-primary" id="save-btn">Save changes</button>
        <button class="btn-ghost" id="reset-all">Reset all to defaults</button>
        <span class="save-msg" id="save-msg">${Object.keys(pending).length} override(s) set</span>
      </div>`;
    wire();
  }

  function setPending(key, value) {
    const t = data.tunables.find(x => x.key === key);
    const same = JSON.stringify(value) === JSON.stringify(t.default);
    if (same) delete pending[key]; else pending[key] = value;
    // re-render just this field's side column + msg
    render();
  }

  function wire() {
    main.querySelectorAll('input[type="range"]').forEach(r => {
      r.oninput = () => {
        const num = main.querySelector(`.f-num[data-key="${r.dataset.key}"]`);
        num.value = r.value;
      };
      r.onchange = () => commitNumber(r.dataset.key, r.value);
    });
    main.querySelectorAll(".f-num").forEach(n => {
      n.onchange = () => commitNumber(n.dataset.key, n.value);
    });
    function commitNumber(key, raw) {
      const t = data.tunables.find(x => x.key === key);
      let v = t.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
      if (isNaN(v)) v = t.default;
      v = Math.min(t.max, Math.max(t.min, v));
      if (t.type === "float") v = Math.round(v * 1000) / 1000;
      setPending(key, v);
    }
    main.querySelectorAll('.toggle input').forEach(c => {
      c.onchange = () => setPending(c.dataset.key, c.checked);
    });
    main.querySelectorAll(".f-select").forEach(s => {
      s.onchange = () => setPending(s.dataset.key, s.value);
    });
    main.querySelectorAll(".f-text").forEach(inp => {
      inp.onchange = () => {
        const t = data.tunables.find(x => x.key === inp.dataset.key);
        const v = inp.value.trim();
        setPending(inp.dataset.key, v === "" && t.nullable ? null : v);
      };
    });
    main.querySelectorAll(".word-add").forEach(inp => {
      inp.onkeydown = (e) => {
        if (e.key !== "Enter") return;
        const key = inp.dataset.add;
        const t = data.tunables.find(x => x.key === key);
        const w = inp.value.trim().toLowerCase();
        if (!w) return;
        const cur = [...(effValue(t) || [])];
        if (!cur.includes(w)) cur.push(w);
        setPending(key, cur.sort());
      };
    });
    main.querySelectorAll(".word-tag button").forEach(b => {
      b.onclick = () => {
        const wrap = b.closest(".words-wrap");
        const key = wrap.dataset.key;
        const t = data.tunables.find(x => x.key === key);
        setPending(key, (effValue(t) || []).filter(w => w !== b.dataset.word));
      };
    });
    main.querySelectorAll("[data-reset]").forEach(b => {
      b.onclick = () => { delete pending[b.dataset.reset]; render(); };
    });
    $("#reset-all").onclick = () => {
      for (const k of Object.keys(pending)) delete pending[k];
      render();
    };
    $("#save-btn").onclick = save;
  }

  async function save() {
    const msg = $("#save-msg");
    msg.className = "save-msg"; msg.textContent = "Saving…";
    try {
      const body = await apiPost("/api/config", pending);
      msg.className = "save-msg ok";
      msg.textContent = `Saved ${body.saved.length} override(s). ` +
        (data.agent_running ? "Restart the agent to apply." : "Applied on next agent start.");
      $("#cfg-banner").classList.add("saved");
    } catch (e) {
      msg.className = "save-msg err";
      msg.textContent = "Not saved: " + errorList(e);
    }
  }

  render();
};

/* ================= Conversation ================= */
let convoAgent = null;  // persona whose thread is shown; null = follow active
let convoSig = null;    // signature of the thread as last rendered
let convoShown = null;  // which persona that render showed (scroll decisions)

/* The transcript is its own scroller (see .main.chat-page), so "go to the
   newest message" is a scroll of that box, not of the window. */
const chatToEnd = () => { const c = $(".chat"); if (c) c.scrollTop = c.scrollHeight; };

views.conversation = async function () {
  /* A live re-render must not eat what the user was doing: keep a typed
     draft and the scroll position — captured FIRST, before the header wipe
     below empties the old DOM (reading them after the await found nothing,
     which silently dropped the draft). */
  const draft = $("#chat-text") ? $("#chat-text").value : "";
  const prev = $(".chat");
  const prevTop = prev ? prev.scrollTop : 0;
  const wasAtEnd = !prev ||
    prev.scrollHeight - prev.scrollTop - prev.clientHeight < 80;
  main.className = "main chat-page";
  main.innerHTML = header("Conversation", "The live history window — what the agent remembers verbatim right now.");
  let data;
  try {
    data = await api("/api/history" + (convoAgent ? `?agent=${convoAgent}` : ""));
  }
  catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }

  /* Jump to the newest message only if the user was already at the bottom or
     just switched threads — never yank them off old messages they scrolled
     up to read. */
  const stick = data.agent !== convoShown || wasAtEnd;
  convoSig = historySig(data);
  convoShown = data.agent;

  /* `key=value  key=value` on one line, for the chip. Strings go in bare so
     the common case (folder=Trading) reads as prose; anything else as JSON. */
  const argLine = (input) => Object.entries(input || {})
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join("  ");
  const oneLine = (s, n = 160) => {
    const t = String(s ?? "").replace(/\s+/g, " ").trim();
    return t.length > n ? t.slice(0, n - 1) + "…" : t;
  };

  function chip(i, name, summary, detail) {
    return `<button class="tool-chip" data-tool="${i}"
              ><span class="tc-name">${esc(name)}</span
              ><span class="tc-args">${esc(oneLine(summary))}</span></button>
            <div class="tool-detail" data-detail="${i}" hidden>${esc(detail)}</div>`;
  }

  function blockHtml(b, i) {
    if (typeof b === "string") return `<div>${esc(b)}</div>`;
    if (b.type === "text") return `<div>${esc(b.text)}</div>`;
    if (b.type === "tool_use")
      return chip(i, `⚙ ${b.name}`, argLine(b.input), JSON.stringify(b.input, null, 2));
    if (b.type === "tool_result") {
      const content = typeof b.content === "string" ? b.content
        : (b.content || []).map(c => c.text || "").join("\n");
      return chip(i, "↩ result", content, content);
    }
    return "";
  }

  /* Time beside the role, and a divider whenever the day changes — the same
     two cues a chat app gives. `ts` is the end of the turn that wrote the
     message and is empty on anything saved before stamping existed, which
     shows as no time rather than a made-up one. */
  let idx = 0, lastDay = "";
  const msgs = data.messages.map(m => {
    const t = m.ts ? new Date(m.ts) : null;
    const stamp = t && !isNaN(t)
      ? t.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }) : "";
    const day = t && !isNaN(t)
      ? t.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" }) : "";
    const divider = day && day !== lastDay ? `<div class="chat-day">${esc(day)}</div>` : "";
    if (day) lastDay = day;
    const content = typeof m.content === "string"
      ? `<div>${esc(m.content)}</div>`
      : m.content.map(b => blockHtml(b, idx++)).join("");
    return `${divider}<div class="msg ${m.role === "user" ? "user" : "assistant"}">
      <div class="m-role">${m.role === "user" ? "You" : "Agent"}${
        stamp ? ` <span class="m-time">${esc(stamp)}</span>` : ""}</div>${content}</div>`;
  }).join("");

  /* Threads are per-persona: one stacked row of big buttons picks whose
     conversation is on screen. */
  const agentBtns = (data.agents || []).map(a => `
    <button class="agent-tab ${a.key === data.agent ? "sel" : ""}"
            data-agent="${esc(a.key)}">${esc(a.name)}</button>`).join("");

  main.innerHTML = header("Conversation",
    `${esc((data.agents || []).find(a => a.key === data.agent)?.name || data.agent)}'s thread — ${data.total} message(s) persisted, restored on next boot.`) +
    `<div class="agent-tabs">${agentBtns}</div>
     <div class="chat">${msgs || '<div class="card empty">No conversation history yet</div>'}</div>
     <form class="composer" id="chat-form">
       <input type="text" id="chat-text" aria-label="Message to the agent"
              placeholder="Type to the agent…" autocomplete="off" disabled>
       <button class="send-btn" id="chat-send" type="submit" disabled>Send</button>
       <div class="send-note" id="chat-note" role="status"></div>
     </form>`;

  main.querySelectorAll(".tool-chip").forEach(b => b.onclick = () => {
    const d = main.querySelector(`[data-detail="${b.dataset.tool}"]`);
    if (!d) return;
    d.hidden = !d.hidden;
    b.classList.toggle("open", !d.hidden);
  });

  main.querySelectorAll(".agent-tab").forEach(b => b.onclick = () => {
    convoAgent = b.dataset.agent;
    views.conversation();
  });

  const note = $("#chat-note");
  $("#chat-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const input = $("#chat-text"), text = input.value.trim();
    if (!text || $("#chat-send").disabled) return;
    try {
      // Send the viewed persona along, or a message typed into Bob's thread
      // is answered by whoever happens to be active.
      await apiPost("/api/control/send_message",
                    convoAgent ? { text, agent: convoAgent } : { text });
      input.value = "";
      note.className = "send-note ok";
      note.textContent = "Sent — the agent answers aloud.";
      // Show it in the transcript straight away. It isn't in history yet (the
      // agent writes that only once it has answered), so it's marked pending
      // until the real exchange arrives and replaces it.
      $(".chat").insertAdjacentHTML("beforeend",
        `<div class="msg user pending"><div class="m-role">You · sending</div>
         <div>${esc(text)}</div></div>`);
      chatToEnd();
      awaitReply(convoSig);
    } catch (e) {
      note.className = "send-note err";
      note.textContent = e.message;
    }
  });

  renderComposer();  // don't wait up to 2 s for the poll to enable it
  $("#chat-text").value = draft;
  if (stick) chatToEnd();
  else $(".chat").scrollTop = prevTop;  // stay where the reader was
  watchConvo();
};

/* Compare CONTENT, not the count: history is trimmed to HISTORY_MAX_MESSAGES,
   so once it is at the cap a new turn pushes old messages off the front and
   the total barely moves — watching `total` meant a typed message never
   appeared until a manual reload. */
const historySig = (h) => JSON.stringify(h.messages);

/* The one live watcher for the Conversation page: while it is open, fetch the
   viewed thread every 3 s and re-render on any change. This is what makes
   SPOKEN exchanges appear as they happen — the old watcher ran only for 40 s
   after a typed send, so voice turns sat invisible until a manual reload.
   Self-terminating on navigation, and guarded so re-renders (which call
   watchConvo again) never stack a second interval. */
let convoTimer = null;
function watchConvo() {
  if (convoTimer) return;
  convoTimer = setInterval(async () => {
    if (currentView() !== "conversation") {
      clearInterval(convoTimer); convoTimer = null; return;
    }
    const h = await api("/api/history" + (convoAgent ? `?agent=${convoAgent}` : ""))
      .catch(() => null);
    if (h && historySig(h) !== convoSig) views.conversation();
  }, 3000);
}

/* History is written only once the agent has finished answering, so a typed
   message and its reply land together, seconds later — watchConvo shows them.
   This only covers the honest-failure path: after ~40 s with no change, say
   so plainly instead of leaving the pending bubble unexplained (a note
   dialogue can hold the floor far longer). */
function awaitReply(before) {
  setTimeout(() => {
    const note = $("#chat-note");  // fresh lookup — re-renders replace it
    if (currentView() !== "conversation" || convoSig !== before || !note) return;
    note.className = "send-note";
    note.textContent = "Still no reply — the agent may be mid-note. Reload to check.";
  }, 40000);
}

/* ================= Memory ================= */
views.memory = async function () {
  main.innerHTML = header("Memory", "Long-term memory staging.");
  let data;
  try { data = await api("/api/memory"); }
  catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }

  const batches = data.pending.map(p => `
    <div class="card">
      <h2>${esc(fmtDate(p.ts))}</h2>
      <div class="card-sub">${(p.lines || []).length} line(s) staged · ${
        p.agent ? `${esc(p.agent)}'s memory` : "shared (pre-isolation)"}</div>
      <div class="mono-list">${(p.lines || []).map(l => `<div>${esc(l)}</div>`).join("")}</div>
    </div>`).join("");

  main.innerHTML = header("Memory",
    "Messages that aged out of the live window, staged here until the next boot consolidates them " +
    `into searchable summaries (needs ≥ ${data.min_messages} lines).`) +
    (batches || '<div class="card empty">Nothing staged — everything has been consolidated.</div>');
};

/* ================= Knowledge ================= */
views.knowledge = async function () {
  main.innerHTML = header("Knowledge", "Ingested reference material.");
  let data;
  try { data = await api("/api/knowledge"); }
  catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }

  let jobTimer = null;
  let confirming = null;   // name of the pending file asking "remove?"
  let target = "common";   // where the next ingest routes new files

  /* Laid out for a narrow visual field: one column, big targets, each control
     next to the thing it acts on. Steps read top to bottom — add, then the
     waiting list with its own Ingest button, then what's already in. */
  function render() {
    const job = data.job || {};
    const running = job.state === "running";
    const pending = data.pending || [];
    const hasMedia = pending.some(p => p.media);

    const pendingRows = pending.map(p => confirming === p.name ? `
      <div class="kb-row confirm" data-name="${esc(p.name)}">
        <div class="kb-row-main">
          <div class="kb-row-name">Remove this file?</div>
          <div class="kb-row-meta">${esc(p.name)}</div>
        </div>
        <div class="kb-row-actions">
          <button class="btn-danger" data-remove="${esc(p.name)}">Yes, remove</button>
          <button class="btn-ghost-lg" data-cancel="1">Keep</button>
        </div>
      </div>` : `
      <div class="kb-row" data-name="${esc(p.name)}">
        <div class="kb-row-main">
          <div class="kb-row-name">${esc(p.name)}</div>
          <div class="kb-row-meta">${p.media ? "Video / audio — will be transcribed"
                                             : "Document"} · ${esc(fmtBytes(p.bytes))}</div>
        </div>
        <div class="kb-row-actions">
          <button class="btn-ghost-lg" data-ask="${esc(p.name)}"
                  ${running ? "disabled" : ""}>Remove</button>
        </div>
      </div>`).join("");

    const targetName = key =>
      (data.targets || []).find(t => t.key === key)?.name || key;
    const docRows = data.docs.map(d => {
      const extent = d.pages ? `${d.pages} pages` : (d.duration ? d.duration : "text");
      const where = (d.collection && d.collection !== "common")
        ? ` · ${esc(targetName(d.collection))}` : "";
      return `<div class="kb-row done">
        <div class="kb-row-main">
          <div class="kb-row-name">${esc(d.title)}</div>
          <div class="kb-row-meta">${esc(extent)} · ${d.chunks} chunk${d.chunks === 1 ? "" : "s"}
            · added ${esc(fmtDate(d.ingested))}${where}</div>
        </div>
      </div>`;
    }).join("");

    // Why Ingest might be unavailable, in the order these actually bite.
    let blocked = "";
    if (running) blocked = "";
    else if (!pending.length) blocked = "Nothing waiting.";
    else if (data.agent_running)
      blocked = "Close the voice agent first — it has the search index open.";

    main.innerHTML = header("Knowledge",
      "Books, documents, and course videos your agent can search.") + `
      <div class="kb-narrow">

        <div class="card kb-step">
          <h2><span class="kb-num">1</span> Add files</h2>
          <button class="btn-big" id="kb-upload">Upload files</button>
          <div class="dropzone" id="kb-drop" tabindex="0" role="button"
               aria-label="Drop knowledge files here">
            …or drag files here
          </div>
          <input type="file" id="kb-file" multiple hidden
                 accept="${esc((data.accept || []).join(","))}">
          <div id="kb-uploads"></div>
          <div class="kb-hint">Audio, video, PDF, or text.</div>
        </div>

        <div class="card kb-step">
          <h2><span class="kb-num">2</span> Waiting to add${pending.length ? ` (${pending.length})` : ""}</h2>
          ${pending.length ? pendingRows
            : '<div class="kb-none">Nothing waiting. Upload a file above.</div>'}

          <div class="kb-targets" role="radiogroup" aria-label="Ingest into">
            <div class="kb-hint">Ingest into:</div>
            ${(data.targets || []).map(t => `
              <label class="kb-target ${t.key === target ? "sel" : ""}">
                <input type="radio" name="kb-target" value="${esc(t.key)}"
                       ${t.key === target ? "checked" : ""}
                       ${running ? "disabled" : ""}>
                ${esc(t.name)}</label>`).join("")}
          </div>

          <button class="btn-big ${running ? "busy" : ""}" id="kb-ingest"
                  ${running || blocked ? "disabled" : ""}>
            ${running ? "Working…" : "Ingest"}</button>
          <div class="kb-status ${job.state === "error" ? "err" : (job.state === "done" ? "ok" : "")}"
               id="kb-job">${esc(blocked || job.message || "")}</div>
          ${running ? `<div class="ingest-note">Leave this page open until it finishes.${
            hasMedia ? " Video takes roughly 20–40 minutes per hour." : ""}</div>` : ""}
        </div>

        <div class="card kb-step">
          <h2><span class="kb-num">3</span> In the knowledge base${data.docs.length ? ` (${data.docs.length})` : ""}</h2>
          ${docRows || '<div class="kb-none">Nothing added yet.</div>'}
        </div>

      </div>`;

    wire();
    if (running) pollJob();
  }

  function wire() {
    const dz = $("#kb-drop"), picker = $("#kb-file");
    const browse = () => picker.click();
    $("#kb-upload").onclick = browse;
    dz.onclick = browse;
    dz.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); browse(); } };
    picker.onchange = () => { upload([...picker.files]); picker.value = ""; };

    // dragover must be cancelled or the browser navigates to the dropped file.
    dz.ondragover = e => { e.preventDefault(); dz.classList.add("over"); };
    dz.ondragleave = () => dz.classList.remove("over");
    dz.ondrop = e => {
      e.preventDefault();
      dz.classList.remove("over");
      upload([...(e.dataTransfer?.files || [])]);
    };

    const btn = $("#kb-ingest");
    if (btn && !btn.disabled) btn.onclick = ingest;

    main.querySelectorAll('input[name="kb-target"]').forEach(r => {
      r.onchange = () => { target = r.value; render(); };
    });

    // Removing is two taps: ask, then confirm. A file can represent a long
    // upload, and there is no undo once it's off disk.
    main.querySelectorAll("[data-ask]").forEach(b => {
      b.onclick = () => { confirming = b.dataset.ask; render(); };
    });
    main.querySelectorAll("[data-cancel]").forEach(b => {
      b.onclick = () => { confirming = null; render(); };
    });
    main.querySelectorAll("[data-remove]").forEach(b => {
      b.onclick = () => remove(b.dataset.remove, b);
    });
  }

  async function remove(name, btn) {
    btn.disabled = true;
    btn.textContent = "Removing…";
    try {
      const r = await fetch("/api/knowledge/remove", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const body = await r.json();
      confirming = null;
      data = await api("/api/knowledge");
      render();
      if (!body.ok) {
        const msg = $("#kb-job");
        msg.className = "kb-status err";
        msg.textContent = body.error || "Could not remove that file.";
      }
    } catch (e) {
      confirming = null;
      render();
      const msg = $("#kb-job");
      msg.className = "kb-status err";
      msg.textContent = e.message;
    }
  }

  /* Uploads go one at a time over XHR — fetch() gives no upload progress, and a
     two-gigabyte course deserves a bar rather than a frozen page. */
  function upload(files) {
    if (!files.length) return;
    const box = $("#kb-uploads");
    let queue = Promise.resolve();
    for (const file of files) {
      const row = document.createElement("div");
      row.className = "up-row";
      row.innerHTML = `<div class="up-head"><span class="up-name"></span>
        <span class="up-pct">waiting…</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:0"></div></div>`;
      row.querySelector(".up-name").textContent = file.name;  // never as HTML
      box.appendChild(row);
      queue = queue.then(() => sendOne(file, row));
    }
    queue.then(async () => {
      data = await api("/api/knowledge");
      render();
    });
  }

  function sendOne(file, row) {
    return new Promise(resolve => {
      const pct = row.querySelector(".up-pct");
      const fill = row.querySelector(".bar-fill");
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/knowledge/upload?name=" + encodeURIComponent(file.name));
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.upload.onprogress = e => {
        if (!e.lengthComputable) return;
        const p = Math.round(e.loaded / e.total * 100);
        fill.style.width = p + "%";
        pct.textContent = p + "%";
      };
      xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch { /* keep the status */ }
        if (xhr.status === 200 && body.ok) {
          row.classList.add("ok");
          pct.textContent = "uploaded";
          fill.style.width = "100%";
        } else {
          row.classList.add("err");
          pct.textContent = body.error || `failed (${xhr.status})`;
          fill.style.width = "0";
        }
        resolve();
      };
      xhr.onerror = () => {
        row.classList.add("err");
        pct.textContent = "upload failed";
        resolve();
      };
      xhr.send(file);
    });
  }

  async function ingest() {
    const btn = $("#kb-ingest"), msg = $("#kb-job");
    btn.disabled = true;
    btn.classList.add("busy");
    btn.textContent = "Working…";
    msg.className = "kb-status";
    msg.textContent = "Starting…";
    try {
      const r = await fetch("/api/knowledge/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target }),
      });
      const body = await r.json();
      if (!body.ok) {
        msg.className = "kb-status err";
        msg.textContent = body.error || "Could not start.";
        btn.disabled = false;
        btn.classList.remove("busy");
        btn.textContent = "Ingest";
        return;
      }
      pollJob();
    } catch (e) {
      msg.className = "kb-status err";
      msg.textContent = e.message;
      btn.disabled = false;
      btn.classList.remove("busy");
      btn.textContent = "Ingest";
    }
  }

  /* Poll while the background job runs. Cleared on the first non-running state
     so a finished ingest doesn't keep the timer (and the view) alive. */
  function pollJob() {
    clearTimeout(jobTimer);
    jobTimer = setTimeout(async () => {
      if (location.hash !== "#/knowledge") return;  // navigated away; stop
      try {
        const job = await api("/api/knowledge/job");
        const msg = $("#kb-job");
        if (job.state === "running") {
          if (msg) { msg.className = "kb-status"; msg.textContent = job.message || "Working…"; }
          pollJob();
          return;
        }
        data = await api("/api/knowledge");
        data.job = job;
        render();
      } catch {
        pollJob();  // server busy mid-ingest; try again
      }
    }, 1500);
  }

  render();
};

/* ================= Discord ================= */
views.discord = async function () {
  main.innerHTML = header("Discord", "Captured notifications and trade alerts.");
  let data;
  try { data = await api("/api/discord"); }
  catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }

  if (!data.available) {
    main.innerHTML = header("Discord", "Captured notifications and trade alerts.") +
      `<div class="card empty">Discord Notifier data not found at <code>${esc(data.dir)}</code></div>`;
    return;
  }
  main.innerHTML = header("Discord",
    "Read-only view over the sibling Discord Notifier project — the same data the voice tools read.") + `
    <div class="two-col">
      <div class="card">
        <h2>Recent trades</h2>
        <div class="card-sub">last ${data.trades.length} trade line(s)</div>
        <div class="mono-list">${data.trades.map(t => `<div>${esc(t)}</div>`).join("") || '<div class="empty">No trades captured</div>'}</div>
      </div>
      <div class="card">
        <h2>Message log tail</h2>
        <div class="card-sub">latest captured notifications</div>
        <div class="log-view">${data.log.map(esc).join("\n") || "empty"}</div>
      </div>
    </div>`;
};

/* ================= Logs ================= */
views.logs = async function () {
  main.innerHTML = header("Logs", "Session logs.");
  let data;
  try { data = await api("/api/logs"); }
  catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }

  main.innerHTML = header("Logs", "Dated session logs — newest first. Click one to tail it.") + `
    <div class="pill-row" id="log-pills">${data.files.map((f, i) => `
      <button class="chip ${i === 0 ? "active" : ""}" data-log="${esc(f.name)}"
              title="${fmtBytes(f.size)} · ${esc(fmtDate(f.modified))}">${esc(f.name.replace("session_", "").replace(".log", ""))}</button>`).join("")}
    </div>
    <div class="card" id="log-card"><div class="empty">Select a log</div></div>`;

  async function open(name) {
    $("#log-card").innerHTML = '<div class="empty">Loading…</div>';
    try {
      const l = await api(`/api/log?name=${encodeURIComponent(name)}&lines=400`);
      const html = l.lines.map(ln => {
        const cls = / ERROR | Traceback|error/i.test(ln) ? "ln-err"
          : / WARNING /.test(ln) ? "ln-warn" : "ln-info";
        return `<span class="${cls}">${esc(ln)}</span>`;
      }).join("\n");
      $("#log-card").innerHTML = `
        <h2>${esc(name)}</h2>
        <div class="card-sub">last ${l.lines.length} lines</div>
        <div class="log-view" id="log-view">${html || "empty"}</div>`;
      const lv = $("#log-view");
      lv.scrollTop = lv.scrollHeight;
    } catch (e) {
      $("#log-card").innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    }
  }

  main.querySelectorAll("[data-log]").forEach(b => b.onclick = () => {
    main.querySelectorAll("#log-pills .chip").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    open(b.dataset.log);
  });
  if (data.files.length) open(data.files[0].name);
};

/* ================= Trading ================= */
/* Real trading via the tastytrade API (see TRADING_PLAN.md). The page gets
   real-time bid/ask itself: it fetches a DXLink token from the local API and
   opens the websocket directly (same pattern as Tasty-Web), so quotes stream
   straight into the browser with no server-side relay. */

const TR = {
  status: null, chain: null, ticket: null, dryRun: null,
  ws: null, wsReady: false, wsSubs: new Set(), quotes: {}, wsErr: "",
};

function trStream() {
  if (TR.ws) return;
  api("/api/trading/quote-token").then(t => {
    if (!t.url || !t.token) throw new Error("no token");
    const ws = new WebSocket(t.url);
    TR.ws = ws;
    let authed = false;
    ws.onopen = () => ws.send(JSON.stringify({
      type: "SETUP", channel: 0, version: "0.1",
      keepaliveTimeout: 60, acceptKeepaliveTimeout: 60 }));
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "SETUP" && !authed) {
        ws.send(JSON.stringify({ type: "AUTH", channel: 0, token: t.token }));
      } else if (m.type === "AUTH_STATE" && m.state === "AUTHORIZED") {
        authed = true;
        ws.send(JSON.stringify({ type: "CHANNEL_REQUEST", channel: 1,
          service: "FEED", parameters: { contract: "AUTO" } }));
      } else if (m.type === "CHANNEL_OPENED" && m.channel === 1) {
        ws.send(JSON.stringify({ type: "FEED_SETUP", channel: 1,
          acceptAggregationPeriod: 0.1, acceptDataFormat: "COMPACT",
          acceptEventFields: { Quote: ["eventSymbol", "bidPrice", "askPrice"],
                               Trade: ["eventSymbol", "price"] } }));
        TR.wsReady = true;
        const subs = [...TR.wsSubs];
        TR.wsSubs = new Set();
        trSubscribe(subs);
      } else if (m.type === "FEED_DATA") {
        trFeedData(m.data || []);
      } else if (m.type === "KEEPALIVE") {
        ws.send(JSON.stringify({ type: "KEEPALIVE", channel: 0 }));
      }
    };
    ws.onclose = () => { TR.ws = null; TR.wsReady = false; };
    ws.onerror = () => { TR.wsErr = "quote stream error"; };
  }).catch(e => {
    TR.wsErr = e.message;
    const el = $("#tr-stream-note");
    if (el) el.textContent = "No live quotes: " + TR.wsErr;
  });
}

function trSubscribe(symbols) {
  const add = [];
  for (const s of symbols) {
    if (!s || TR.wsSubs.has(s)) continue;
    TR.wsSubs.add(s);
    add.push({ type: "Quote", symbol: s }, { type: "Trade", symbol: s });
  }
  if (add.length && TR.wsReady) {
    TR.ws.send(JSON.stringify({ type: "FEED_SUBSCRIPTION", channel: 1, add }));
  }
}

function trFeedData(data) {
  for (let i = 0; i + 1 < data.length; i += 2) {
    const type = data[i], v = data[i + 1];
    if (!Array.isArray(v)) continue;
    const stride = type === "Quote" ? 3 : type === "Trade" ? 2 : 0;
    if (!stride) continue;
    for (let j = 0; j + stride <= v.length; j += stride) {
      const q = TR.quotes[v[j]] || (TR.quotes[v[j]] = {});
      if (type === "Quote") { q.bid = v[j + 1]; q.ask = v[j + 2]; }
      else q.last = v[j + 1];
    }
  }
  trPaintQuotes();
}

const trNum = (x, dp = 2) =>
  (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(dp);
const trMid = (q) => (q && q.bid != null && q.ask != null)
  ? (q.bid + q.ask) / 2 : (q ? q.last : null);

function trNetMid() {
  if (!TR.ticket) return null;
  let net = 0;
  for (const leg of TR.ticket.legs) {
    const m = trMid(TR.quotes[leg.streamer]);
    if (m == null) return null;
    net += (leg.side === "Short" ? m : -m) * leg.size;
  }
  return net;
}

function trPaintQuotes() {
  document.querySelectorAll("[data-qsym]").forEach(el => {
    const q = TR.quotes[el.dataset.qsym];
    const f = el.dataset.qfield;
    el.textContent = trNum(f === "mid" ? trMid(q) : q && q[f]);
  });
  const net = trNetMid();
  const el = $("#tr-netmid");
  if (el) el.textContent = net == null ? "—"
    : `${trNum(Math.abs(net))} ${net > 0 ? "credit" : "debit"} (mid)`;
}

function trLegRow(leg, i) {
  const ch = TR.chain;
  const exps = ch ? ch.expirations : [leg.expiration];
  const strikes = ch ? (ch.strikes[leg.expiration] || []) : [leg.strike];
  const opt = (v, cur, label) =>
    `<option value="${esc(v)}"${String(v) === String(cur) ? " selected" : ""}>${esc(label ?? v)}</option>`;
  return `<tr data-leg="${i}">
    <td><select class="f-select" data-f="side">
      ${opt("Long", leg.side)}${opt("Short", leg.side)}</select></td>
    <td><input class="f-num tr-size" data-f="size" type="number" min="1" value="${leg.size}"></td>
    <td><select class="f-select" data-f="expiration">
      ${exps.map(e => opt(e, leg.expiration)).join("")}</select></td>
    <td><select class="f-select" data-f="strike">
      ${strikes.map(s => opt(s, leg.strike)).join("")}</select></td>
    <td><select class="f-select" data-f="cp">
      ${opt("Call", leg.cp)}${opt("Put", leg.cp)}</select></td>
    <td class="num" data-qsym="${esc(leg.streamer || "")}" data-qfield="bid">—</td>
    <td class="num" data-qsym="${esc(leg.streamer || "")}" data-qfield="ask">—</td>
    <td class="num" data-qsym="${esc(leg.streamer || "")}" data-qfield="mid">—</td>
    <td><button class="btn-ghost tr-del" title="remove leg">✕</button></td>
  </tr>`;
}

function trRenderTicket() {
  const box = $("#tr-ticket");
  if (!box) return;
  const tk = TR.ticket;
  if (!tk) {
    box.innerHTML = '<div class="empty">No ticket — build a strategy above, or ask the agent by voice.</div>';
    return;
  }
  const problems = (tk.problems || []).map(p => `<div class="tr-problem">⚠ ${esc(p)}</div>`).join("");
  box.innerHTML = `
    <div class="card-sub">${esc(tk.underlying)} · ${esc(tk.strategy || "custom")}
      ${tk.review ? '<span class="badge">reviewed</span>' : ""}</div>
    <table class="table tr-legs">
      <thead><tr><th>Side</th><th>Qty</th><th>Expiration</th><th>Strike</th>
        <th>C/P</th><th>Bid</th><th>Ask</th><th>Mid</th><th></th></tr></thead>
      <tbody>${tk.legs.map(trLegRow).join("")}</tbody>
    </table>
    ${problems}
    <div class="tr-terms">
      <label>Limit <input id="tr-price" class="f-num" type="number" step="0.01" min="0"
        value="${tk.limit_price ?? ""}" placeholder="price"></label>
      <select id="tr-effect" class="f-select">
        <option value="">effect…</option>
        <option value="Credit"${tk.price_effect === "Credit" ? " selected" : ""}>Credit</option>
        <option value="Debit"${tk.price_effect === "Debit" ? " selected" : ""}>Debit</option>
      </select>
      <select id="tr-tif" class="f-select">
        <option${tk.tif === "Day" ? " selected" : ""}>Day</option>
        <option${tk.tif === "GTC" ? " selected" : ""}>GTC</option>
      </select>
      <span class="card-sub">net <span id="tr-netmid">—</span></span>
      <button id="tr-usemid" class="btn-ghost">Use mid</button>
      <span style="flex:1"></span>
      <button id="tr-clear" class="btn-ghost">Clear</button>
      <button id="tr-review" class="btn-primary">Review (dry run)</button>
    </div>
    <div id="tr-dryrun"></div>`;

  if (TR.chain) trSubscribe(tk.legs.map(l => l.streamer).filter(Boolean));
  trPaintQuotes();

  const push = () => trPushTicket();
  box.querySelectorAll("select[data-f],input[data-f]").forEach(el =>
    el.onchange = push);
  box.querySelectorAll(".tr-del").forEach((b, idx) => b.onclick = () => {
    TR.ticket.legs.splice(idx, 1); trPushTicket();
  });
  $("#tr-price").onchange = push;
  $("#tr-effect").onchange = push;
  $("#tr-tif").onchange = push;
  $("#tr-usemid").onclick = () => {
    const net = trNetMid();
    if (net == null) return;
    $("#tr-price").value = Math.abs(net).toFixed(2);
    $("#tr-effect").value = net > 0 ? "Credit" : "Debit";
    push();
  };
  $("#tr-clear").onclick = async () => {
    await apiPost("/api/trading/ticket", { op: "clear" });
    TR.ticket = null; TR.dryRun = null; trRenderTicket();
  };
  $("#tr-review").onclick = trReview;
}

function trReadTicketDom() {
  const legs = [];
  document.querySelectorAll("#tr-ticket tr[data-leg]").forEach(tr => {
    const g = (f) => tr.querySelector(`[data-f="${f}"]`).value;
    legs.push({ side: g("side"), size: parseInt(g("size"), 10) || 1,
      expiration: g("expiration"), strike: parseFloat(g("strike")),
      cp: g("cp") });
  });
  return {
    op: "set",
    underlying: TR.ticket.underlying,
    strategy: TR.ticket.strategy,
    legs,
    tif: $("#tr-tif") ? $("#tr-tif").value : "Day",
    limit_price: $("#tr-price") && $("#tr-price").value !== "" ? $("#tr-price").value : null,
    price_effect: $("#tr-effect") ? $("#tr-effect").value || null : null,
  };
}

async function trPushTicket() {
  try {
    const body = TR.ticket.legs.length && document.querySelector("#tr-ticket tr[data-leg]")
      ? trReadTicketDom()
      : { op: "set", underlying: TR.ticket.underlying, strategy: TR.ticket.strategy,
          legs: TR.ticket.legs, tif: TR.ticket.tif,
          limit_price: TR.ticket.limit_price, price_effect: TR.ticket.price_effect };
    const r = await apiPost("/api/trading/ticket", body);
    TR.ticket = r.ticket; TR.dryRun = null;
    trRenderTicket();
  } catch (e) {
    const el = $("#tr-dryrun");
    if (el) el.innerHTML = `<div class="tr-problem">⚠ ${esc(e.message)}</div>`;
  }
}

async function trReview() {
  const el = $("#tr-dryrun");
  el.innerHTML = '<div class="empty">Running dry run…</div>';
  let r;
  try { r = await apiPost("/api/trading/dry-run", {}); }
  catch (e) { el.innerHTML = `<div class="tr-problem">⚠ ${esc(e.message)}</div>`; return; }
  TR.dryRun = r;
  if (r.ticket) TR.ticket = r.ticket;
  if (!r.ok) {
    el.innerHTML = r.errors.map(x => `<div class="tr-problem">✕ ${esc(x)}</div>`).join("");
    return;
  }
  const env = TR.status ? TR.status.env : "?";
  el.innerHTML = `
    <div class="tr-review">
      <div><strong>Dry run OK.</strong>
        Buying-power effect ${trNum(r.buying_power_change)} ·
        fees ${trNum(r.total_fees)} ·
        new buying power ${trNum(r.new_buying_power)}</div>
      ${r.warnings.map(w => `<div class="tr-problem">⚠ ${esc(w)}</div>`).join("")}
      <button id="tr-submit" class="btn-primary">Submit to ${esc(env)}…</button>
    </div>`;
  $("#tr-submit").onclick = async () => {
    const desc = TR.ticket.description || "this order";
    if (!window.confirm(`Place with ${env.toUpperCase()} account?\n\n${desc}`)) return;
    try {
      const s = await apiPost("/api/trading/submit",
        { confirm: true, fingerprint: r.fingerprint });
      el.innerHTML = `<div class="tr-review"><strong>Submitted.</strong>
        Order ${esc(s.order.id)} — ${esc(s.order.status)}</div>`;
      trLoadOrders();
    } catch (e2) {
      el.innerHTML = `<div class="tr-problem">✕ ${esc(e2.message)}</div>`;
    }
  };
}

async function trLoadOrders() {
  const box = $("#tr-orders");
  if (!box) return;
  let data;
  try { data = await api("/api/trading/orders"); }
  catch (e) { box.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  const rows = data.orders.map(o => `
    <tr><td>${esc(o.id)}</td><td>${esc(o.underlying)}</td>
      <td>${esc(o.status)}</td>
      <td class="num">${o.price != null ? trNum(o.price) + " " + esc(o.price_effect || "") : "—"}</td>
      <td>${o.legs.map(l => esc(`${l.action} ${l.quantity} ${l.symbol}`)).join("<br>")}</td>
      <td>${o.working ? `<button class="btn-ghost tr-cancel" data-id="${esc(o.id)}">Cancel</button>` : ""}</td>
    </tr>`).join("");
  const log = (data.log || []).slice(-12).reverse().map(e =>
    `<div>${esc(e.at)} · ${esc(e.env)} · ${esc(e.kind)}${e.order_id ? " #" + esc(e.order_id) : ""}</div>`).join("");
  box.innerHTML = `
    <table class="table">
      <thead><tr><th>Id</th><th>Under</th><th>Status</th><th>Price</th><th>Legs</th><th></th></tr></thead>
      <tbody>${rows || '<tr><td colspan="6" class="empty">No orders today</td></tr>'}</tbody>
    </table>
    <div class="card-sub" style="margin-top:8px">Audit log</div>
    <div class="mono-list">${log || '<div class="empty">empty</div>'}</div>`;
  box.querySelectorAll(".tr-cancel").forEach(b => b.onclick = async () => {
    b.disabled = true;
    try { await apiPost("/api/trading/cancel", { order_id: b.dataset.id }); }
    catch (e) { window.alert("Cancel failed: " + e.message); }
    trLoadOrders();
  });
}

async function trLoadPositions() {
  const box = $("#tr-positions");
  if (!box) return;
  let data;
  try { data = await api("/api/trading/positions"); }
  catch (e) { box.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  const rows = data.positions.map(p => `
    <tr><td>${esc(p.symbol)}</td><td class="num">${p.quantity}</td>
      <td class="num">${trNum(p.avg_open)}</td>
      <td class="num">${p.streamer
        ? `<span data-qsym="${esc(p.streamer)}" data-qfield="mid">${trNum(p.mark)}</span>`
        : trNum(p.mark)}</td>
      <td class="num ${p.unrealized > 0 ? "tr-pos" : p.unrealized < 0 ? "tr-neg" : ""}">${trNum(p.unrealized)}</td>
    </tr>`).join("");
  box.innerHTML = `
    <table class="table">
      <thead><tr><th>Symbol</th><th>Qty</th><th>Avg open</th><th>Mark</th><th>Unrealized</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="5" class="empty">No open positions</td></tr>'}</tbody>
    </table>
    <div class="card-sub" style="margin-top:8px">
      Total unrealized: <strong>${trNum(data.total_unrealized)}</strong>
      ${data.unmarked ? ` · ${data.unmarked} unmarked` : ""}</div>`;
  trSubscribe(data.positions.map(p => p.streamer).filter(Boolean));
}

async function trLoadPnl() {
  const box = $("#tr-pnl-out");
  box.innerHTML = '<div class="empty">Crunching transactions…</div>';
  const qs = new URLSearchParams({
    start: $("#tr-pnl-start").value, end: $("#tr-pnl-end").value });
  const u = $("#tr-pnl-under").value.trim();
  if (u) qs.set("underlying", u);
  let rep;
  try { rep = await api("/api/trading/pnl?" + qs); }
  catch (e) { box.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  const by = Object.entries(rep.by_underlying || {})
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([k, v]) => `<tr><td>${esc(k)}</td>
      <td class="num ${v > 0 ? "tr-pos" : v < 0 ? "tr-neg" : ""}">${trNum(v)}</td></tr>`).join("");
  const events = (rep.events || []).slice(-10).reverse().map(ev =>
    `<div>${esc((ev.closed_at || "").slice(0, 16))} · ${esc(ev.symbol)} ×${ev.quantity} → ${trNum(ev.pnl)}</div>`).join("");
  box.innerHTML = `
    <div class="tr-pnl-head">
      Realized <strong class="${rep.realized > 0 ? "tr-pos" : rep.realized < 0 ? "tr-neg" : ""}">${trNum(rep.realized)}</strong>
      · fees ${trNum(rep.fees)} · ${rep.events.length} closes
      ${rep.unrealized != null ? ` · unrealized now <strong>${trNum(rep.unrealized)}</strong>` : ""}
    </div>
    ${rep.note ? `<div class="tr-problem">${esc(rep.note)}</div>` : ""}
    <div class="two-col">
      <div><div class="card-sub">By underlying</div>
        <table class="table"><tbody>${by || '<tr><td class="empty">none</td></tr>'}</tbody></table></div>
      <div><div class="card-sub">Recent closes</div>
        <div class="mono-list">${events || '<div class="empty">none</div>'}</div></div>
    </div>`;
}

views.trading = async function () {
  main.innerHTML = header("Trading", "Loading trading status…");
  let st;
  try { st = await api("/api/trading/status"); }
  catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }
  TR.status = st;

  const today = new Date().toISOString().slice(0, 10);
  const envBadge = `<span class="badge tr-env ${st.env === "live" ? "tr-live" : ""}">${esc(st.env.toUpperCase())}</span>`;
  main.innerHTML = header("Trading",
    "Multi-leg options on the tastytrade API — the same ticket the voice agent builds.") + `
    <div class="card">
      <div class="tr-status">${envBadge}
        ${st.configured
          ? `account <code>${esc(st.account || "?")}</code> · stream: ${esc(st.stream || "n/a")}`
          : `<span class="tr-problem">${esc(st.problem || "not configured")}</span>`}
        <span id="tr-stream-note" class="card-sub"></span>
      </div>
    </div>
    <div class="card">
      <h2>Strategy builder</h2>
      <div class="tr-build">
        <input id="tr-sym" class="f-text" placeholder="SPX, /ES, AAPL…" value="SPX">
        <select id="tr-strat" class="f-select">
          ${(st.strategies || []).map(s => `<option>${esc(s)}</option>`).join("")}
        </select>
        <input id="tr-size" class="f-num" type="number" min="1" value="1" title="contracts per leg">
        <button id="tr-build" class="btn-primary">Build</button>
      </div>
      <div id="tr-ticket"><div class="empty">Loading ticket…</div></div>
    </div>
    <div class="two-col">
      <div class="card"><h2>Orders</h2><div id="tr-orders"><div class="empty">Loading…</div></div></div>
      <div class="card"><h2>Positions</h2><div id="tr-positions"><div class="empty">Loading…</div></div></div>
    </div>
    <div class="card">
      <h2>P&amp;L</h2>
      <div class="tr-build">
        <label>from <input id="tr-pnl-start" class="f-text" type="date" value="${today}"></label>
        <label>to <input id="tr-pnl-end" class="f-text" type="date" value="${today}"></label>
        <input id="tr-pnl-under" class="f-text" placeholder="underlying (optional)">
        <button id="tr-pnl-run" class="btn-primary">Report</button>
      </div>
      <div id="tr-pnl-out" class="card-sub">Pick a period and run the report.</div>
    </div>`;

  if (!st.configured) {
    const note = '<div class="empty">Waiting for trading credentials (see status above).</div>';
    $("#tr-ticket").innerHTML = note;
    $("#tr-orders").innerHTML = note;
    $("#tr-positions").innerHTML = note;
    return;
  }
  trStream();

  $("#tr-build").onclick = async () => {
    const btn = $("#tr-build");
    btn.disabled = true; btn.textContent = "Building…";
    try {
      const r = await apiPost("/api/trading/ticket", {
        op: "build", symbol: $("#tr-sym").value,
        strategy: $("#tr-strat").value,
        size: parseInt($("#tr-size").value, 10) || 1 });
      TR.ticket = r.ticket;
      TR.chain = await api("/api/trading/chain?symbol=" +
        encodeURIComponent(r.ticket.underlying));
      trSubscribe([TR.chain.underlying_streamer]);
      trRenderTicket();
    } catch (e) {
      $("#tr-ticket").innerHTML = `<div class="tr-problem">⚠ ${esc(e.message)}</div>`;
    }
    btn.disabled = false; btn.textContent = "Build";
  };
  $("#tr-pnl-run").onclick = trLoadPnl;

  // Existing ticket (e.g. built by voice) — load it and its chain.
  try {
    const t = await api("/api/trading/ticket");
    TR.ticket = t.ticket;
    if (t.ticket) {
      TR.chain = await api("/api/trading/chain?symbol=" +
        encodeURIComponent(t.ticket.underlying));
      trSubscribe([TR.chain.underlying_streamer]);
    }
  } catch { TR.ticket = null; }
  trRenderTicket();
  trLoadOrders();
  trLoadPositions();
};

route();
