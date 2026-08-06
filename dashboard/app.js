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

/* ---------- Router ---------- */
const views = {};
function route() {
  const name = (location.hash.replace(/^#\//, "") || "overview").split("/")[0];
  document.querySelectorAll(".nav-links a").forEach(a =>
    a.classList.toggle("active", a.dataset.view === name));
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
          <label class="ac-field"><span>Model</span>
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
      const r = await fetch("/api/agents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pending),
      });
      const body = await r.json();
      if (body.ok) {
        msg.className = "save-msg ok";
        msg.textContent = "Saved. " + (data.agent_running
          ? "Restart the agent to apply." : "Applied on next agent start.");
        for (const k of Object.keys(pending)) delete pending[k];
        data = await api("/api/agents");
        render();
        $("#agents-msg").className = "save-msg ok";
        $("#agents-msg").textContent = "Saved. " + (data.agent_running
          ? "Restart the agent to apply." : "Applied on next agent start.");
      } else {
        msg.className = "save-msg err";
        msg.textContent = "Not saved: " + Object.entries(body.errors || {})
          .map(([k, v]) => `${k}: ${v}`).join("; ");
      }
    } catch (e) {
      msg.className = "save-msg err";
      msg.textContent = "Save failed: " + e.message;
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
      const r = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pending),
      });
      const body = await r.json();
      if (body.ok) {
        msg.className = "save-msg ok";
        msg.textContent = `Saved ${body.saved.length} override(s). ` +
          (data.agent_running ? "Restart the agent to apply." : "Applied on next agent start.");
        const banner = $("#cfg-banner");
        banner.classList.add("saved");
      } else {
        msg.className = "save-msg err";
        msg.textContent = "Not saved: " + Object.entries(body.errors || {})
          .map(([k, v]) => `${k}: ${v}`).join("; ");
      }
    } catch (e) {
      msg.className = "save-msg err";
      msg.textContent = "Save failed: " + e.message;
    }
  }

  render();
};

/* ================= Conversation ================= */
views.conversation = async function () {
  main.innerHTML = header("Conversation", "The live history window — what the agent remembers verbatim right now.");
  let data;
  try { data = await api("/api/history"); }
  catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }

  function blockHtml(b, i) {
    if (typeof b === "string") return `<div>${esc(b)}</div>`;
    if (b.type === "text") return `<div>${esc(b.text)}</div>`;
    if (b.type === "tool_use")
      return `<button class="tool-chip" data-tool="${i}">⚙ ${esc(b.name)}</button>
              <div class="tool-detail" data-detail="${i}" hidden>${esc(JSON.stringify(b.input, null, 2))}</div>`;
    if (b.type === "tool_result") {
      const content = typeof b.content === "string" ? b.content
        : (b.content || []).map(c => c.text || "").join("\n");
      return `<button class="tool-chip" data-tool="${i}">↩ result</button>
              <div class="tool-detail" data-detail="${i}" hidden>${esc(content)}</div>`;
    }
    return "";
  }

  let idx = 0;
  const msgs = data.messages.map(m => {
    const content = typeof m.content === "string"
      ? `<div>${esc(m.content)}</div>`
      : m.content.map(b => blockHtml(b, idx++)).join("");
    return `<div class="msg ${m.role === "user" ? "user" : "assistant"}">
      <div class="m-role">${m.role === "user" ? "You" : "Agent"}</div>${content}</div>`;
  }).join("");

  main.innerHTML = header("Conversation",
    `The live history window — ${data.total} message(s) persisted, restored on next boot.`) +
    `<div class="chat">${msgs || '<div class="card empty">No conversation history yet</div>'}</div>`;

  main.querySelectorAll(".tool-chip").forEach(b => b.onclick = () => {
    const d = main.querySelector(`[data-detail="${b.dataset.tool}"]`);
    if (d) d.hidden = !d.hidden;
  });
  window.scrollTo(0, document.body.scrollHeight);
};

/* ================= Memory ================= */
views.memory = async function () {
  main.innerHTML = header("Memory", "Long-term memory staging.");
  let data;
  try { data = await api("/api/memory"); }
  catch (e) { main.innerHTML += `<div class="card empty">${esc(e.message)}</div>`; return; }

  const batches = data.pending.map(p => `
    <div class="card">
      <h2>${esc(fmtDate(p.ts))}</h2>
      <div class="card-sub">${(p.lines || []).length} line(s) staged</div>
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

    const docRows = data.docs.map(d => {
      const extent = d.pages ? `${d.pages} pages` : (d.duration ? d.duration : "text");
      return `<div class="kb-row done">
        <div class="kb-row-main">
          <div class="kb-row-name">${esc(d.title)}</div>
          <div class="kb-row-meta">${esc(extent)} · ${d.chunks} chunk${d.chunks === 1 ? "" : "s"}
            · added ${esc(fmtDate(d.ingested))}</div>
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
      const r = await fetch("/api/knowledge/ingest", { method: "POST" });
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

route();
