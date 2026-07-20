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
    return `
      <div class="field" data-field="${t.key}">
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

  const rows = data.docs.map(d => `
    <tr><td>${esc(d.title)}</td><td>${esc(d.source)}</td>
    <td class="num">${d.chunks}</td><td class="num">${esc(fmtDate(d.ingested))}</td></tr>`).join("");

  main.innerHTML = header("Knowledge",
    "Reference books and documents ingested into the searchable knowledge base. " +
    "Drop files into the knowledge folder and run --ingest to add more.") + `
    <div class="card">
      <table class="table">
        <thead><tr><th>Title</th><th>Source file</th><th>Chunks</th><th>Ingested</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4" class="empty">Nothing ingested yet</td></tr>'}</tbody>
      </table>
      <div class="card-sub" style="margin-top:10px">folder: <code>${esc(data.dir)}</code></div>
    </div>`;
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
