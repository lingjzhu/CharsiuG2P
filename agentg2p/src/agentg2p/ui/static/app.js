"use strict";
/* The agentg2p window. Every result comes from the library through the local
   server (server.py); this file only asks, renders and remembers preferences. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const TOKEN = $('meta[name="agentg2p-token"]').content;
const NUM = new Intl.NumberFormat("en");
const VIEWS = ["transcribe", "list", "compare", "dictionaries"];
const LIST_CHUNK = 2000;
const LIST_PAGE = 2000;

const pct = (p) => (p == null ? "–" : `${(p * 100).toFixed(2)}%`);
const mb = (b) => (!b ? "" : b >= 1e7 ? `${Math.round(b / 1e6)} MB` : `${(b / 1e6).toFixed(1)} MB`);
const plural = (n, one, many = `${one}s`) => `${NUM.format(n)} ${n === 1 ? one : many}`;

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : String(kid));
  }
  return el;
}

function icon(id, cls) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", cls);
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(ns, "use");
  use.setAttribute("href", `#${id}`);
  svg.append(use);
  return svg;
}

async function api(name, args = {}) {
  let r;
  try {
    r = await fetch(`/api/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Agentg2p-Token": TOKEN },
      body: JSON.stringify(args),
    });
  } catch {
    throw new Error("The agentg2p process has stopped. Close this window and run agentg2p ui again.");
  }
  const data = await r.json().catch(() => ({ error: `Unexpected reply from agentg2p (${r.status}).` }));
  if (!r.ok || data.error) {
    const err = new Error(data.error || `Unexpected reply from agentg2p (${r.status}).`);
    err.detail = data.detail || "";
    throw err;
  }
  return data.result;
}

/* ---- state ------------------------------------------------------------------ */

const S = {
  info: null, languages: [], byTag: new Map(),
  lang: null, dicts: [], dict: null,
  backend: "auto", lexicon: true, joined: false,
  recent: [], dictByLang: {}, view: "transcribe",
  t: { seq: 0, rows: [], prev: [], sel: -1, notes: [], busy: false, error: null, downloading: null },
  l: { seq: 0, name: "", words: [], rows: [], filter: "all", find: "", sel: -1, limit: LIST_PAGE,
       busy: false, done: 0, error: null, notes: [], downloading: null },
  c: { seq: 0, data: null, busy: false, error: null },
  d: { all: null, find: "", filter: "all", sort: { k: "language", dir: 1 }, error: null },
};

let saveTimer;
function savePrefs() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    api("save_prefs", { prefs: {
      lang: S.lang, recent: S.recent, dictByLang: S.dictByLang, backend: S.backend,
      lexicon: S.lexicon, joined: S.joined, view: S.view, text: $("#t-input").value.slice(0, 20000),
    } }).catch(() => {});
  }, 400);
}

let msgTimer;
function say(text, ms = 5000) {
  $("#s-msg").textContent = text;
  clearTimeout(msgTimer);
  if (ms) msgTimer = setTimeout(() => { $("#s-msg").textContent = ""; }, ms);
}

const langName = (tag) => S.byTag.get(tag)?.name || tag;
const phonesText = (p) => (p ? p.join(S.joined ? "" : " ") : "");
const dictById = (id) => S.dicts.find((d) => d.id === id);

function markInstalled(info) {
  for (const list of [S.dicts, S.d.all || []]) {
    const d = list.find((x) => x.id === info.id);
    if (d) Object.assign(d, info);
  }
  if (S.dict?.id === info.id) Object.assign(S.dict, info);
}

async function download(d) {
  markInstalled(await api("download", { dictionary: d.id }));
}

/* ---- views and toolbar -------------------------------------------------------- */

function setView(v) {
  if (!VIEWS.includes(v)) v = "transcribe";
  S.view = v;
  document.body.dataset.view = v;
  for (const b of $$(".views button")) b.setAttribute("aria-selected", String(b.dataset.view === v));
  for (const id of VIEWS) $(`#view-${id}`).hidden = id !== v;
  closePop();
  if (v === "dictionaries") loadAllDicts();
  if (v === "compare") runCompare();
  if (v === "list") (S.l.stale ? runList() : renderList());
  if (v === "transcribe") (S.t.stale ? runTranscribe() : renderTranscribe());
  renderStatus();
  savePrefs();
}

function focusView() {
  const target = { transcribe: "#t-input", list: "#l-open", compare: "#c-word", dictionaries: "#d-find" }[S.view];
  $(target)?.focus();
}

async function setLanguage(tag, dictId) {
  if (!S.byTag.has(tag)) return;
  S.lang = tag;
  S.recent = [tag, ...S.recent.filter((t) => t !== tag && S.byTag.has(t))].slice(0, 6);
  const name = langName(tag);
  $("#lang-label").textContent = name;
  $("#t-lang").textContent = name;
  $("#t-input").placeholder = `Type or paste ${name} text`;
  $("#c-meta").textContent = `in every ${name} dictionary`;
  try {
    S.dicts = await api("dictionaries", { lang: tag });
  } catch (e) {
    S.dicts = [];
    S.dict = null;
    say(e.message, 0);
    return;
  }
  if (S.lang !== tag) return;
  S.dict = dictById(dictId) || dictById(S.dictByLang[tag]) || S.dicts[0];
  S.dictByLang[tag] = S.dict.id;
  renderToolbar();
  savePrefs();
  refresh();
}

function setDictionary(id) {
  const d = dictById(id);
  if (!d) return;
  S.dict = d;
  S.dictByLang[S.lang] = d.id;
  renderToolbar();
  savePrefs();
  refresh();
}

function refresh() {
  if (S.view === "transcribe") runTranscribe();
  if (S.view === "list") runList();
  if (S.view === "compare") runCompare();
  S.t.stale = S.view !== "transcribe";
  S.l.stale = S.view !== "list";
}

function renderToolbar() {
  const d = S.dict;
  $("#dict-label").textContent = d ? d.name : "–";
  $("#dict-btn").title = d ? `${d.name} · ${d.notation}` : "";
  const model = S.backend === "auto" ? `Auto${d?.model ? ` · ${d.model}` : ""}` : S.backend === "fst" ? "FST" : "ByT5";
  $("#model-label").textContent = S.lexicon ? model : `${model} only`;
}

function renderExport() {
  const ok = (S.view === "transcribe" && S.t.rows.length) || (S.view === "list" && S.l.rows.length);
  $("#export-btn").disabled = !ok;
}

/* ---- shared table rendering ---------------------------------------------------- */

function sourceCell(src) {
  if (src === "lexicon") return h("span", { class: "src", text: "Dictionary" });
  if (src === "fst") return h("span", { class: "src model", text: "FST", title: "Not in the dictionary: predicted by the FST" });
  if (src === "byt5") return h("span", { class: "src model", text: "ByT5", title: "Not in the dictionary: predicted by ByT5" });
  return h("span", { class: "src muted", text: "–" });
}

function sourceLong(src) {
  if (src === "lexicon") return `Entry in ${S.dict?.name}`;
  if (src === "fst") return `Predicted by the FST trained on ${S.dict?.name}`;
  if (src === "byt5") return "Predicted by ByT5";
  return "Not transcribed";
}

function wordRow(r, i, st, opts) {
  const cls = [r.flag ? "flagged" : "", i === st.sel ? "sel" : "", opts.fresh ? "fresh" : ""].filter(Boolean).join(" ");
  const tr = h("tr", { "data-i": i, tabindex: i === Math.max(st.sel, opts.first) ? "0" : "-1", class: cls || null,
                       "aria-expanded": String(i === st.sel) });
  if (opts.numbered) tr.append(h("td", { class: "n", text: NUM.format(i + 1) }));
  const ph = h("td", { class: "ipa-cell" },
    r.phones ? h("span", { class: "ipa", text: phonesText(r.phones) }) : h("span", { class: "nophones", text: "–" }));
  if (!opts.numbered && r.variants.length > 1) {
    ph.append(h("span", { class: "more", text: `+${plural(r.variants.length - 1, "variant")}` }));
  }
  if (r.flag) ph.append(h("span", { class: "flag", text: r.flag }));
  tr.append(h("td", { class: "word", text: r.text }), ph, h("td", {}, sourceCell(r.source)));
  if (opts.numbered) tr.append(h("td", { class: "n", text: r.variants.length > 1 ? String(r.variants.length) : "" }));
  return tr;
}

function detailRow(r, numbered) {
  // Mirrors the table's columns: the label sits under Word, the variants under Phones.
  const variants = r.variants.length
    ? r.variants.map((v) => h("span", { class: "ipa", dir: "ltr", text: phonesText(v) }))
    : [h("span", { class: "muted", text: "No transcription" })];
  const label = h("td", { class: "k", text: r.variants.length > 1 ? "Variants" : "Phones" });
  const body = h("td", {}, h("div", { class: "variants" }, variants),
    h("span", { class: "sub", text: r.kind === "number" ? "Numbers are not read out" : sourceLong(r.source) }),
    r.kind === "word" ? h("div", { class: "detail-actions" },
      h("button", { class: "secondary small", text: "Compare across dictionaries", onclick: () => goCompare(r.text) })) : null);
  return numbered ? h("tr", { class: "detail" }, h("td"), label, body, h("td"), h("td"))
    : h("tr", { class: "detail" }, label, body, h("td"));
}

function fillRows(tbody, rows, st, opts) {
  const frag = document.createDocumentFragment();
  const prev = opts.prev || [];
  const first = rows.length ? rows[0].i : 0;
  for (const { r, i } of rows) {
    const old = prev[i];
    const fresh = prev.length > 0 && (!old || old.text !== r.text || phonesText(old.phones) !== phonesText(r.phones));
    frag.append(wordRow(r, i, st, { ...opts, fresh, first }));
    if (i === st.sel) frag.append(detailRow(r, opts.numbered));
  }
  tbody.replaceChildren(frag);
}

function bindRows(tbody, st, rerender) {
  const toggle = (i) => {
    st.sel = st.sel === i ? -1 : i;
    rerender();
    tbody.querySelector(`tr[data-i="${i}"]`)?.focus();
  };
  tbody.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    const tr = e.target.closest("tr[data-i]");
    if (tr && !window.getSelection().toString()) toggle(Number(tr.dataset.i));
  });
  tbody.addEventListener("keydown", (e) => {
    const tr = e.target.closest("tr[data-i]");
    if (!tr) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      let next = e.key === "ArrowDown" ? tr.nextElementSibling : tr.previousElementSibling;
      while (next && !next.dataset.i) next = e.key === "ArrowDown" ? next.nextElementSibling : next.previousElementSibling;
      if (next) { tr.tabIndex = -1; next.tabIndex = 0; next.focus(); }
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggle(Number(tr.dataset.i));
    }
  });
}

function stateBlock(title, ...body) {
  return [h("h3", { text: title }), ...body];
}

function errorBlock(err, retry) {
  return stateBlock(err.title || "That didn't work", h("p", { class: "error", text: err.message }),
    err.detail ? h("details", { class: "detail-text" }, h("summary", { text: "Technical details" }), h("p", { text: err.detail })) : null,
    h("button", { class: "secondary", text: "Try again", onclick: retry }));
}

function asError(e, title) {
  return { message: e.message, detail: e.detail || "", title };
}

function downloadBlock(d, done) {
  return stateBlock(`Downloading ${d.name}${d.size ? ` (${mb(d.size)})` : ""}`,
    h("p", { text: "Each dictionary is downloaded once, then works offline." }),
    h("div", { class: "progress" }, h("i")), done);
}

function renderNotes(el, notes) {
  el.hidden = !notes?.length;
  el.textContent = (notes || []).join(" ");
}

/* ---- Transcribe ---------------------------------------------------------------- */

let tTimer;
function onTextInput() {
  clearTimeout(tTimer);
  tTimer = setTimeout(runTranscribe, 160);
  $("#t-clear").hidden = !$("#t-input").value;
  savePrefs();
}

async function runTranscribe() {
  const t = S.t;
  const text = $("#t-input").value;
  const seq = ++t.seq;
  t.stale = false;
  if (!S.dict) return;
  if (!text.trim()) {
    Object.assign(t, { rows: [], prev: [], sel: -1, notes: [], error: null, busy: false });
    return renderTranscribe();
  }
  t.error = null;
  const slow = setTimeout(() => { if (seq === t.seq) { t.busy = true; renderTranscribeHead(); } }, 200);
  try {
    if (!S.dict.installed) {
      t.downloading = S.dict;
      renderTranscribe();
      await download(S.dict);
      t.downloading = null;
      if (seq !== t.seq) return;
    }
    const res = await api("transcribe", { dictionary: S.dict.id, text, backend: S.backend, use_lexicon: S.lexicon });
    if (seq !== t.seq) return;
    t.prev = t.rows;
    t.rows = res.rows;
    t.notes = res.notes;
    if (t.sel >= t.rows.length) t.sel = -1;
  } catch (e) {
    if (seq === t.seq) { t.error = asError(e, t.downloading ? `Couldn't download ${t.downloading.name}` : "Couldn't transcribe the text"); t.downloading = null; }
  } finally {
    clearTimeout(slow);
    if (seq === t.seq) { t.busy = false; renderTranscribe(); }
  }
}

function renderTranscribeHead() {
  const d = S.dict;
  $("#t-notation").textContent = d ? `${d.notation}${S.t.busy ? " · working…" : ""}` : "";
}

function emptyTranscribe() {
  const legend = h("ul", { class: "legend" },
    h("li", {}, h("span", {}, sourceCell("lexicon")), h("span", { text: "The word is in the dictionary." })),
    h("li", {}, h("span", {}, sourceCell("fst")), h("span", { text: "Not in the dictionary, so a model predicted it (FST or ByT5)." })),
    h("li", {}, h("span", { class: "flag-key", text: "Red note" }), h("span", { text: "Something was lost: letters the model does not know, or numbers, which are not read out." })));
  return stateBlock("Each word appears here with its phones",
    h("p", { text: "Type or paste text on the left. You can see where every pronunciation comes from:" }), legend);
}

function renderTranscribe() {
  const t = S.t;
  const table = $("#t-table");
  const state = $("#t-state");
  renderTranscribeHead();
  renderNotes($("#t-notes"), t.notes);
  $("#t-copy").disabled = !t.rows.some((r) => r.phones);
  state.replaceChildren();
  table.hidden = true;
  if (!S.dict) { /* still loading */ }
  else if (t.downloading) state.append(...downloadBlock(t.downloading));
  else if (t.error) state.append(...errorBlock(t.error, runTranscribe));
  else if (!$("#t-input").value.trim()) state.append(...emptyTranscribe());
  else if (!t.rows.length && !t.busy) state.append(...stateBlock("No words in this text", h("p", { text: "Only spaces and punctuation were found." })));
  else {
    table.hidden = false;
    fillRows($("#t-table tbody"), t.rows.map((r, i) => ({ r, i })), t, { prev: t.prev });
    t.prev = [];
  }
  renderStatus();
  renderExport();
}

async function copyTranscription() {
  const text = S.t.rows.filter((r) => r.phones).map((r) => phonesText(r.phones)).join(" ");
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = h("textarea", {}, text);
    document.body.append(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  say("Copied the transcription");
}

/* ---- Word list ------------------------------------------------------------------ */

function firstField(line, sep) {
  if (!sep) return line;
  if (line.startsWith('"')) {
    const end = line.indexOf('"', 1);
    return end > 0 ? line.slice(1, end) : line.slice(1);
  }
  const k = line.indexOf(sep);
  return k < 0 ? line : line.slice(0, k);
}

function parseWords(name, text) {
  const sep = /\.tsv$/i.test(name) ? "\t" : /\.csv$/i.test(name) ? "," : null;
  const words = text.replace(/^﻿/, "").split(/\r?\n/).map((l) => firstField(l, sep).trim()).filter(Boolean);
  if (words.length && /^(words?|orthography|tokens?|items?)$/i.test(words[0])) words.shift();
  return words;
}

async function openFile(file) {
  if (!file) return;
  let text;
  try {
    text = await file.text();
  } catch {
    S.l.error = { message: `${file.name} could not be read.`, title: "Couldn't open the file" };
    return renderList();
  }
  const words = parseWords(file.name, text);
  Object.assign(S.l, { name: file.name, words, rows: [], sel: -1, limit: LIST_PAGE, error: null, find: "", filter: "all" });
  $("#l-find").value = "";
  syncClear($("#l-find"));
  if (!words.length) {
    S.l.error = { message: `No words found in ${file.name}. Put one word per line, or the words in the first column.`, title: "No words in this file" };
    return renderList();
  }
  runList();
}

async function runList() {
  const l = S.l;
  l.stale = false;
  if (!l.words.length || !S.dict) return renderList();
  const seq = ++l.seq;
  Object.assign(l, { busy: true, done: 0, rows: [], error: null, notes: [] });
  renderList();
  try {
    if (!S.dict.installed) {
      l.downloading = S.dict;
      renderList();
      await download(S.dict);
      l.downloading = null;
    }
    for (let k = 0; k < l.words.length; k += LIST_CHUNK) {
      const res = await api("words", { dictionary: S.dict.id, words: l.words.slice(k, k + LIST_CHUNK),
                                       backend: S.backend, use_lexicon: S.lexicon });
      if (seq !== l.seq) return;
      l.rows.push(...res.rows);
      l.notes = res.notes;
      l.done = l.rows.length;
      renderListProgress();
    }
  } catch (e) {
    if (seq === l.seq) { l.error = asError(e, l.downloading ? `Couldn't download ${l.downloading.name}` : "Couldn't transcribe the list"); l.downloading = null; }
  } finally {
    if (seq === l.seq) { l.busy = false; renderList(); }
  }
}

function listCounts() {
  const c = { all: S.l.rows.length, lexicon: 0, predicted: 0, flagged: 0 };
  for (const r of S.l.rows) {
    if (r.source === "lexicon") c.lexicon++;
    else if (r.source) c.predicted++;
    if (r.flag) c.flagged++;
  }
  return c;
}

function renderListFilter() {
  const c = listCounts();
  const box = $("#l-filter");
  box.hidden = !S.l.rows.length;
  box.replaceChildren(...[["all", "All"], ["lexicon", "Dictionary"], ["predicted", "Predicted"], ["flagged", "Flagged"]]
    .map(([k, label]) => h("button", { "data-f": k, "aria-pressed": String(S.l.filter === k),
                                       text: `${label} ${NUM.format(c[k])}` })));
  $("#l-find").parentElement.hidden = !S.l.rows.length;
}

function renderListProgress() {
  const l = S.l;
  const bar = $("#l-state .progress i");
  if (bar) bar.style.width = `${Math.max(4, (100 * l.done) / l.words.length)}%`;
  const p = $("#l-state p.count");
  if (p) p.textContent = `${NUM.format(l.done)} of ${NUM.format(l.words.length)} words`;
  renderStatus();
}

function renderList() {
  const l = S.l;
  const table = $("#l-table");
  const state = $("#l-state");
  $("#l-name").textContent = l.name ? `${l.name} · ${plural(l.words.length, "word")}` : "";
  $("#l-open").hidden = !l.words.length;
  $("#l-open").textContent = "Open another list…";
  renderNotes($("#l-notes"), l.notes);
  renderListFilter();
  state.replaceChildren();
  state.classList.remove("drop");
  table.hidden = true;
  if (l.downloading) state.append(...downloadBlock(l.downloading));
  else if (l.error) state.append(...errorBlock(l.error, () => (l.words.length ? runList() : $("#l-open").click())));
  else if (!l.words.length) {
    state.classList.add("drop");
    state.append(...stateBlock("Transcribe a word list",
      h("p", { text: "Open a .txt file with one word per line, or a .csv or .tsv file with the words in the first column. You can also drop the file here." }),
      h("button", { class: "secondary", text: "Open word list…", onclick: () => $("#l-file").click() })));
  } else if (l.busy) {
    state.append(...stateBlock(`Transcribing ${l.name}`, h("p", { class: "count", text: `0 of ${NUM.format(l.words.length)} words` }),
      h("div", { class: "progress det" }, h("i"))));
    renderListProgress();
  } else {
    const q = l.find.trim().toLocaleLowerCase();
    const keep = (r) => (l.filter === "all" || (l.filter === "lexicon" && r.source === "lexicon")
      || (l.filter === "predicted" && r.source && r.source !== "lexicon") || (l.filter === "flagged" && r.flag))
      && (!q || r.text.toLocaleLowerCase().includes(q));
    const shown = [];
    for (let i = 0; i < l.rows.length; i++) if (keep(l.rows[i])) shown.push({ r: l.rows[i], i });
    if (!shown.length) {
      state.append(...stateBlock("No words match", h("p", { text: "Change the filter or the search." })));
    } else {
      table.hidden = false;
      fillRows($("#l-table tbody"), shown.slice(0, l.limit), l, { numbered: true });
      if (shown.length > l.limit) {
        state.append(h("button", { class: "secondary show-more", text: `Show ${NUM.format(Math.min(LIST_PAGE, shown.length - l.limit))} more of ${NUM.format(shown.length - l.limit)}`,
                                   onclick: () => { l.limit += LIST_PAGE; renderList(); } }));
      }
    }
  }
  renderStatus();
  renderExport();
}

/* ---- Compare --------------------------------------------------------------------- */

let cTimer;
function goCompare(word) {
  $("#c-word").value = word;
  setView("compare");
  $("#c-word").focus();
}

async function runCompare() {
  const c = S.c;
  const word = $("#c-word").value.trim();
  const seq = ++c.seq;
  if (!S.lang) return;
  if (!word) {
    c.data = null;
    c.error = null;
    return renderCompare();
  }
  c.busy = true;
  try {
    const data = await api("compare", { lang: S.lang, word });
    if (seq !== c.seq) return;
    c.data = data;
    c.error = null;
  } catch (e) {
    if (seq === c.seq) c.error = asError(e, "Couldn't compare the word");
  } finally {
    if (seq === c.seq) { c.busy = false; renderCompare(); }
  }
}

function compareRow(row) {
  const d = row.dictionary;
  const inUse = d.id === S.dict?.id;
  let phones;
  if (!d.installed) phones = h("span", { class: "muted", text: "Not downloaded" });
  else if (!row.variants.length) phones = h("span", { class: "nophones", text: "No transcription" });
  else phones = h("div", { class: "variants" }, row.variants.map((v) => h("span", { class: "ipa", text: phonesText(v) })));
  const cell = h("td", { class: "ipa-cell" }, phones, row.flag ? h("span", { class: "flag", text: row.flag }) : null);
  let action;
  if (!d.installed) {
    const btn = h("button", { class: "secondary small", text: d.size ? `Download · ${mb(d.size)}` : "Download",
      onclick: async (e) => {
        e.target.disabled = true;
        e.target.textContent = "Downloading…";
        try { await download(d); runCompare(); } catch (err) { say(err.message, 0); e.target.disabled = false; e.target.textContent = "Download"; }
      } });
    action = inUse ? h("span", { class: "compare-act" }, h("span", { class: "in-use", text: "In use" }), btn) : btn;
  } else if (inUse) action = h("span", { class: "in-use", text: "In use" });
  else action = h("button", { class: "secondary small", text: "Use", onclick: () => { setDictionary(d.id); say(`Using ${d.name}`); } });
  return h("tr", { class: row.flag ? "flagged" : null },
    h("td", {}, h("span", { text: d.name }), h("span", { class: "sub", text: `${d.source} · ${plural(d.words, "word")}` })),
    h("td", { text: d.notation }), cell,
    h("td", {}, d.installed && row.variants.length ? sourceCell(row.source) : h("span", { class: "muted", text: "–" })),
    h("td", { class: "num", text: pct(d.per) }), h("td", { class: "c-act" }, action));
}

function renderCompare() {
  const c = S.c;
  const table = $("#c-table");
  const state = $("#c-state");
  state.replaceChildren();
  table.hidden = true;
  if (c.error) state.append(...errorBlock(c.error, runCompare));
  else if (!c.data) {
    state.append(...stateBlock(`Compare a word across ${langName(S.lang)} dictionaries`,
      h("p", { text: "Type a word to see how each dictionary of this language pronounces it, with every variant. Dictionaries use different notations, so a difference is not always an error." })));
  } else {
    table.hidden = false;
    $("#c-table tbody").replaceChildren(...c.data.rows.map(compareRow));
  }
  renderStatus();
}

/* ---- Dictionaries -------------------------------------------------------------- */

async function loadAllDicts() {
  if (S.d.all) return renderDicts();
  $("#d-state").replaceChildren(...stateBlock("Loading the dictionary list", h("div", { class: "progress" }, h("i"))));
  try {
    S.d.all = await api("dictionaries");
    S.d.error = null;
  } catch (e) {
    S.d.error = asError(e, "Couldn't load the dictionary list");
  }
  renderDicts();
}

function renderDicts() {
  const d = S.d;
  const table = $("#d-table");
  const state = $("#d-state");
  state.replaceChildren();
  table.hidden = true;
  if (d.error) {
    state.append(...errorBlock(d.error, () => { d.error = null; loadAllDicts(); }));
    return renderStatus();
  }
  if (!d.all) return;
  const q = d.find.trim().toLocaleLowerCase();
  const rows = d.all.filter((x) => (d.filter === "all" || x.installed)
    && (!q || [x.language, x.lang, x.name, x.notation, x.source].some((v) => String(v).toLocaleLowerCase().includes(q))));
  const { k, dir } = d.sort;
  rows.sort((a, b) => {
    if (k === "language" && (a.language === a.lang) !== (b.language === b.lang)) return a.language === a.lang ? 1 : -1;
    const av = a[k] ?? Infinity;
    const bv = b[k] ?? Infinity;
    const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv));
    return cmp * dir || a.language.localeCompare(b.language) || a.name.localeCompare(b.name);
  });
  for (const b of $$("#d-table .sort")) {
    if (b.dataset.k === k) b.setAttribute("aria-sort", dir > 0 ? "ascending" : "descending");
    else b.removeAttribute("aria-sort");
  }
  $("#d-meta").textContent = `${NUM.format(rows.length)} of ${plural(d.all.length, "dictionary", "dictionaries")}`;
  if (!rows.length) {
    state.append(...stateBlock("No dictionaries match", h("p", { text: "Try an English language name (German), a code (deu) or a dictionary name." })));
    return renderStatus();
  }
  table.hidden = false;
  $("#d-table tbody").replaceChildren(...rows.map((x) => {
    let action;
    if (x.installed) {
      action = h("button", { class: "secondary small row-act", text: "Use", "aria-label": `Use ${x.name}`, onclick: async () => {
        await setLanguage(x.lang, x.id);
        setView("transcribe");
        focusView();
      } });
    } else {
      action = h("button", { class: "secondary small", text: x.size ? `Download · ${mb(x.size)}` : "Download",
        onclick: async (e) => {
          e.target.disabled = true;
          e.target.textContent = "Downloading…";
          try { await download(x); renderDicts(); } catch (err) { say(err.message, 0); e.target.disabled = false; e.target.textContent = "Download"; }
        } });
    }
    const named = x.language !== x.lang;
    return h("tr", {},
      h("td", { class: named ? null : "muted", text: named ? x.language : "No English name" }), h("td", { class: "muted", text: x.lang }),
      h("td", {}, x.name, x.default ? h("span", { class: "sub", text: "Default for this language" }) : null),
      h("td", { text: x.notation }), h("td", { class: "num", text: NUM.format(x.words) }),
      h("td", { text: x.model }), h("td", { class: "num", text: pct(x.per) }), h("td", { class: "c-act" }, action));
  }));
  renderStatus();
}

/* ---- status bar -------------------------------------------------------------------- */

function countsFor(rows) {
  const c = { words: 0, lexicon: 0, fst: 0, byt5: 0, flagged: 0 };
  for (const r of rows) {
    if (r.kind === "word") c.words++;
    if (r.source) c[r.source]++;
    if (r.flag) c.flagged++;
  }
  return c;
}

function stat(n, label, warn) {
  return h("span", { class: warn && n ? "warn" : null }, h("b", { text: NUM.format(n) }), ` ${label}`);
}

function renderStatus() {
  const left = $("#s-left");
  const right = $("#s-right");
  left.replaceChildren();
  right.replaceChildren();
  const showDict = S.view === "transcribe" || S.view === "list";
  const rows = S.view === "transcribe" ? S.t.rows : S.view === "list" ? S.l.rows : null;
  if (rows && rows.length) {
    const c = countsFor(rows);
    left.append(stat(c.words, c.words === 1 ? "word" : "words"), stat(c.lexicon, "dictionary"),
      stat(c.fst, "FST"), stat(c.byt5, "ByT5"), stat(c.flagged, "flagged", true));
  } else if (S.view === "list" && S.l.busy) {
    left.append(`Transcribing ${NUM.format(S.l.done)} of ${NUM.format(S.l.words.length)} words`);
  } else if (S.view === "compare" && S.dicts.length) {
    const n = S.dicts.filter((d) => d.installed).length;
    left.append(`${plural(S.dicts.length, `${langName(S.lang)} dictionary`, `${langName(S.lang)} dictionaries`)} · ${NUM.format(n)} downloaded`);
  } else if (S.view === "dictionaries" && S.d.all) {
    const n = S.d.all.filter((d) => d.installed).length;
    left.append(`${plural(S.d.all.length, "dictionary", "dictionaries")} · ${plural(S.languages.length, "language")} · ${NUM.format(n)} downloaded`);
  }
  if (showDict && S.dict) {
    right.append(h("span", { text: S.dict.name }),
      h("button", { class: "info-btn", id: "per-btn", "aria-haspopup": "dialog", "aria-expanded": "false",
                    onclick: (e) => togglePop($("#per-pop"), e.currentTarget) },
        `Test PER ${pct(S.dict.per)}`, icon("i-info", "")));
  }
}

/* ---- popovers ------------------------------------------------------------------------ */

let pop = null;

function placePop(el, anchor) {
  const r = anchor.getBoundingClientRect();
  const w = el.offsetWidth;
  const hgt = el.offsetHeight;
  const left = Math.max(8, Math.min(r.left, innerWidth - w - 8));
  const below = r.bottom + 6;
  const top = below + hgt > innerHeight - 8 ? Math.max(8, r.top - hgt - 6) : below;
  el.style.left = `${left}px`;
  el.style.top = `${top}px`;
}

function togglePop(el, anchor, onOpen) {
  if (pop?.el === el) return closePop(true);
  closePop();
  el.hidden = false;
  onOpen?.();
  placePop(el, anchor);
  anchor.setAttribute("aria-expanded", "true");
  pop = { el, anchor };
}

function closePop(returnFocus) {
  if (!pop) return;
  pop.el.hidden = true;
  pop.anchor.setAttribute("aria-expanded", "false");
  if (returnFocus && document.contains(pop.anchor)) pop.anchor.focus();
  pop = null;
}

function listNav(box, input, choose) {
  const opts = () => $$(".opt:not([aria-disabled='true'])", box);
  const active = () => $(".opt[aria-selected='true']", box);
  const setActive = (o) => {
    active()?.setAttribute("aria-selected", "false");
    if (!o) return;
    o.setAttribute("aria-selected", "true");
    o.scrollIntoView({ block: "nearest" });
    input?.setAttribute("aria-activedescendant", o.id);
  };
  box.addEventListener("mousemove", (e) => { const o = e.target.closest(".opt"); if (o && o !== active() && o.getAttribute("aria-disabled") !== "true") setActive(o); });
  box.addEventListener("click", (e) => { const o = e.target.closest(".opt"); if (o && o.getAttribute("aria-disabled") !== "true") choose(o.dataset.v); });
  (input || box).addEventListener("keydown", (e) => {
    const all = opts();
    const k = all.indexOf(active());
    if (e.key === "ArrowDown") { e.preventDefault(); setActive(all[Math.min(all.length - 1, k + 1)]); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive(all[Math.max(0, k - 1)]); }
    else if (e.key === "Enter" && active()) { e.preventDefault(); choose(active().dataset.v); }
  });
  return { first: () => setActive(opts().find((o) => o.classList.contains("cur")) || opts()[0]) };
}

function langOption(l) {
  return h("div", { class: `opt${l.tag === S.lang ? " cur" : ""}`, role: "option", id: `lo-${l.tag}`, "data-v": l.tag, "aria-selected": "false" },
    icon("i-check", "check"), h("span", { class: "name", text: l.name }),
    h("span", { class: "tag", text: l.count > 1 ? `${l.tag} · ${l.count} dictionaries` : l.tag }));
}

let langNav;
function renderLangOpts() {
  const q = $("#lang-find").value.trim().toLocaleLowerCase();
  const box = $("#lang-opts");
  const match = (l) => l.name.toLocaleLowerCase().includes(q) || l.tag.startsWith(q) || l.alias === q;
  const kids = [];
  if (!q) {
    const recent = S.recent.map((t) => S.byTag.get(t)).filter(Boolean);
    if (recent.length) kids.push(h("div", { class: "group", text: "Recent" }), ...recent.map(langOption));
    kids.push(h("div", { class: "group", text: "All languages" }), ...S.languages.map(langOption));
  } else {
    const found = S.languages.filter(match);
    found.sort((a, b) => (b.alias === q || b.tag === q) - (a.alias === q || a.tag === q)
      || a.name.toLocaleLowerCase().startsWith(q) * -1 + b.name.toLocaleLowerCase().startsWith(q) * 1);
    if (found.length) kids.push(...found.map(langOption));
    else kids.push(h("div", { class: "none", text: `No language matches “${q}”. Try an English name (German) or a code (de, deu).` }));
  }
  box.replaceChildren(...kids);
  langNav.first();
}

function renderDictOpts() {
  const box = $("#dict-opts");
  box.replaceChildren(...S.dicts.map((d) => h("div", { class: `opt two${d.id === S.dict?.id ? " cur" : ""}`, role: "option", id: `do-${d.id.replace("/", "-")}`, "data-v": d.id, "aria-selected": "false" },
    h("span", { class: "row1" }, icon("i-check", "check"), h("span", { class: "name", text: d.name }),
      d.default ? h("span", { class: "def", text: "Default" }) : null,
      h("span", { class: "side", text: d.installed ? "Downloaded" : d.size ? mb(d.size) : "Not downloaded" })),
    h("span", { class: "sub", text: `${d.notation} · ${plural(d.words, "word")} · ${d.model} · test PER ${pct(d.per)}` }))));
  dictNav.first();
}

let dictNav;
function renderModelOpts() {
  const d = S.dict;
  const radio = (value, label, sub, disabled) => h("label", { class: `choice${disabled ? " off" : ""}` },
    h("input", { type: "radio", name: "backend", value, checked: S.backend === value, disabled }),
    h("span", {}, h("span", { text: label }), h("span", { class: "sub", text: sub })));
  $("#model-opts").replaceChildren(
    radio("auto", "Auto", d ? `The better model for ${d.name}: ${d.model}` : "The better model for this dictionary"),
    radio("fst", "FST", "Phonetisaurus model trained on this dictionary. Fast."),
    radio("byt5", "ByT5", S.info?.byt5 ? "Neural model. Slower." : "Not installed. Add it with pip install 'agentg2p[neural]'", !S.info?.byt5),
    h("div", { class: "sep" }),
    h("label", { class: "choice" }, h("input", { type: "checkbox", id: "lexicon-box", checked: S.lexicon }),
      h("span", {}, h("span", { text: "Look words up in the dictionary first" }),
        h("span", { class: "sub", text: "When off, the model predicts every word." }))));
}

/* ---- export ---------------------------------------------------------------------------- */

async function exportTSV() {
  const fromList = S.view === "list";
  const rows = fromList ? S.l.rows : S.t.rows;
  if (!rows.length || !S.dict) return;
  const base = fromList ? S.l.name.replace(/\.[^.]+$/, "") || "words" : "agentg2p";
  const name = `${base}-${S.dict.name}.tsv`;
  const clean = (v) => String(v).replace(/[\t\r\n]+/g, " ");
  const lines = ["word\tphones\tsource\tvariants\tnote", ...rows.map((r) => [
    r.text, (r.phones || []).join(" "), r.source || "", r.variants.map((v) => v.join(" ")).join(" | "), r.flag || "",
  ].map(clean).join("\t"))];
  const content = `${lines.join("\n")}\n`;
  try {
    const res = await api("save", { name, content });
    if (res.download) {
      const a = h("a", { href: URL.createObjectURL(new Blob([content], { type: "text/tab-separated-values" })), download: name });
      document.body.append(a);
      a.click();
      a.remove();
      say(`Exported ${name}`);
    } else if (res.saved) {
      say(`Saved ${res.saved}`);
    }
  } catch (e) {
    say(e.message, 0);
  }
}

/* ---- start --------------------------------------------------------------------------- */

function guessLanguage() {
  const [base, region] = (navigator.language || "en-US").toLowerCase().split("-");
  const tags = S.languages.filter((l) => l.alias === base || l.tag === base);
  const exact = tags.find((l) => region && l.tag.endsWith(`-${region}`));
  return (exact || tags.find((l) => !l.tag.includes("-")) || tags[0] || S.byTag.get("eng-us") || S.languages[0]).tag;
}

function fatal(message) {
  $("main").replaceChildren(h("div", { class: "state" }, ...errorBlock({ ...message, title: "agentg2p could not start" }, () => location.reload())));
  $("#lang-label").textContent = "–";
}

function syncClear(input) {
  const btn = input.parentElement.querySelector(".clear");
  if (btn) btn.hidden = !input.value;
}

function bind() {
  for (const wrap of $$(".search")) {
    const input = $("input", wrap);
    input.addEventListener("input", () => syncClear(input));
    $(".clear", wrap).addEventListener("click", () => {
      input.value = "";
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.focus();
    });
  }
  for (const b of $$(".views button")) b.addEventListener("click", () => setView(b.dataset.view));
  $("#t-input").addEventListener("input", onTextInput);
  $("#t-clear").addEventListener("click", () => { $("#t-input").value = ""; onTextInput(); $("#t-input").focus(); });
  $("#t-copy").addEventListener("click", copyTranscription);
  for (const b of $$("#view-transcribe .seg button")) {
    b.addEventListener("click", () => {
      S.joined = b.dataset.join === "1";
      for (const x of $$("#view-transcribe .seg button")) x.setAttribute("aria-pressed", String(x === b));
      renderTranscribe();
      if (S.l.rows.length) renderList();
      savePrefs();
    });
  }
  $("#export-btn").addEventListener("click", exportTSV);

  langNav = listNav($("#lang-opts"), $("#lang-find"), (tag) => { closePop(true); setLanguage(tag); });
  dictNav = listNav($("#dict-opts"), $("#dict-opts"), (id) => { closePop(true); setDictionary(id); });
  $("#lang-find").addEventListener("input", renderLangOpts);
  $("#lang-btn").addEventListener("click", (e) => togglePop($("#lang-pop"), e.currentTarget, () => {
    $("#lang-find").value = "";
    syncClear($("#lang-find"));
    renderLangOpts();
    requestAnimationFrame(() => $("#lang-find").focus());
  }));
  $("#dict-btn").addEventListener("click", (e) => togglePop($("#dict-pop"), e.currentTarget, () => {
    renderDictOpts();
    requestAnimationFrame(() => $("#dict-opts").focus());
  }));
  $("#dict-opts").tabIndex = 0;
  $("#model-btn").addEventListener("click", (e) => togglePop($("#model-pop"), e.currentTarget, () => {
    renderModelOpts();
    requestAnimationFrame(() => $("#model-opts input:checked")?.focus());
  }));
  $("#model-opts").addEventListener("change", (e) => {
    if (e.target.name === "backend") S.backend = e.target.value;
    if (e.target.id === "lexicon-box") S.lexicon = e.target.checked;
    renderToolbar();
    renderModelOpts();
    $("#model-opts input:checked")?.focus();
    savePrefs();
    refresh();
  });

  $("#l-open").addEventListener("click", () => $("#l-file").click());
  $("#l-file").addEventListener("change", (e) => { openFile(e.target.files[0]); e.target.value = ""; });
  const wrap = $("#l-wrap");
  wrap.addEventListener("dragover", (e) => { e.preventDefault(); $("#l-state").classList.add("over"); });
  wrap.addEventListener("dragleave", () => $("#l-state").classList.remove("over"));
  wrap.addEventListener("drop", (e) => { e.preventDefault(); $("#l-state").classList.remove("over"); openFile(e.dataTransfer.files[0]); });
  $("#l-filter").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { S.l.filter = b.dataset.f; S.l.limit = LIST_PAGE; renderList(); } });
  $("#l-find").addEventListener("input", (e) => { S.l.find = e.target.value; S.l.limit = LIST_PAGE; renderList(); });

  $("#c-word").addEventListener("input", () => { clearTimeout(cTimer); cTimer = setTimeout(runCompare, 220); });
  $("#c-word").addEventListener("keydown", (e) => { if (e.key === "Enter") { clearTimeout(cTimer); runCompare(); } });

  $("#d-find").addEventListener("input", (e) => { S.d.find = e.target.value; renderDicts(); });
  $("#d-filter").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    S.d.filter = b.dataset.f;
    for (const x of $$("#d-filter button")) x.setAttribute("aria-pressed", String(x === b));
    renderDicts();
  });
  for (const b of $$("#d-table .sort")) {
    b.addEventListener("click", () => {
      const s = S.d.sort;
      S.d.sort = { k: b.dataset.k, dir: s.k === b.dataset.k ? -s.dir : 1 };
      renderDicts();
    });
  }

  bindRows($("#t-table tbody"), S.t, renderTranscribe);
  bindRows($("#l-table tbody"), S.l, renderList);

  document.addEventListener("mousedown", (e) => {
    if (pop && !pop.el.contains(e.target) && !pop.anchor.contains(e.target)) closePop();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && pop) { e.preventDefault(); closePop(true); return; }
    const mod = e.metaKey || e.ctrlKey;
    if (mod && /^[1-4]$/.test(e.key)) { e.preventDefault(); setView(VIEWS[Number(e.key) - 1]); focusView(); }
    if (mod && e.key.toLowerCase() === "l" && S.view !== "dictionaries") { e.preventDefault(); $("#lang-btn").click(); }
  });
  addEventListener("resize", () => pop && placePop(pop.el, pop.anchor));
}

async function init() {
  bind();
  let prefs = {};
  try {
    [S.info, prefs, S.languages] = await Promise.all([api("info"), api("prefs"), api("languages")]);
  } catch (e) {
    return fatal(asError(e));
  }
  S.byTag = new Map(S.languages.map((l) => [l.tag, l]));
  $("#lang-find").placeholder = `Search ${NUM.format(S.languages.length)} languages or codes`;
  const q = new URLSearchParams(location.search);
  S.backend = ["auto", "fst", "byt5"].includes(prefs.backend) ? prefs.backend : "auto";
  if (S.backend === "byt5" && !S.info.byt5) S.backend = "auto";
  S.lexicon = prefs.lexicon !== false;
  S.joined = !!prefs.joined;
  S.recent = Array.isArray(prefs.recent) ? prefs.recent : [];
  S.dictByLang = prefs.dictByLang && typeof prefs.dictByLang === "object" ? prefs.dictByLang : {};
  for (const x of $$("#view-transcribe .seg button")) x.setAttribute("aria-pressed", String((x.dataset.join === "1") === S.joined));
  $("#t-input").value = q.get("text") ?? prefs.text ?? "";
  $("#t-clear").hidden = !$("#t-input").value;
  if (q.get("word")) $("#c-word").value = q.get("word");
  if (q.get("select")) S.t.sel = Number(q.get("select"));
  S.view = VIEWS.includes(q.get("view")) ? q.get("view") : VIEWS.includes(prefs.view) ? prefs.view : "transcribe";
  setView(S.view);
  const lang = [q.get("lang"), prefs.lang].find((t) => t && S.byTag.has(t)) || guessLanguage();
  await setLanguage(lang, q.get("dict"));
  if (S.view === "transcribe") $("#t-input").focus();
}

init();
