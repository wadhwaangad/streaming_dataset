const state = {
  records: [],
  query: "",
  domain: "all",
  type: "all"
};

const recordsEl = document.querySelector("#records");
const searchInput = document.querySelector("#searchInput");
const domainFilter = document.querySelector("#domainFilter");
const typeFilter = document.querySelector("#typeFilter");

function textForSearch(record) {
  return [
    record.title,
    record.domain,
    record.source_family,
    record.deviation_type,
    record.summary,
    record.status,
    record.rights?.license_status,
    record.rights?.hosting_status,
    record.curation?.visual_evidence,
    ...(record.cue_hits || []).map((hit) => hit.phrase)
  ].join(" ").toLowerCase();
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function card(record) {
  const annotation = record.annotation || {};
  const curation = record.curation || {};
  const rights = record.rights || {};
  const window = annotation.intervention_window;
  const timing = window
    ? `${window.start_seconds}s-${window.end_seconds}s`
    : annotation.intervention_timing || "source";
  const evidence = curation.visual_evidence
    ? `<p><strong>Evidence:</strong> ${escapeHtml(curation.visual_evidence)}</p>`
    : "";
  const rightsLine = rights.hosting_status || rights.license_status
    ? `<span class="pill">${escapeHtml(rights.hosting_status || "rights unknown")}</span>`
    : "";

  const sourceHref = record.intervention_url || record.clip_url || record.url;
  const sourceLabel = record.intervention_url ? "Open 30s before deviation" : "Open source";

  return `
    <article class="record-card">
      <div class="record-topline">
        <span class="pill">${escapeHtml(record.record_type)}</span>
      </div>
      <h2>${escapeHtml(record.title)}</h2>
      <p>${escapeHtml(record.summary || record.best_use)}</p>
      <div class="meta">
        <span class="pill">${escapeHtml(record.domain)}</span>
        <span class="pill">${escapeHtml(record.deviation_type)}</span>
        <span class="pill">${escapeHtml(record.status)}</span>
        <span class="pill">${escapeHtml(timing)}</span>
        ${rightsLine}
      </div>
      ${evidence}
      ${annotation.assistant_response ? `<p>${escapeHtml(annotation.assistant_response)}</p>` : ""}
      ${sourceHref ? `<a href="${escapeHtml(sourceHref)}" target="_blank" rel="noreferrer">${escapeHtml(sourceLabel)}</a>` : ""}
    </article>
  `;
}

function filteredRecords() {
  return state.records.filter((record) => {
    const matchesQuery = !state.query || textForSearch(record).includes(state.query);
    const matchesDomain = state.domain === "all" || record.domain === state.domain;
    const matchesType = state.type === "all" || record.record_type === state.type;
    return matchesQuery && matchesDomain && matchesType;
  });
}

function render() {
  const records = filteredRecords();
  recordsEl.innerHTML = records.length
    ? records.map(card).join("")
    : `<div class="empty">No records match the current filters.</div>`;
}

function populateFilters(records) {
  const domains = [...new Set(records.map((record) => record.domain).filter(Boolean))].sort();
  domainFilter.innerHTML += domains
    .map((domain) => `<option value="${escapeHtml(domain)}">${escapeHtml(domain)}</option>`)
    .join("");
}

async function init() {
  const response = await fetch("data/dataset.json");
  if (!response.ok) {
    throw new Error("Could not load data/dataset.json. Run the pipeline first.");
  }

  const manifest = await response.json();
  state.records = manifest.records || [];

  document.querySelector("#recordCount").textContent = state.records.length;
  document.querySelector("#candidateCount").textContent = state.records.filter((record) => record.record_type === "candidate").length;
  document.querySelector("#sourceCount").textContent = state.records.filter((record) => record.record_type === "source").length;
  document.querySelector("#taskText").textContent = manifest.task || "";

  populateFilters(state.records);
  render();
}

searchInput.addEventListener("input", (event) => {
  state.query = event.target.value.trim().toLowerCase();
  render();
});

domainFilter.addEventListener("change", (event) => {
  state.domain = event.target.value;
  render();
});

typeFilter.addEventListener("change", (event) => {
  state.type = event.target.value;
  render();
});

init().catch((error) => {
  recordsEl.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
});
