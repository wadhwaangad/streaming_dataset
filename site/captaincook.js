const state = {
  records: [],
  query: "",
  status: "all",
  deviation: "all"
};

const recordsEl = document.querySelector("#records");
const searchInput = document.querySelector("#searchInput");
const statusFilter = document.querySelector("#statusFilter");
const deviationFilter = document.querySelector("#deviationFilter");

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function textForSearch(record) {
  const annotation = record.annotation || {};
  const curation = record.curation || {};
  return [
    record.title,
    record.status,
    record.deviation_type,
    record.summary,
    annotation.goal,
    annotation.current_state,
    annotation.expected_next_state,
    annotation.assistant_response,
    curation.visual_evidence,
    curation.notes
  ].join(" ").toLowerCase();
}

function card(record) {
  const annotation = record.annotation || {};
  const curation = record.curation || {};
  const window = annotation.intervention_window;
  const timing = window
    ? `${window.start_seconds}s-${window.end_seconds}s`
    : annotation.intervention_timing || "source";
  const evidence = curation.visual_evidence
    ? `<p><strong>Evidence:</strong> ${escapeHtml(curation.visual_evidence)}</p>`
    : "";
  const rejection = curation.notes && record.status === "rejected"
    ? `<p><strong>Rejected:</strong> ${escapeHtml(curation.notes)}</p>`
    : "";

  return `
    <article class="record-card">
      <div class="record-topline">
        <span class="pill">${escapeHtml(record.status)}</span>
        <span class="score">${escapeHtml(record.score)}</span>
      </div>
      <h2>${escapeHtml(record.title)}</h2>
      <p>${escapeHtml(record.summary || record.best_use)}</p>
      <div class="meta">
        <span class="pill">${escapeHtml(record.deviation_type)}</span>
        <span class="pill">${escapeHtml(timing)}</span>
        <span class="pill">${escapeHtml(record.rights?.hosting_status || "metadata_only")}</span>
      </div>
      ${evidence}
      ${rejection}
      ${annotation.assistant_response ? `<p>${escapeHtml(annotation.assistant_response)}</p>` : ""}
    </article>
  `;
}

function filteredRecords() {
  return state.records.filter((record) => {
    const matchesQuery = !state.query || textForSearch(record).includes(state.query);
    const matchesStatus = state.status === "all" || record.status === state.status;
    const matchesDeviation = state.deviation === "all" || record.deviation_type === state.deviation;
    return matchesQuery && matchesStatus && matchesDeviation;
  });
}

function render() {
  const records = filteredRecords();
  recordsEl.innerHTML = records.length
    ? records.map(card).join("")
    : `<div class="empty">No CaptainCook4D records match the current filters.</div>`;
}

function populateFilters(records) {
  const deviations = [...new Set(records.map((record) => record.deviation_type).filter(Boolean))].sort();
  deviationFilter.innerHTML += deviations
    .map((type) => `<option value="${escapeHtml(type)}">${escapeHtml(type)}</option>`)
    .join("");
}

async function init() {
  const response = await fetch("data/captaincook_dataset.json");
  if (!response.ok) {
    throw new Error("Could not load data/captaincook_dataset.json. Run the CaptainCook pipeline first.");
  }

  const manifest = await response.json();
  state.records = manifest.records || [];

  document.querySelector("#recordCount").textContent = state.records.length;
  document.querySelector("#curatedCount").textContent = state.records.filter((record) => record.status === "curated").length;
  document.querySelector("#rejectedCount").textContent = state.records.filter((record) => record.status === "rejected").length;
  document.querySelector("#taskText").textContent = manifest.task || "";

  populateFilters(state.records);
  render();
}

searchInput.addEventListener("input", (event) => {
  state.query = event.target.value.trim().toLowerCase();
  render();
});

statusFilter.addEventListener("change", (event) => {
  state.status = event.target.value;
  render();
});

deviationFilter.addEventListener("change", (event) => {
  state.deviation = event.target.value;
  render();
});

init().catch((error) => {
  recordsEl.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
});
