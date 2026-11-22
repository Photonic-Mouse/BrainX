// BrainX — Minimal, Clean Tabular MRI Analysis Client Logic

let currentFile = null;
let currentSamplePath = null;

const CLASS_BADGES = {
  "Mild Demented": "badge-mild",
  "Moderate Demented": "badge-mod",
  "Non Demented": "badge-non",
  "Very Mild Demented": "badge-verymild"
};

document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    lucide.createIcons();
  }

  setupDropZone();
  setupAnalyze();
  fetchSamples();
  checkStatus();
});

// Check API Health
async function checkStatus() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      const txt = document.getElementById("system-status-text");
      if (txt) {
        txt.textContent = `${data.device.toUpperCase()} Active (3 Models)`;
      }
    }
  } catch (e) {
    console.warn("Status check:", e);
  }
}

// Drag & Drop
function setupDropZone() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const btnClear = document.getElementById("btn-clear-file");

  ["dragenter", "dragover"].forEach(ev => {
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(ev => {
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  btnClear.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFile();
  });
}

function handleFile(file) {
  const allowed = ["image/jpeg", "image/png", "image/jpg"];
  if (!allowed.includes(file.type)) {
    alert("Please select a valid PNG or JPEG image.");
    return;
  }

  currentFile = file;
  currentSamplePath = null;
  document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById("preview-image");
    img.src = e.target.result;

    const temp = new Image();
    temp.src = e.target.result;
    temp.onload = () => {
      document.getElementById("preview-dimensions").textContent = `${temp.naturalWidth} x ${temp.naturalHeight} px (${(file.size / 1024).toFixed(1)} KB)`;
    };

    document.getElementById("preview-filename").textContent = file.name;
    document.getElementById("drop-zone-idle").style.display = "none";
    document.getElementById("drop-zone-preview").style.display = "flex";
    document.getElementById("btn-analyze").disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearFile() {
  currentFile = null;
  currentSamplePath = null;
  document.getElementById("file-input").value = "";
  document.getElementById("preview-image").src = "";
  document.getElementById("drop-zone-idle").style.display = "block";
  document.getElementById("drop-zone-preview").style.display = "none";
  document.getElementById("btn-analyze").disabled = true;
  document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
}

// Fetch Sample Images for Quick Testing
async function fetchSamples() {
  const container = document.getElementById("sample-chips-container");
  try {
    const res = await fetch("/api/sample-images");
    if (!res.ok) throw new Error("Could not load samples");
    const samples = await res.json();

    container.innerHTML = "";
    samples.forEach(s => {
      const chip = document.createElement("button");
      chip.className = "sample-chip";
      chip.textContent = `${s.class_name.replace('_', ' ').split(' ')[0]}: ${s.filename}`;
      chip.addEventListener("click", () => {
        document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");

        currentSamplePath = s.path;
        currentFile = null;

        document.getElementById("preview-filename").textContent = `${s.filename} (${s.class_name})`;
        document.getElementById("preview-dimensions").textContent = "Test Cohort Sample";
        document.getElementById("drop-zone-idle").style.display = "none";
        document.getElementById("drop-zone-preview").style.display = "flex";
        document.getElementById("preview-image").src = "/results/app_runs/sample_placeholder.png";
        document.getElementById("btn-analyze").disabled = false;
      });
      container.appendChild(chip);
    });
  } catch (e) {
    container.innerHTML = "<span style='font-size:0.75rem; color:#888;'>Upload file above</span>";
  }
}

// Analyze Handler
function setupAnalyze() {
  const btn = document.getElementById("btn-analyze");
  btn.addEventListener("click", async () => {
    if (!currentFile && !currentSamplePath) return;

    const loading = document.getElementById("loading-container");
    const results = document.getElementById("results-wrapper");

    loading.style.display = "block";
    results.style.display = "none";
    btn.disabled = true;

    try {
      let res;
      if (currentFile) {
        const fd = new FormData();
        fd.append("file", currentFile);
        res = await fetch("/api/analyze", { method: "POST", body: fd });
      } else {
        res = await fetch("/api/analyze-sample", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sample_path: currentSamplePath })
        });
      }

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Analysis request failed");
      }

      const data = await res.json();
      renderTabularResults(data);
    } catch (e) {
      alert(`Analysis error: ${e.message}`);
    } finally {
      loading.style.display = "none";
      btn.disabled = false;
    }
  });
}

// Render Results in Tabular Form
function renderTabularResults(data) {
  const resultsWrapper = document.getElementById("results-wrapper");
  resultsWrapper.style.display = "block";

  if (data.preprocessing && data.preprocessing.original) {
    document.getElementById("preview-image").src = data.preprocessing.original;
  }

  // 1. Agreement Banner
  const agreement = data.agreement;
  const agreeBox = document.getElementById("agreement-box");
  const agreeTitle = document.getElementById("agreement-status-title");
  const agreeText = document.getElementById("agreement-summary-text");
  const agreeIcon = document.getElementById("agreement-icon");

  agreeTitle.textContent = `Model Consensus: ${agreement.status}`;
  agreeText.textContent = agreement.summary;

  if (agreement.unanimous) {
    agreeBox.className = "agreement-box agreement-success";
    agreeIcon.setAttribute("data-lucide", "check-circle");
  } else if (agreement.level === "medium") {
    agreeBox.className = "agreement-box agreement-warning";
    agreeIcon.setAttribute("data-lucide", "alert-triangle");
  } else {
    agreeBox.className = "agreement-box agreement-danger";
    agreeIcon.setAttribute("data-lucide", "x-circle");
  }

  // 2. Predictions & Probability Table
  const tableBody = document.getElementById("results-table-body");
  tableBody.innerHTML = "";

  const modelKeys = ["baseline", "he", "clahe"];
  modelKeys.forEach(k => {
    const m = data.models[k];
    const probs = m.probabilities;
    const badgeClass = CLASS_BADGES[m.predicted_class] || "badge-default";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${m.name}</strong></td>
      <td><code>${m.preprocessing}</code></td>
      <td><span class="badge ${badgeClass}">${m.predicted_class}</span></td>
      <td><strong>${m.confidence_pct}%</strong></td>
      <td>${(probs["Mild Demented"] * 100).toFixed(1)}%</td>
      <td>${(probs["Moderate Demented"] * 100).toFixed(1)}%</td>
      <td>${(probs["Non Demented"] * 100).toFixed(1)}%</td>
      <td>${(probs["Very Mild Demented"] * 100).toFixed(1)}%</td>
    `;
    tableBody.appendChild(tr);
  });

  // 3. Visual Saliency / Grad-CAM Table
  const visualTableBody = document.getElementById("visual-table-body");
  visualTableBody.innerHTML = "";

  modelKeys.forEach(k => {
    const m = data.models[k];
    const viz = m.visualizations;
    const badgeClass = CLASS_BADGES[m.predicted_class] || "badge-default";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${m.name}</strong><br/><small style="color:#64748b;">${m.preprocessing}</small></td>
      <td><img src="${viz.original_b64}" class="table-img" alt="Original"></td>
      <td><img src="${viz.preprocessed_b64}" class="table-img" alt="Preprocessed"></td>
      <td><img src="${viz.heatmap_b64}" class="table-img" alt="Heatmap"></td>
      <td><img src="${viz.overlay_b64}" class="table-img" alt="Overlay"></td>
      <td>
        <span class="badge ${badgeClass}">${m.predicted_class}</span>
        <div style="font-size: 0.8rem; margin-top: 0.3rem; font-family: monospace;">${m.confidence_pct}%</div>
      </td>
    `;
    visualTableBody.appendChild(tr);
  });

  if (window.lucide) {
    lucide.createIcons();
  }

  resultsWrapper.scrollIntoView({ behavior: "smooth", block: "start" });
}
