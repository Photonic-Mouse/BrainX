// BrainX Web Application Client Logic

let currentFile = null;
let currentSamplePath = null;
let activeRunId = null;
let probabilityChart = null;

const CLASS_COLORS = {
  "Mild Demented": "#38bdf8",
  "Moderate Demented": "#ec4899",
  "Non Demented": "#10b981",
  "Very Mild Demented": "#a855f7"
};

// Initialize app
document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    lucide.createIcons();
  }

  setupNavigation();
  setupDropZone();
  setupAnalyzeAction();
  fetchSampleImages();
  fetchSystemStatus();
});

// Tab Navigation
function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-btn");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const targetId = btn.getAttribute("data-tab");
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      if (targetId === "tab-history") {
        fetchHistory();
      }

      if (window.lucide) lucide.createIcons();
    });
  });
}

// Fetch System Status
async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      const label = document.getElementById("system-device-label");
      if (label) {
        label.textContent = `${data.device.toUpperCase()} Active • 3 Models`;
      }
    }
  } catch (e) {
    console.warn("Health check error:", e);
  }
}

// Setup Drag & Drop Zone
function setupDropZone() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const btnClear = document.getElementById("btn-clear-file");

  ["dragenter", "dragover"].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt.files && dt.files.length > 0) {
      handleSelectedFile(dt.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  btnClear.addEventListener("click", (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });
}

function handleSelectedFile(file) {
  const validTypes = ["image/jpeg", "image/png", "image/jpg"];
  if (!validTypes.includes(file.type)) {
    alert("Please select a valid PNG or JPEG image.");
    return;
  }

  currentFile = file;
  currentSamplePath = null;

  // Unselect any active chips
  document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById("preview-image");
    img.src = e.target.result;
    
    // Read dimensions once loaded
    const tempImg = new Image();
    tempImg.src = e.target.result;
    tempImg.onload = () => {
      document.getElementById("preview-dimensions").textContent = `${tempImg.naturalWidth} x ${tempImg.naturalHeight} px (${(file.size / 1024).toFixed(1)} KB)`;
    };

    document.getElementById("preview-filename").textContent = file.name;
    document.getElementById("drop-zone-idle").style.display = "none";
    document.getElementById("drop-zone-preview").style.display = "flex";
    document.getElementById("btn-analyze").disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearSelectedFile() {
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
async function fetchSampleImages() {
  const container = document.getElementById("sample-chips-container");
  try {
    const res = await fetch("/api/sample-images");
    if (!res.ok) throw new Error("Could not fetch samples");
    const samples = await res.json();

    container.innerHTML = "";
    samples.forEach(sample => {
      const chip = document.createElement("button");
      chip.className = "sample-chip";
      chip.innerHTML = `<span><strong>${sample.class_name.split(' ')[0]}:</strong> ${sample.filename}</span>`;
      chip.addEventListener("click", () => {
        selectSampleImage(sample, chip);
      });
      container.appendChild(chip);
    });
  } catch (e) {
    container.innerHTML = `<span style="font-size:0.75rem; color:#94a3b8;">Custom file upload available</span>`;
  }
}

function selectSampleImage(sample, chipElement) {
  document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
  chipElement.classList.add("active");

  currentSamplePath = sample.path;
  currentFile = null;

  // Show in preview
  document.getElementById("preview-filename").textContent = `${sample.filename} (${sample.class_name})`;
  document.getElementById("preview-dimensions").textContent = "Test Cohort Sample";
  document.getElementById("drop-zone-idle").style.display = "none";
  document.getElementById("drop-zone-preview").style.display = "flex";
  document.getElementById("preview-image").src = "/results/app_runs/sample_placeholder.png"; // temporary until analyzed
  document.getElementById("btn-analyze").disabled = false;
}

// Analyze Action
function setupAnalyzeAction() {
  const btn = document.getElementById("btn-analyze");
  btn.addEventListener("click", runAnalysis);

  const btnPdf = document.getElementById("btn-download-pdf");
  btnPdf.addEventListener("click", () => {
    if (activeRunId) {
      window.open(`/api/report/${activeRunId}`, "_blank");
    }
  });
}

async function runAnalysis() {
  if (!currentFile && !currentSamplePath) return;

  const loading = document.getElementById("loading-container");
  const results = document.getElementById("results-wrapper");
  const btn = document.getElementById("btn-analyze");

  loading.style.display = "block";
  results.style.display = "none";
  btn.disabled = true;

  try {
    let response;
    if (currentFile) {
      const formData = new FormData();
      formData.append("file", currentFile);
      response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });
    } else {
      response = await fetch("/api/analyze-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_path: currentSamplePath })
      });
    }

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Analysis failed");
    }

    const data = await response.json();
    renderAnalysisResults(data);
  } catch (err) {
    alert(`Error during analysis: ${err.message}`);
  } finally {
    loading.style.display = "none";
    btn.disabled = false;
  }
}

// Render Results
function renderAnalysisResults(data) {
  activeRunId = data.run_id;
  const results = document.getElementById("results-wrapper");
  results.style.display = "block";

  // Update preview image with processed raw image if sample was selected
  if (data.preprocessing && data.preprocessing.original) {
    document.getElementById("preview-image").src = data.preprocessing.original;
  }

  // 1. Agreement Banner
  const agreement = data.agreement;
  const agreeBanner = document.getElementById("agreement-banner");
  const agreeTitle = document.getElementById("agreement-status-title");
  const agreeSummary = document.getElementById("agreement-summary-text");
  const agreeIcon = document.getElementById("agreement-icon");

  agreeTitle.textContent = agreement.status;
  agreeSummary.textContent = agreement.summary;

  if (agreement.unanimous) {
    agreeBanner.style.borderColor = "rgba(16, 185, 129, 0.4)";
    agreeBanner.style.background = "rgba(16, 185, 129, 0.1)";
    agreeIcon.setAttribute("data-lucide", "check-circle");
    agreeIcon.style.color = "#34d399";
  } else if (agreement.level === "medium") {
    agreeBanner.style.borderColor = "rgba(245, 158, 11, 0.4)";
    agreeBanner.style.background = "rgba(245, 158, 11, 0.1)";
    agreeIcon.setAttribute("data-lucide", "alert-triangle");
    agreeIcon.style.color = "#fbbf24";
  } else {
    agreeBanner.style.borderColor = "rgba(239, 68, 68, 0.4)";
    agreeBanner.style.background = "rgba(239, 68, 68, 0.1)";
    agreeIcon.setAttribute("data-lucide", "x-circle");
    agreeIcon.style.color = "#f87171";
  }

  // 2. Populate 3 Model Cards
  ["baseline", "he", "clahe"].forEach(key => {
    const m = data.models[key];
    document.getElementById(`pred-class-${key}`).textContent = m.predicted_class;
    document.getElementById(`conf-pill-${key}`).textContent = `Confidence: ${m.confidence_pct}%`;

    // Probability Bars
    const container = document.getElementById(`prob-bars-${key}`);
    container.innerHTML = "";
    for (const [cname, prob] of Object.entries(m.probabilities)) {
      const pct = (prob * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-labels">
          <span>${cname}</span>
          <span>${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${pct}%; background-color: ${CLASS_COLORS[cname] || '#3b82f6'};"></div>
        </div>
      `;
      container.appendChild(row);
    }
  });

  // 3. Render Probability Comparison Chart (Chart.js)
  renderProbabilityChart(data.models, data.class_names);

  // 4. Preprocessing Comparison Images
  document.getElementById("prep-img-orig").src = data.preprocessing.original;
  document.getElementById("prep-img-he").src = data.preprocessing.he;
  document.getElementById("prep-img-clahe").src = data.preprocessing.clahe;

  // 5. Populate Grad-CAM Grid
  ["baseline", "he", "clahe"].forEach(key => {
    const m = data.models[key];
    const viz = m.visualizations;
    document.getElementById(`gc-pred-${key}`).textContent = `Predicted: ${m.predicted_class} (${m.confidence_pct}%)`;
    document.getElementById(`gc-orig-${key}`).src = viz.original_b64;
    document.getElementById(`gc-prep-${key}`).src = viz.preprocessed_b64;
    document.getElementById(`gc-heat-${key}`).src = viz.heatmap_b64;
    document.getElementById(`gc-over-${key}`).src = viz.overlay_b64;
  });

  if (window.lucide) lucide.createIcons();

  // Scroll smoothly to results
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Chart.js Probability Bar Chart
function renderProbabilityChart(models, classNames) {
  const ctx = document.getElementById("probability-chart").getContext("2d");

  const baselineData = classNames.map(c => models.baseline.probabilities[c]);
  const heData = classNames.map(c => models.he.probabilities[c]);
  const claheData = classNames.map(c => models.clahe.probabilities[c]);

  if (probabilityChart) {
    probabilityChart.destroy();
  }

  probabilityChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: classNames,
      datasets: [
        {
          label: "Baseline (Original MRI)",
          data: baselineData,
          backgroundColor: "#3b82f6",
          borderRadius: 4
        },
        {
          label: "Histogram Equalization (HE)",
          data: heData,
          backgroundColor: "#ef4444",
          borderRadius: 4
        },
        {
          label: "CLAHE ResNet18",
          data: claheData,
          backgroundColor: "#10b981",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { family: "'Plus Jakarta Sans'" } }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", font: { family: "'Plus Jakarta Sans'" } },
          grid: { color: "rgba(255, 255, 255, 0.05)" }
        },
        y: {
          min: 0,
          max: 1.0,
          ticks: {
            color: "#94a3b8",
            callback: (v) => `${(v * 100).toFixed(0)}%`
          },
          grid: { color: "rgba(255, 255, 255, 0.05)" }
        }
      }
    }
  });
}

// History Fetch & Render
async function fetchHistory() {
  const container = document.getElementById("history-container");
  const badge = document.getElementById("history-count-badge");
  try {
    const res = await fetch("/api/history");
    if (!res.ok) throw new Error("Failed to fetch history");
    const history = await res.json();

    badge.textContent = `${history.length} Analyses`;

    if (history.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <i data-lucide="inbox"></i>
          <p>No analyses conducted yet in this session.</p>
        </div>
      `;
      if (window.lucide) lucide.createIcons();
      return;
    }

    container.innerHTML = "";
    history.forEach(item => {
      const el = document.createElement("div");
      el.className = "history-item";
      el.innerHTML = `
        <div class="history-meta">
          <strong>${item.filename} — ${item.consensus_class}</strong>
          <span>${item.timestamp} • ${item.agreement}</span>
        </div>
        <button class="btn btn-sm btn-secondary" onclick="window.open('/api/report/${item.run_id}', '_blank')">
          <i data-lucide="file-text"></i> PDF Report
        </button>
      `;
      container.appendChild(el);
    });
    if (window.lucide) lucide.createIcons();
  } catch (e) {
    container.innerHTML = `<p style="color:#ef4444; font-size:0.85rem;">Failed to load history</p>`;
  }
}
