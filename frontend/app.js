/**
 * BrainScope-XAI — Clinical Decision Support & XAI Dashboard
 * Client-Side Application Controller
 */

// Global State
let currentFile = null;
let currentSamplePath = null;
let currentAnalysisResult = null;
let activeGradcamMode = "overlay"; // 'overlay' | 'heatmap' | 'preprocessed' | 'quad'

const CLASS_COLORS = {
  "Mild Demented": "badge-mild",
  "Moderate Demented": "badge-mod",
  "Non Demented": "badge-non",
  "Very Mild Demented": "badge-verymild"
};

const BREADCRUMB_MAP = {
  "tab-analysis": "Live MRI Analysis & Consensus",
  "tab-gradcam": "Grad-CAM Saliency Tri-View Studio",
  "tab-benchmarks": "Phase 1 Quantitative Benchmarks",
  "tab-roadmap": "Literature Review & Phase 2 Roadmap"
};

// Lifecycle Ingestion
document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    lucide.createIcons();
  }

  setupNavigation();
  setupDropZone();
  setupAnalyzeAction();
  setupGradcamModeToggle();
  fetchSampleCohort();
  checkSystemHealth();
});

// 1. Navigation & Tab Switching
function setupNavigation() {
  const navItems = document.querySelectorAll(".sidebar-nav .nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const targetTab = item.getAttribute("data-tab");
      switchTab(targetTab);
    });
  });

  // Jump from analysis results straight to Grad-CAM
  const btnJump = document.getElementById("btn-jump-gradcam");
  if (btnJump) {
    btnJump.addEventListener("click", () => switchTab("tab-gradcam"));
  }

  // Mobile sidebar toggle
  const btnToggle = document.getElementById("btn-toggle-sidebar");
  const sidebar = document.getElementById("sidebar");
  if (btnToggle && sidebar) {
    btnToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }
}

function switchTab(tabId) {
  // Update sidebar active state
  document.querySelectorAll(".sidebar-nav .nav-item").forEach(el => {
    if (el.getAttribute("data-tab") === tabId) {
      el.classList.add("active");
    } else {
      el.classList.remove("active");
    }
  });

  // Update tab panes
  document.querySelectorAll(".tab-pane").forEach(pane => {
    if (pane.id === tabId) {
      pane.classList.add("active");
    } else {
      pane.classList.remove("active");
    }
  });

  // Update breadcrumb
  const breadcrumb = document.getElementById("breadcrumb-title");
  if (breadcrumb && BREADCRUMB_MAP[tabId]) {
    breadcrumb.textContent = BREADCRUMB_MAP[tabId];
  }

  // Close mobile sidebar if open
  const sidebar = document.getElementById("sidebar");
  if (sidebar && sidebar.classList.contains("open")) {
    sidebar.classList.remove("open");
  }

  if (window.lucide) {
    lucide.createIcons();
  }
}

// 2. System Health Check
async function checkSystemHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      const topStatus = document.getElementById("top-status-text");
      const sbDev = document.getElementById("sb-device-status");
      const devName = data.device.toUpperCase();

      if (topStatus) {
        topStatus.textContent = `${devName} • 3 Checkpoints Mounted`;
      }
      if (sbDev) {
        sbDev.textContent = `${devName} Active`;
      }
    }
  } catch (err) {
    console.warn("[System Health] Offline or uninitialized:", err);
  }
}

// 3. Drag & Drop and Local Ingestion
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
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  btnClear.addEventListener("click", (e) => {
    e.stopPropagation();
    clearIngestedScan();
  });
}

function handleSelectedFile(file) {
  const validTypes = ["image/jpeg", "image/png", "image/jpg"];
  if (!validTypes.includes(file.type)) {
    alert("Invalid format: Please provide a valid axial brain MRI scan in PNG or JPEG format.");
    return;
  }

  currentFile = file;
  currentSamplePath = null;

  // Deactivate sample chips
  document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));

  const reader = new FileReader();
  reader.onload = (e) => {
    const imgData = e.target.result;
    displayScanPreview(imgData, file.name, `${(file.size / 1024).toFixed(1)} KB`);
  };
  reader.readAsDataURL(file);
}

function displayScanPreview(src, filename, metaStr) {
  const previewImg = document.getElementById("preview-image");
  previewImg.src = src;

  document.getElementById("preview-filename").textContent = filename;
  document.getElementById("preview-dimensions").textContent = `Axial Slice • ${metaStr}`;

  document.getElementById("drop-zone-idle").style.display = "none";
  document.getElementById("drop-zone-preview").style.display = "flex";
  document.getElementById("btn-analyze").disabled = false;
}

function clearIngestedScan() {
  currentFile = null;
  currentSamplePath = null;
  document.getElementById("file-input").value = "";
  document.getElementById("preview-image").src = "";
  document.getElementById("drop-zone-idle").style.display = "flex";
  document.getElementById("drop-zone-preview").style.display = "none";
  document.getElementById("btn-analyze").disabled = true;
  document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
}

// 4. Test Cohort Sample Picker
async function fetchSampleCohort() {
  const container = document.getElementById("sample-chips-container");
  try {
    const res = await fetch("/api/sample-images");
    if (!res.ok) throw new Error("Could not retrieve sample cohort.");
    const samples = await res.json();

    container.innerHTML = "";
    samples.forEach(s => {
      const chip = document.createElement("button");
      chip.className = "sample-chip";
      
      let shortClass = s.class_name.replace(" Demented", "");
      if (shortClass === "Very Mild") shortClass = "V. Mild";
      chip.innerHTML = `<i data-lucide="scan"></i><span>${shortClass} (${s.filename.slice(0, 10)})</span>`;

      chip.addEventListener("click", () => {
        document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");

        currentSamplePath = s.path;
        currentFile = null;

        const thumbSrc = s.thumbnail_b64 || "";
        displayScanPreview(thumbSrc, `${s.filename} [${s.class_name}]`, "Standard Test Cohort");
      });

      container.appendChild(chip);
    });

    if (window.lucide) {
      lucide.createIcons();
    }
  } catch (err) {
    container.innerHTML = `<span class="sample-chip-loading">Upload a custom MRI above</span>`;
  }
}

// 5. Analysis Trigger & Real-Time Engine Execution
function setupAnalyzeAction() {
  const btn = document.getElementById("btn-analyze");
  btn.addEventListener("click", async () => {
    if (!currentFile && !currentSamplePath) return;

    const processing = document.getElementById("processing-strip");
    const results = document.getElementById("results-container");
    const statusText = document.getElementById("processing-status-detail");

    processing.style.display = "flex";
    results.style.display = "none";
    btn.disabled = true;

    // Simulated progress indicators
    const steps = [
      "Ingesting & standardizing 224x224 axial MRI slice...",
      "Synthesizing Raw, HE, and CLAHE preprocessed modalities...",
      "Executing feedforward inference on 3 ResNet-18 architectures...",
      "Backpropagating gradients to layer4[-1] for Grad-CAM heatmaps..."
    ];
    let stepIdx = 0;
    const interval = setInterval(() => {
      if (stepIdx < steps.length) {
        statusText.textContent = steps[stepIdx];
        stepIdx++;
      }
    }, 400);

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

      clearInterval(interval);

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Analysis pipeline execution failed.");
      }

      const data = await res.json();
      currentAnalysisResult = data;
      renderAnalysisOutput(data);
    } catch (e) {
      clearInterval(interval);
      alert(`Pipeline execution error: ${e.message}`);
    } finally {
      processing.style.display = "none";
      btn.disabled = false;
    }
  });
}

// 6. Render Full Output (Consensus, Probabilities, Grad-CAM)
function renderAnalysisOutput(data) {
  const resultsContainer = document.getElementById("results-container");
  resultsContainer.style.display = "block";

  // A. Consensus Banner
  const agreement = data.agreement;
  const banner = document.getElementById("consensus-banner");
  const badge = document.getElementById("consensus-badge");
  const title = document.getElementById("consensus-title");
  const summary = document.getElementById("consensus-summary");
  const icon = document.getElementById("consensus-icon");

  title.textContent = agreement.status;
  summary.textContent = agreement.summary;

  if (agreement.unanimous) {
    banner.className = "consensus-banner consensus-success";
    badge.textContent = "UNANIMOUS CONSENSUS (3/3)";
    badge.style.color = "#047857";
    icon.setAttribute("data-lucide", "check-circle-2");
  } else if (agreement.level === "medium") {
    banner.className = "consensus-banner consensus-warning";
    badge.textContent = "MAJORITY AGREEMENT (2/3)";
    badge.style.color = "#B45309";
    icon.setAttribute("data-lucide", "alert-triangle");
  } else {
    banner.className = "consensus-banner consensus-danger";
    badge.textContent = "DIVERGENT CLASSIFICATION";
    badge.style.color = "#B91C1C";
    icon.setAttribute("data-lucide", "alert-octagon");
  }

  // Setup Patient PDF Download button
  const btnDownloadReport = document.getElementById("btn-download-run-report");
  if (btnDownloadReport) {
    btnDownloadReport.onclick = () => {
      window.open(`/api/report/${data.run_id}`, "_blank");
    };
  }

  // B. Tri-Model Comparative Cards
  const grid = document.getElementById("models-tri-grid");
  grid.innerHTML = "";

  const modelKeys = ["baseline", "he", "clahe"];
  const classOrder = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"];

  modelKeys.forEach(k => {
    const m = data.models[k];
    const isWinner = k === "clahe";

    const card = document.createElement("div");
    card.className = `model-stat-card ${isWinner ? 'winner' : ''}`;

    let badgeClass = CLASS_COLORS[m.predicted_class] || "badge-outline";

    let probListHtml = "";
    classOrder.forEach(c => {
      const pVal = m.probabilities[c] || 0;
      const pPct = (pVal * 100).toFixed(1);
      const isTop = c === m.predicted_class;
      probListHtml += `
        <div class="prob-row">
          <div class="prob-meta">
            <span class="prob-class-name ${isTop ? 'font-bold text-primary' : ''}">${c}</span>
            <span class="prob-class-val">${pPct}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill ${isTop ? (isWinner ? 'winner-fill' : '') : ''}" style="width: ${pPct}%;"></div>
          </div>
        </div>
      `;
    });

    card.innerHTML = `
      <div class="m-header">
        <h4>${m.name}</h4>
        <span class="badge ${isWinner ? 'badge-success' : 'badge-outline'}">${m.preprocessing}</span>
      </div>
      <div class="m-body">
        <div class="m-pred-highlight">
          <div>
            <div style="font-size:0.68rem; color:var(--text-muted); font-weight:700; text-transform:uppercase;">Model Diagnosis</div>
            <span class="badge ${badgeClass}" style="margin-top:0.25rem;">${m.predicted_class}</span>
          </div>
          <div class="m-pred-conf">${m.confidence_pct}%</div>
        </div>
        <div class="prob-list">
          ${probListHtml}
        </div>
      </div>
      <div class="m-footer-stat">
        <span>Benchmark Acc: <strong>${m.benchmark_acc}</strong></span>
        <span>Macro-F1: <strong>${m.benchmark_macro_f1}</strong></span>
      </div>
    `;

    grid.appendChild(card);
  });

  // C. Tabular Comparison Matrix
  const tbody = document.getElementById("matrix-table-body");
  tbody.innerHTML = "";

  modelKeys.forEach(k => {
    const m = data.models[k];
    const probs = m.probabilities;
    const badgeClass = CLASS_COLORS[m.predicted_class] || "badge-outline";

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
    tbody.appendChild(tr);
  });

  // D. Populate Grad-CAM Tri-View Tab
  populateGradcamStudio(data);

  if (window.lucide) {
    lucide.createIcons();
  }

  // Smooth scroll down to results
  resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
}

// 7. Grad-CAM Tri-View Studio Controller
function setupGradcamModeToggle() {
  const toggleBtns = document.querySelectorAll("#gradcam-view-toggle .toggle-btn");
  toggleBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      toggleBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeGradcamMode = btn.getAttribute("data-mode");
      if (currentAnalysisResult) {
        updateGradcamImages(currentAnalysisResult);
      }
    });
  });
}

function populateGradcamStudio(data) {
  const emptyState = document.getElementById("gradcam-empty-state");
  const studioGrid = document.getElementById("gradcam-studio-grid");

  if (emptyState) emptyState.style.display = "none";
  if (studioGrid) studioGrid.style.display = "grid";

  // Fill in diagnosis badges
  const models = data.models;

  // Baseline
  document.getElementById("xai-class-baseline").textContent = models.baseline.predicted_class;
  document.getElementById("xai-conf-baseline").textContent = `${models.baseline.confidence_pct}%`;

  // HE
  document.getElementById("xai-class-he").textContent = models.he.predicted_class;
  document.getElementById("xai-conf-he").textContent = `${models.he.confidence_pct}%`;

  // CLAHE
  document.getElementById("xai-class-clahe").textContent = models.clahe.predicted_class;
  document.getElementById("xai-conf-clahe").textContent = `${models.clahe.confidence_pct}%`;

  updateGradcamImages(data);
}

function updateGradcamImages(data) {
  const models = data.models;
  const tagBaseline = document.getElementById("xai-tag-baseline");
  const tagHE = document.getElementById("xai-tag-he");
  const tagCLAHE = document.getElementById("xai-tag-clahe");

  let imgBase, imgHE, imgCLAHE, tagLabel;

  if (activeGradcamMode === "heatmap") {
    tagLabel = "Grad-CAM Heatmap (Jet)";
    imgBase = models.baseline.visualizations.heatmap_b64;
    imgHE = models.he.visualizations.heatmap_b64;
    imgCLAHE = models.clahe.visualizations.heatmap_b64;
  } else if (activeGradcamMode === "preprocessed") {
    tagLabel = "Filtered Slice";
    imgBase = models.baseline.visualizations.preprocessed_b64;
    imgHE = models.he.visualizations.preprocessed_b64;
    imgCLAHE = models.clahe.visualizations.preprocessed_b64;
  } else {
    // Default 'overlay' or 'quad'
    tagLabel = "Overlay Blend (60/40)";
    imgBase = models.baseline.visualizations.overlay_b64;
    imgHE = models.he.visualizations.overlay_b64;
    imgCLAHE = models.clahe.visualizations.overlay_b64;
  }

  document.getElementById("xai-img-baseline").src = imgBase;
  document.getElementById("xai-img-he").src = imgHE;
  document.getElementById("xai-img-clahe").src = imgCLAHE;

  if (tagBaseline) tagBaseline.textContent = tagLabel;
  if (tagHE) tagHE.textContent = tagLabel;
  if (tagCLAHE) tagCLAHE.textContent = tagLabel;
}

// 8. Modal Zoom & Inspection Dialog
function openZoomModal(modelKey) {
  if (!currentAnalysisResult) return;
  const m = currentAnalysisResult.models[modelKey];
  const src = m.visualizations.overlay_b64;
  const title = `${m.name} — Grad-CAM Layer-4 Anatomical Inspection`;
  const caption = `Predicted Stage: ${m.predicted_class} (${m.confidence_pct}% Confidence) | Preprocessing: ${m.preprocessing}`;

  openImageModal(src, title, caption);
}

function openImageModal(imgSrc, title, caption = "") {
  const modal = document.getElementById("image-modal");
  const display = document.getElementById("modal-image-display");
  const titleEl = document.getElementById("modal-image-title");
  const capEl = document.getElementById("modal-image-caption");

  display.src = imgSrc;
  titleEl.textContent = title;
  capEl.textContent = caption;

  modal.style.display = "flex";
  if (window.lucide) {
    lucide.createIcons();
  }
}

function closeImageModal() {
  const modal = document.getElementById("image-modal");
  if (modal) {
    modal.style.display = "none";
  }
}
