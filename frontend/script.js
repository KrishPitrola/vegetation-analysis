/**
 * VegeVision — script.js
 * Handles file upload, drag-drop, mode selection, API calls and result rendering.
 */

'use strict';

// ─── Config ───────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone       = document.getElementById('dropZone');
const imageInput     = document.getElementById('imageInput');
const previewWrap    = document.getElementById('previewWrap');
const previewImg     = document.getElementById('previewImg');
const previewName    = document.getElementById('previewName');
const removeBtn      = document.getElementById('removeBtn');
const dropZoneInner  = dropZone.querySelector('.drop-zone-inner');

const modeCards      = document.querySelectorAll('.mode-card');
const analyzeBtn     = document.getElementById('analyzeBtn');
const btnText        = analyzeBtn.querySelector('.btn-text');
const btnSpinner     = analyzeBtn.querySelector('.btn-spinner');
const submitHint     = document.getElementById('submitHint');

const resultsPanel      = document.getElementById('resultsPanel');
const newAnalysisBtn    = document.getElementById('newAnalysisBtn');
const resultsIcon       = document.getElementById('resultsIcon');
const resultError       = document.getElementById('resultError');
const errorMsg          = document.getElementById('errorMsg');
const resultSatellite   = document.getElementById('resultSatellite');
const resultTrees       = document.getElementById('resultTrees');
const resultLeaf        = document.getElementById('resultLeaf');

const statClass       = document.getElementById('statClass');
const statConf        = document.getElementById('statConf');
const confidenceBar   = document.getElementById('confidenceBar');
const vegetationMsg   = document.getElementById('vegetationMsg');

const statTreeCount   = document.getElementById('statTreeCount');
const statVegLevel    = document.getElementById('statVegLevel');
const statCoverage    = document.getElementById('statCoverage');
const coverageBar     = document.getElementById('coverageBar');
const statSatCoverage      = document.getElementById('statSatCoverage');
const statSatCoverageRange = document.getElementById('statSatCoverageRange');
const satellitePreviewImg  = document.getElementById('satellitePreviewImg');
const annotatedImg    = document.getElementById('annotatedImg');
const downloadBtn     = document.getElementById('downloadBtn');


const statLeafClass     = document.getElementById('statLeafClass');
const statLeafConf      = document.getElementById('statLeafConf');
const leafConfidenceBar = document.getElementById('leafConfidenceBar');
const leafPreviewImg    = document.getElementById('leafPreviewImg');

// ─── State ────────────────────────────────────────────────────────────────────
let uploadedFile  = null;
let selectedMode  = null;
let previewUrl    = null;   // object URL of the currently uploaded file

// ─── Utilities ────────────────────────────────────────────────────────────────
function toggleClass(el, cls, force) {
  if (force === undefined) el.classList.toggle(cls);
  else if (force) el.classList.add(cls);
  else el.classList.remove(cls);
}

function showEl(el)  { toggleClass(el, 'hidden', false); }
function hideEl(el)  { toggleClass(el, 'hidden', true);  }



// ─── File handling ────────────────────────────────────────────────────────────
function setFile(file) {
  if (!file) return;

  const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp',
                   'image/tiff', 'image/tif', ''];
  // We allow empty MIME (tif files sometimes have no MIME on Windows)
  const ext = file.name.split('.').pop().toLowerCase();
  const allowedExts = ['jpg','jpeg','png','webp','bmp','tif','tiff'];
  if (!allowedExts.includes(ext)) {
    showError(`Unsupported file type ".${ext}". Please upload JPG, PNG, TIF, BMP or WebP.`);
    return;
  }

  uploadedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);  // free previous blob
  previewUrl = URL.createObjectURL(file);
  previewImg.src = previewUrl;
  previewName.textContent = file.name;
  hideEl(dropZoneInner);
  showEl(previewWrap);
  updateAnalyzeBtn();
}

function clearFile() {
  uploadedFile = null;
  imageInput.value = '';
  previewImg.src = '';
  previewName.textContent = '';
  hideEl(previewWrap);
  showEl(dropZoneInner);
  updateAnalyzeBtn();
}



imageInput.addEventListener('change', (e) => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

removeBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFile();
});

// Drag & Drop
['dragenter', 'dragover'].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
});
['dragleave', 'drop'].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
  });
});
dropZone.addEventListener('drop', (e) => {
  const file = e.dataTransfer?.files[0];
  if (file) setFile(file);
});
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    imageInput.click();
  }
});
dropZone.addEventListener('click', (e) => {
  if (!previewWrap.classList.contains('hidden')) return; // don't re-trigger when preview shown
  if (e.target === removeBtn) return;
  imageInput.click();
});

// ─── Mode selection ───────────────────────────────────────────────────────────
modeCards.forEach(card => {
  card.addEventListener('click', () => {
    card.querySelector('input[type=radio]').checked = true;
    modeCards.forEach(c => {
      c.classList.remove('selected');
      c.setAttribute('aria-checked', 'false');
    });
    card.classList.add('selected');
    card.setAttribute('aria-checked', 'true');
    selectedMode = card.querySelector('input[type=radio]').value;
    updateAnalyzeBtn();
  });
  card.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.click(); }
  });
});

// ─── Analyze button state ─────────────────────────────────────────────────────
function updateAnalyzeBtn() {
  const ready = uploadedFile && selectedMode;
  analyzeBtn.disabled = !ready;
  analyzeBtn.setAttribute('aria-disabled', String(!ready));
  if (!uploadedFile && !selectedMode) submitHint.textContent = 'Upload an image and select a mode to continue.';
  else if (!uploadedFile) submitHint.textContent = 'Please upload an image.';
  else if (!selectedMode) submitHint.textContent = 'Please select an analysis mode.';
  else submitHint.textContent = 'Ready! Click Analyze Image to run inference.';
}

// ─── Loading state ────────────────────────────────────────────────────────────
function setLoading(on) {
  analyzeBtn.disabled = on;
  toggleClass(btnText, 'hidden', on);
  toggleClass(btnSpinner, 'hidden', !on);
}

// ─── Reset results ────────────────────────────────────────────────────────────
function resetResults() {
  hideEl(resultError);
  hideEl(resultSatellite);
  hideEl(resultTrees);
  hideEl(resultLeaf);
}

// ─── Show error ───────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  showEl(resultError);
  hideEl(resultSatellite);
  hideEl(resultTrees);
  hideEl(resultLeaf);
  resultsIcon.textContent = '⚠️';
  showEl(resultsPanel);
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Render satellite result ──────────────────────────────────────────────────
function renderSatelliteResult(data) {
  resultsIcon.textContent = '🛰️';
  statClass.textContent = data.predicted_class;
  statConf.textContent  = `${data.confidence.toFixed(1)} %`;
  vegetationMsg.textContent = data.vegetation_message;

  // Show uploaded image preview
  if (previewUrl) satellitePreviewImg.src = previewUrl;

  // Estimated coverage
  if (data.estimated_coverage !== undefined) {
    statSatCoverage.textContent = `${data.estimated_coverage.toFixed(1)} %`;
    statSatCoverageRange.textContent = data.estimated_coverage_range
      ? `Range: ${data.estimated_coverage_range}` : '';
  } else {
    statSatCoverage.textContent = 'N/A';
    statSatCoverageRange.textContent = '';
  }

  // Animate confidence bar
  setTimeout(() => {
    confidenceBar.style.width = `${Math.min(data.confidence, 100)}%`;
  }, 50);

  hideEl(resultError);
  hideEl(resultTrees);
  hideEl(resultLeaf);
  showEl(resultSatellite);
  showEl(resultsPanel);
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Render tree result ────────────────────────────────────────────────────── 
function renderTreeResult(data) {
  resultsIcon.textContent = '🌳';
  statTreeCount.textContent = data.tree_count;
  statVegLevel.textContent  = data.vegetation_level;

  // Coverage percentage
  const pct = (typeof data.coverage_percentage === 'number') ? data.coverage_percentage : 0;
  statCoverage.textContent = `${pct.toFixed(2)} %`;
  setTimeout(() => {
    if (coverageBar) coverageBar.style.width = `${Math.min(pct, 100)}%`;
  }, 50);

  // Annotated image
  const imgSrc = `data:image/png;base64,${data.annotated_image}`;
  annotatedImg.src = imgSrc;
  downloadBtn.href = imgSrc;

  hideEl(resultError);
  hideEl(resultSatellite);
  hideEl(resultLeaf);
  showEl(resultTrees);
  showEl(resultsPanel);
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Render leaf result ────────────────────────────────────────────────────── 
function renderLeafResult(data) {
  resultsIcon.textContent = '🌿';
  statLeafClass.textContent = data.species;
  // Convert 0-1 confidence to percentage
  const confPercent = data.confidence * 100;
  statLeafConf.textContent  = `${confPercent.toFixed(2)} %`;

  // Show uploaded image preview
  if (previewUrl) leafPreviewImg.src = previewUrl;

  // Animate confidence bar
  setTimeout(() => {
    leafConfidenceBar.style.width = `${Math.min(confPercent, 100)}%`;
  }, 50);

  hideEl(resultError);
  hideEl(resultSatellite);
  hideEl(resultTrees);
  showEl(resultLeaf);
  showEl(resultsPanel);
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Main analyze handler ─────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!uploadedFile || !selectedMode) return;

  resetResults();
  setLoading(true);

  try {
    let mode = selectedMode;

    const fd = new FormData();
    fd.append('file', uploadedFile);

    let endpoint, data;
    if (mode === 'satellite') {
      endpoint = `${API_BASE}/predict-satellite`;
    } else if (mode === 'leaf') {
      endpoint = `${API_BASE}/predict-leaf`;
    } else {
      endpoint = `${API_BASE}/predict-trees`;
    }

    const res = await fetch(endpoint, { method: 'POST', body: fd });
    const body = await res.json();

    if (!res.ok) {
      throw new Error(body.detail || `Server error ${res.status}`);
    }

    data = body;

    if (mode === 'satellite') {
      renderSatelliteResult(data);
    } else if (mode === 'leaf') {
      renderLeafResult(data);
    } else {
      renderTreeResult(data);
    }

  } catch (err) {
    console.error(err);
    showError(err.message || 'An unexpected error occurred. Is the backend running?');
  } finally {
    setLoading(false);
    updateAnalyzeBtn();
  }
});

// ─── New analysis ─────────────────────────────────────────────────────────────
newAnalysisBtn.addEventListener('click', () => {
  hideEl(resultsPanel);
  resetResults();
  clearFile();
  confidenceBar.style.width = '0%';
  modeCards.forEach(c => {
    c.classList.remove('selected');
    c.setAttribute('aria-checked', 'false');
  });
  selectedMode = null;
  updateAnalyzeBtn();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
