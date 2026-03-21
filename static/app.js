const fileInput = document.getElementById('fileInput');
const patientIdInput = document.getElementById('patientId');
const patientIdBadge = document.getElementById('patientIdBadge');
const newPatientBtn = document.getElementById('newPatientBtn');
const patientSearch = document.getElementById('patientSearch');
const searchBtn = document.getElementById('searchBtn');
const searchResults = document.getElementById('searchResults');
const predictForm = document.getElementById('predictForm');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const statusBox = document.getElementById('status');
const previewImage = document.getElementById('previewImage');
const previewHint = document.getElementById('previewHint');
const resultsPanel = document.getElementById('resultsPanel');
const uploadStage = document.getElementById('uploadStage');
const loadingStage = document.getElementById('loadingStage');
const analysisStage = document.getElementById('analysisStage');
const analysisImage = document.getElementById('analysisImage');
const analysisImageHint = document.getElementById('analysisImageHint');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const stepUpload = document.getElementById('stepUpload');
const stepLoading = document.getElementById('stepLoading');
const stepAnalysis = document.getElementById('stepAnalysis');
const stepHistory = document.getElementById('stepHistory');
const historyStage = document.getElementById('historyStage');

// History Stage elements
const historySearchInput = document.getElementById('historySearchInput');
const historySearchBtn = document.getElementById('historySearchBtn');
const historyPatientTrend = document.getElementById('historyPatientTrend');
const historyFullList = document.getElementById('historyFullList');
const historyRecordViewer = document.getElementById('historyRecordViewer');
const closeHistoryViewerBtn = document.getElementById('closeHistoryViewerBtn');
const historyRecordImage = document.getElementById('historyRecordImage');
const historyRecordSummary = document.getElementById('historyRecordSummary');
const historyRecordRules = document.getElementById('historyRecordRules');
const historyRecordReasoning = document.getElementById('historyRecordReasoning');

// Quick viewer in Upload stage
const quickRecordViewer = document.getElementById('quickRecordViewer');
const closeQuickRecord = document.getElementById('closeQuickRecord');
const quickRecordSummary = document.getElementById('quickRecordSummary');
const quickRecordRules = document.getElementById('quickRecordRules');

const primaryDisease = document.getElementById('primaryDisease');
const primaryConf = document.getElementById('primaryConf');
const secondaryDisease = document.getElementById('secondaryDisease');
const secondaryConf = document.getElementById('secondaryConf');
const bandBadge = document.getElementById('bandBadge');
const recommendation = document.getElementById('recommendation');
const rulesList = document.getElementById('rulesList');
const reasoningList = document.getElementById('reasoningList');
const patientTrend = document.getElementById('patientTrend');
const historyList = document.getElementById('historyList');
const downloadJsonBtn = document.getElementById('downloadJsonBtn');
const openPrintBtn = document.getElementById('openPrintBtn');
const recordViewer = document.getElementById('recordViewer');
const recordImage = document.getElementById('recordImage');
const recordSummary = document.getElementById('recordSummary');
const recordRules = document.getElementById('recordRules');
const recordReasoning = document.getElementById('recordReasoning');

let progressTimer = null;

function setCurrentPatientId(patientId) {
  patientIdInput.value = patientId;
  patientIdBadge.textContent = patientId;
}

async function fetchNewPatientId() {
  const response = await fetch('/patient/new-id');
  const data = await response.json();
  setCurrentPatientId(data.patient_id);
}

function renderSearchResults(items) {
  searchResults.innerHTML = '';
  if (!items.length) {
    const li = document.createElement('li');
    li.textContent = 'No matching patients found.';
    searchResults.appendChild(li);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement('li');
    li.className = 'search-item';
    li.innerHTML = `
      <div>
        <strong>${item.patient_id}</strong><br>
        <small>Uploads: ${item.total_uploads} | Last seen: ${item.last_seen || '-'}</small>
      </div>
    `;

    const btnWrapper = document.createElement('div');
    btnWrapper.style.display = 'flex';
    btnWrapper.style.gap = '6px';
    btnWrapper.style.flexDirection = 'column';

    const useBtn = document.createElement('button');
    useBtn.type = 'button';
    useBtn.className = 'mini-btn';
    useBtn.textContent = 'Use ID';
    useBtn.addEventListener('click', async () => {
      setCurrentPatientId(item.patient_id);
      await loadPatientHistory(item.patient_id);
      setStatus(`Selected existing patient ${item.patient_id}. You can upload another X-ray under same ID.`);
    });

    const viewBtn = document.createElement('button');
    viewBtn.type = 'button';
    viewBtn.className = 'mini-btn';
    viewBtn.textContent = 'View';
    viewBtn.addEventListener('click', async () => {
      try {
        const res = await fetch(`/patient/${encodeURIComponent(item.patient_id)}/history?limit=1`);
        const data = await res.json();
        if (data.history && data.history.length > 0) {
           const rec = data.history[0];
           quickRecordViewer.hidden = false;
           quickRecordSummary.textContent = `${rec.created_at} | ${rec.predicted_disease} (${(rec.prediction_confidence * 100).toFixed(1)}%)`;
           quickRecordRules.innerHTML = '';
           
           const detailRes = await fetch(`/prediction/${rec.id}`);
           const detailData = await detailRes.json();
           (detailData.rules_fired || []).forEach(r => {
             const rl = document.createElement('li');
             rl.textContent = r;
             quickRecordRules.appendChild(rl);
           });
        } else {
           setStatus('No records found.', true);
        }
      } catch(e) {}
    });

    btnWrapper.appendChild(useBtn);
    btnWrapper.appendChild(viewBtn);
    li.appendChild(btnWrapper);
    searchResults.appendChild(li);
  });
}

closeQuickRecord?.addEventListener('click', () => {
  quickRecordViewer.hidden = true;
});

async function runPatientSearch() {
  const q = patientSearch.value.trim();
  const response = await fetch(`/patients/search?q=${encodeURIComponent(q)}&limit=12`);
  const data = await response.json();
  renderSearchResults(data.results || []);
}

async function loadRecord(recordId, useHistoryViewer = false) {
  const response = await fetch(`/prediction/${recordId}`);
  if (!response.ok) {
    throw new Error('Failed to load record details.');
  }
  const record = await response.json();
  if (useHistoryViewer) {
    renderHistoryRecordView(record);
  } else {
    renderRecordView(record);
  }
}

function renderHistoryRecordView(record) {
  historyRecordViewer.hidden = false;
  if (record.image_url) {
    historyRecordImage.src = record.image_url;
  } else {
    historyRecordImage.removeAttribute('src');
  }

  historyRecordSummary.textContent = `Record #${record.id} | ${record.created_at} | ${record.predicted_disease} (${(record.prediction_confidence * 100).toFixed(2)}%)`;

  historyRecordRules.innerHTML = '';
  (record.rules_fired || []).forEach((rule) => {
    const li = document.createElement('li');
    li.textContent = rule;
    historyRecordRules.appendChild(li);
  });

  historyRecordReasoning.innerHTML = '';
  (record.if_then_reasoning || []).forEach((line) => {
    const li = document.createElement('li');
    li.textContent = line;
    historyRecordReasoning.appendChild(li);
  });
}

function renderRecordView(record) {
  recordViewer.hidden = false;
  if (record.image_url) {
    recordImage.src = record.image_url;
  } else {
    recordImage.removeAttribute('src');
  }

  recordSummary.textContent = `Record #${record.id} | ${record.created_at} | ${record.predicted_disease} (${(record.prediction_confidence * 100).toFixed(2)}%) | secondary: ${record.secondary_disease} (${(record.secondary_confidence * 100).toFixed(2)}%)`;

  recordRules.innerHTML = '';
  (record.rules_fired || []).forEach((rule) => {
    const li = document.createElement('li');
    li.textContent = rule;
    recordRules.appendChild(li);
  });

  recordReasoning.innerHTML = '';
  (record.if_then_reasoning || []).forEach((line) => {
    const li = document.createElement('li');
    li.textContent = line;
    recordReasoning.appendChild(li);
  });
}

async function loadRecord(recordId) {
  const response = await fetch(`/prediction/${recordId}`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || 'Unable to load record details.');
  }
  renderRecordView(data);
}

async function loadPatientHistory(patientId) {
  const response = await fetch(`/patient/${encodeURIComponent(patientId)}/history?limit=8`);
  const data = await response.json();

  const trend = data.patient_summary?.trend || 'no_history';
  patientTrend.textContent = `Patient: ${patientId} | Trend: ${trend} | Total records: ${data.patient_summary?.total_predictions ?? 0}`;

  historyList.innerHTML = '';
  (data.history || []).forEach((h) => {
    const li = document.createElement('li');
    const top = document.createElement('div');
    top.className = 'history-row-top';
    top.innerHTML = `<div><strong>${h.predicted_disease}</strong> | ${(h.prediction_confidence * 100).toFixed(2)}% | ${h.created_at}</div>`;

    const actionWrap = document.createElement('div');
    actionWrap.className = 'history-actions';

    const viewImg = document.createElement('a');
    viewImg.className = 'mini-btn';
    viewImg.textContent = 'View X-ray';
    viewImg.href = h.image_url || '#';
    viewImg.target = '_blank';
    viewImg.rel = 'noopener';

    const viewReport = document.createElement('button');
    viewReport.type = 'button';
    viewReport.className = 'mini-btn';
    viewReport.textContent = 'View Report';
    viewReport.addEventListener('click', async () => {
      try {
        await loadRecord(h.id);
      } catch (err) {
        setStatus(err.message, true);
      }
    });

    actionWrap.appendChild(viewImg);
    actionWrap.appendChild(viewReport);
    top.appendChild(actionWrap);
    li.appendChild(top);

    const rec = document.createElement('div');
    rec.textContent = `Band: ${h.confidence_band} | Secondary: ${h.secondary_disease || '-'} (${((h.secondary_confidence || 0) * 100).toFixed(2)}%)`;
    li.appendChild(rec);
    historyList.appendChild(li);
  });
}

function setStep(step) {
  [stepUpload, stepLoading, stepAnalysis, stepHistory].forEach((s) => {
    if (s) {
      s.classList.remove('is-active');
      s.classList.remove('is-done');
    }
  });

  if (step === 'upload') {
    stepUpload.classList.add('is-active');
  } else if (step === 'loading') {
    stepUpload.classList.add('is-done');
    stepLoading.classList.add('is-active');
  } else if (step === 'analysis') {
    stepUpload.classList.add('is-done');
    stepLoading.classList.add('is-done');
    stepAnalysis.classList.add('is-active');
  } else if (step === 'history') {
    stepUpload.classList.add('is-done');
    stepLoading.classList.add('is-done');
    stepAnalysis.classList.add('is-done');
    stepHistory.classList.add('is-active');
  }
}

function showStage(stage) {
  uploadStage.hidden = stage !== 'upload';
  loadingStage.hidden = stage !== 'loading';
  analysisStage.hidden = stage !== 'analysis';
  historyStage.hidden = stage !== 'history';

  if (stage === 'history') {
    renderMainHistory(patientIdInput.value);
  }
}

async function renderMainHistory(patientId) {
  if (!patientId) {
    historyPatientTrend.textContent = 'Enter or generate a patient ID first.';
    historyFullList.innerHTML = '';
    return;
  }
  
  historySearchInput.value = patientId;

  try {
    const response = await fetch(`/patient/${encodeURIComponent(patientId)}/history?limit=50`);
    const data = await response.json();
    
    const trend = data.patient_summary?.trend || 'no_history';
    historyPatientTrend.textContent = `Patient: ${patientId} | Trend: ${trend} | Total uploads: ${data.patient_summary?.total_predictions ?? 0}`;
    
    historyFullList.innerHTML = '';
    (data.history || []).forEach((h) => {
      const li = document.createElement('li');
      li.className = 'search-item';
      
      const infoDiv = document.createElement('div');
      infoDiv.innerHTML = `<strong>${h.created_at}</strong><br>
        Predicted: ${h.predicted_disease} (${(h.prediction_confidence * 100).toFixed(1)}%)<br>
        <span class="badge ${h.confidence_band}">${h.confidence_band}</span>`;
      
      const actions = document.createElement('div');
      actions.style.display = 'flex';
      actions.style.gap = '6px';
      
      const viewImg = document.createElement('a');
      viewImg.className = 'mini-btn';
      viewImg.textContent = 'View X-ray';
      viewImg.href = h.image_url || '#';
      viewImg.target = '_blank';
      
      const viewRep = document.createElement('button');
      viewRep.type = 'button';
      viewRep.className = 'mini-btn';
      viewRep.textContent = 'View Report';
      viewRep.addEventListener('click', async () => {
         await loadRecord(h.id, true);
      });
      
      actions.appendChild(viewImg);
      actions.appendChild(viewRep);
      
      li.appendChild(infoDiv);
      li.appendChild(actions);
      historyFullList.appendChild(li);
    });
  } catch (err) {
    console.error(err);
    historyPatientTrend.textContent = 'Error loading history.';
  }
}

historySearchBtn?.addEventListener('click', () => {
   const q = historySearchInput.value.trim();
   if (q) {
     setCurrentPatientId(q);
     renderMainHistory(q);
   }
});

closeHistoryViewerBtn?.addEventListener('click', () => {
   historyRecordViewer.hidden = true;
});

function startFakeProgress() {
  let value = 0;
  progressBar.style.width = '0%';
  progressText.textContent = '0%';

  if (progressTimer) {
    clearInterval(progressTimer);
  }

  progressTimer = setInterval(() => {
    if (value < 90) {
      value += Math.random() * 9;
      if (value > 90) {
        value = 90;
      }
      progressBar.style.width = `${value.toFixed(0)}%`;
      progressText.textContent = `${value.toFixed(0)}%`;
    }
  }, 220);
}

function completeProgress() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
  progressBar.style.width = '100%';
  progressText.textContent = '100%';
}

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.style.color = isError ? '#b42318' : '#495857';
}

function clearResults() {
  showStage('upload');
  setStep('upload');
  resultsPanel.hidden = false;
  rulesList.innerHTML = '';
  reasoningList.innerHTML = '';
  historyList.innerHTML = '';
  patientTrend.textContent = '';
  searchResults.innerHTML = '';
  if(quickRecordViewer) quickRecordViewer.hidden = true;
  if(historyRecordViewer) historyRecordViewer.hidden = true;
  recordViewer.hidden = true;
  recordImage.removeAttribute('src');
  recordSummary.textContent = '';
  recordRules.innerHTML = '';
  recordReasoning.innerHTML = '';
  downloadJsonBtn.setAttribute('href', '#');
  openPrintBtn.setAttribute('href', '#');
  predictBtn.hidden = true;
  analysisImage.removeAttribute('src');
  analysisImageHint.textContent = 'Waiting for upload.';
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
  progressBar.style.width = '0%';
  progressText.textContent = '0%';
  bandBadge.className = 'badge';
}

function setPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    analysisImage.src = e.target.result;
    previewHint.textContent = file.name;
    analysisImageHint.textContent = file.name;
  };
  reader.readAsDataURL(file);
}

fileInput.addEventListener('change', () => {
  const keepPatientId = patientIdInput.value;
  clearResults();
  setCurrentPatientId(keepPatientId);
  const file = fileInput.files[0];
  if (file) {
    setPreview(file);
    predictBtn.hidden = false;
    setStatus('Ready to predict.');
  }
});

clearBtn.addEventListener('click', () => {
  const keepPatientId = patientIdInput.value;
  fileInput.value = '';
  previewImage.removeAttribute('src');
  previewHint.textContent = 'No image selected.';
  clearResults();
  setCurrentPatientId(keepPatientId);
  setStatus('');
});

newPatientBtn.addEventListener('click', async () => {
  try {
    await fetchNewPatientId();
    historyList.innerHTML = '';
    patientTrend.textContent = `Patient: ${patientIdInput.value} | Trend: no_history | Total records: 0`;
    setStatus(`Generated new patient ID ${patientIdInput.value}.`);
  } catch {
    setStatus('Failed to generate new patient ID.', true);
  }
});

searchBtn.addEventListener('click', async () => {
  try {
    await runPatientSearch();
  } catch {
    setStatus('Patient search failed.', true);
  }
});

patientSearch.addEventListener('keydown', async (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    try {
      await runPatientSearch();
    } catch {
      setStatus('Patient search failed.', true);
    }
  }
});

predictForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = fileInput.files[0];
  const patientId = patientIdInput.value.trim();

  if (!patientId) {
    setStatus('Please enter a Patient ID.', true);
    return;
  }

  if (!file) {
    setStatus('Please upload an image first.', true);
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('patient_id', patientId);

  setStep('loading');
  showStage('loading');
  startFakeProgress();
  setStatus('Running prediction and symbolic reasoning...');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    completeProgress();
    setStep('analysis');
    showStage('analysis');

    primaryDisease.textContent = data.predicted_disease;
    primaryConf.textContent = `Confidence: ${(data.prediction_confidence * 100).toFixed(2)}%`;

    secondaryDisease.textContent = data.secondary_disease;
    secondaryConf.textContent = `Confidence: ${(data.secondary_confidence * 100).toFixed(2)}%`;

    const band = data.symbolic_explanation.confidence_band;
    bandBadge.textContent = `${band} confidence`;
    bandBadge.className = `badge ${band}`;

    recommendation.textContent = data.symbolic_explanation.recommendation;

    rulesList.innerHTML = '';
    data.symbolic_explanation.rules_fired.forEach((rule) => {
      const li = document.createElement('li');
      li.textContent = rule;
      rulesList.appendChild(li);
    });

    reasoningList.innerHTML = '';
    data.symbolic_explanation.if_then_reasoning.forEach((line) => {
      const li = document.createElement('li');
      li.textContent = line;
      reasoningList.appendChild(li);
    });

    const trend = data.patient_summary?.trend || 'no_history';
    patientTrend.textContent = `Patient: ${data.patient_id} | Trend: ${trend} | Total records: ${data.patient_summary?.total_predictions ?? 0}`;

    historyList.innerHTML = '';
    (data.patient_history || []).forEach((h) => {
      const li = document.createElement('li');
      const top = document.createElement('div');
      top.className = 'history-row-top';
      top.innerHTML = `<div><strong>${h.predicted_disease}</strong> | ${(h.prediction_confidence * 100).toFixed(2)}% | ${h.created_at}</div>`;

      const actions = document.createElement('div');
      actions.className = 'history-actions';

      const viewImg = document.createElement('a');
      viewImg.className = 'mini-btn';
      viewImg.textContent = 'View X-ray';
      viewImg.href = h.image_url || '#';
      viewImg.target = '_blank';
      viewImg.rel = 'noopener';

      const viewReport = document.createElement('button');
      viewReport.type = 'button';
      viewReport.className = 'mini-btn';
      viewReport.textContent = 'View Report';
      viewReport.addEventListener('click', async () => {
        try {
          await loadRecord(h.id);
        } catch (err) {
          setStatus(err.message, true);
        }
      });

      actions.appendChild(viewImg);
      actions.appendChild(viewReport);
      top.appendChild(actions);
      li.appendChild(top);

      const rec = document.createElement('div');
      rec.textContent = `Band: ${h.confidence_band} | Secondary: ${h.secondary_disease || '-'} (${((h.secondary_confidence || 0) * 100).toFixed(2)}%)`;
      li.appendChild(rec);
      historyList.appendChild(li);
    });

    downloadJsonBtn.setAttribute('href', data.report_json_url || '#');
    openPrintBtn.setAttribute('href', data.report_print_url || '#');

    resultsPanel.hidden = false;
    setStatus('Prediction completed successfully.');
  } catch (error) {
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
    setStep('upload');
    showStage('upload');
    setStatus(error.message, true);
  }
});

(async () => {
  clearResults();
  try {
    await fetchNewPatientId();
    await runPatientSearch();
    setStatus('Ready. New patient ID generated automatically.');
  } catch {
    setStatus('Could not initialize patient ID. Refresh page and try again.', true);
  }
})();
