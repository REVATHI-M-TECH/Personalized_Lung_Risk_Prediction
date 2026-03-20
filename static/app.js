const fileInput = document.getElementById('fileInput');
const patientIdInput = document.getElementById('patientId');
const predictForm = document.getElementById('predictForm');
const clearBtn = document.getElementById('clearBtn');
const statusBox = document.getElementById('status');
const previewImage = document.getElementById('previewImage');
const previewHint = document.getElementById('previewHint');
const resultsPanel = document.getElementById('resultsPanel');

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

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.style.color = isError ? '#b42318' : '#495857';
}

function clearResults() {
  resultsPanel.hidden = true;
  rulesList.innerHTML = '';
  reasoningList.innerHTML = '';
  historyList.innerHTML = '';
  patientTrend.textContent = '';
  bandBadge.className = 'badge';
}

function setPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewHint.textContent = file.name;
  };
  reader.readAsDataURL(file);
}

fileInput.addEventListener('change', () => {
  clearResults();
  const file = fileInput.files[0];
  if (file) {
    setPreview(file);
    setStatus('Ready to predict.');
  }
});

clearBtn.addEventListener('click', () => {
  fileInput.value = '';
  previewImage.removeAttribute('src');
  previewHint.textContent = 'No image selected.';
  clearResults();
  setStatus('');
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
      li.textContent = `${h.created_at} | ${h.predicted_disease} | conf ${(h.prediction_confidence * 100).toFixed(2)}% | ${h.confidence_band}`;
      historyList.appendChild(li);
    });

    resultsPanel.hidden = false;
    setStatus('Prediction completed successfully.');
  } catch (error) {
    setStatus(error.message, true);
  }
});
