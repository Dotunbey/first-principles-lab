const boardEl = document.getElementById('board');
const statusText = document.getElementById('statusText');
const gameCounter = document.getElementById('gameCounter');
const moveExplanation = document.getElementById('moveExplanation');
const mctsStatsEl = document.getElementById('mctsStats');
const trainingMathEl = document.getElementById('trainingMath');
const outcomeLogEl = document.getElementById('outcomeLog');
const initialWeightsEl = document.getElementById('initialWeights');
const lossChart = document.getElementById('lossChart');
const lossCtx = lossChart.getContext('2d');
const winChart = document.getElementById('winChart');
const winCtx = winChart.getContext('2d');

const NUMERIC_CONFIG_FIELDS = [
  'mcts_iterations', 'temperature_moves', 'noise_frac', 'learning_rate',
  'batch_size', 'games_per_iteration', 'iterations_per_run', 'move_delay',
];

function readConfigFromForm() {
  const cfg = {};
  NUMERIC_CONFIG_FIELDS.forEach(key => {
    cfg[key] = parseFloat(document.getElementById('cfg_' + key).value);
  });
  return cfg;
}

function postJSON(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  }).then(r => r.json());
}

document.getElementById('initBtn').onclick = () => postJSON('/api/init', readConfigFromForm());
document.getElementById('applyBtn').onclick = () => postJSON('/api/update_params', readConfigFromForm());
document.getElementById('startBtn').onclick = () => postJSON('/api/start', readConfigFromForm());
document.getElementById('pauseBtn').onclick = () => postJSON('/api/pause');
document.getElementById('resumeBtn').onclick = () => postJSON('/api/resume');
document.getElementById('stopBtn').onclick = () => postJSON('/api/stop');

function renderBoard(board) {
  boardEl.innerHTML = '';
  board.forEach(v => {
    const cell = document.createElement('div');
    cell.className = 'cell' + (v === 1 ? ' x' : v === 2 ? ' o' : '');
    cell.textContent = v === 1 ? 'X' : v === 2 ? 'O' : '';
    boardEl.appendChild(cell);
  });
}

function renderMctsStats(stats) {
  if (!stats || stats.length === 0) {
    mctsStatsEl.textContent = 'Waiting for a move...';
    return;
  }
  const maxVisits = Math.max(...stats.map(s => s.visits));
  const rows = stats.map(s => {
    const chosen = s.visits === maxVisits ? ' class="chosen"' : '';
    return `<tr${chosen}><td>${JSON.stringify(s.move)}</td><td>${s.visits}</td><td>${s.prior}</td><td>${s.value}</td><td>${s.puct}</td></tr>`;
  }).join('');
  mctsStatsEl.innerHTML =
    `<table><thead><tr><th>move</th><th>visits</th><th>prior</th><th>value</th><th>puct</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderTrainingMath(m) {
  if (!m) {
    trainingMathEl.textContent = 'Waiting for the first training step...';
    return;
  }
  const norms = Object.entries(m.weight_update_norms || {})
    .map(([k, v]) => `  ||Δ${k}|| = ${v}`).join('\n');
  trainingMathEl.innerHTML =
    `<span class="formula">${m.value_formula}</span>\n` +
    `<span class="formula">${m.policy_formula}</span>\n\n` +
    `Weight-update sizes this step:\n${norms}`;
}

function renderInitialWeights(w) {
  if (!w) { initialWeightsEl.innerHTML = ''; return; }
  initialWeightsEl.innerHTML =
    `<strong>Network:</strong>\n` +
    `hidden units = ${w.hidden_size}\n` +
    `learning rate = ${w.learning_rate}\n` +
    `${w.note}`;
}

function renderOutcomeLog(log) {
  outcomeLogEl.innerHTML = log.map(o =>
    `<div>iter ${o.iteration}: X=${o.wins_X} O=${o.wins_O} draw=${o.draws}</div>`
  ).join('');
}

function drawLossChart(history) {
  const points = history.filter(pt => pt.value_error !== null && pt.value_error !== undefined);
  lossCtx.clearRect(0, 0, lossChart.width, lossChart.height);
  lossCtx.strokeStyle = '#333';
  lossCtx.beginPath();
  lossCtx.moveTo(40, 10); lossCtx.lineTo(40, 230); lossCtx.lineTo(470, 230);
  lossCtx.stroke();

  if (points.length === 0) {
    lossCtx.fillStyle = '#6b7280';
    lossCtx.font = '12px sans-serif';
    lossCtx.fillText('No iterations with value_error yet', 45, 120);
    return;
  }

  const maxIter = points[points.length - 1].iteration;
  const maxError = Math.max(...points.map(p => p.value_error), 0.01);
  const xFor = it => 40 + (it / maxIter) * 420;
  const yFor = err => 230 - (err / maxError) * 220;

  lossCtx.fillStyle = '#6b7280';
  lossCtx.font = '11px sans-serif';
  lossCtx.fillText(maxError.toFixed(3), 4, 14);
  lossCtx.fillText('0', 12, 230);

  lossCtx.strokeStyle = '#f59e0b';
  lossCtx.beginPath();
  points.forEach((pt, i) => {
    const x = xFor(pt.iteration), y = yFor(pt.value_error);
    if (i === 0) lossCtx.moveTo(x, y); else lossCtx.lineTo(x, y);
  });
  lossCtx.stroke();

  const last = points[points.length - 1];
  lossCtx.fillStyle = '#e8e8ea';
  lossCtx.font = '12px sans-serif';
  lossCtx.fillText(`latest: ${last.value_error.toFixed(4)} (X=${(last.value_error_X||0).toFixed(3)}, O=${(last.value_error_O||0).toFixed(3)}) after iter ${last.iteration}`, 45, 20);
}

function drawWinChart(history) {
  winCtx.clearRect(0, 0, winChart.width, winChart.height);
  winCtx.strokeStyle = '#333';
  winCtx.beginPath();
  winCtx.moveTo(40, 10); winCtx.lineTo(40, 230); winCtx.lineTo(470, 230);
  winCtx.stroke();

  if (history.length === 0) return;

  const maxIter = history[history.length - 1].iteration;
  const gamesPerIter = history[0].wins_X + history[0].wins_O + history[0].draws || 1;
  const xFor = it => 40 + (it / maxIter) * 420;
  const yFor = frac => 230 - frac * 220;

  winCtx.fillStyle = '#6b7280';
  winCtx.font = '11px sans-serif';
  winCtx.fillText('100%', 4, 14);
  winCtx.fillText('0%', 12, 230);

  const series = [
    { key: 'wins_X', color: '#3b82f6', label: 'X wins' },
    { key: 'wins_O', color: '#ef4444', label: 'O wins' },
    { key: 'draws', color: '#9aa0aa', label: 'draws' },
  ];
  series.forEach(s => {
    winCtx.strokeStyle = s.color;
    winCtx.beginPath();
    history.forEach((pt, i) => {
      const total = pt.wins_X + pt.wins_O + pt.draws || 1;
      const x = xFor(pt.iteration), y = yFor(pt[s.key] / total);
      if (i === 0) winCtx.moveTo(x, y); else winCtx.lineTo(x, y);
    });
    winCtx.stroke();
  });

  let legendX = 300;
  series.forEach(s => {
    winCtx.fillStyle = s.color;
    winCtx.fillText('●', legendX, 20);
    winCtx.fillStyle = '#9aa0aa';
    winCtx.fillText(s.label, legendX + 10, 20);
    legendX += 60;
  });
}

let formSynced = false;
function syncFormFromConfig(cfg) {
  if (formSynced || !cfg) return;
  NUMERIC_CONFIG_FIELDS.forEach(key => {
    const el = document.getElementById('cfg_' + key);
    if (el && cfg[key] !== undefined && cfg[key] !== null) el.value = cfg[key];
  });
  formSynced = true;
}

async function poll() {
  try {
    const res = await fetch('/api/state');
    const s = await res.json();

    syncFormFromConfig(s.config);
    renderBoard(s.board);
    gameCounter.textContent = s.iterations_target
      ? `- iter ${s.iteration}/${s.iterations_target}, self-play game ${s.game_in_iteration}/${s.games_this_iteration_target}`
      : (s.iteration ? `- ${s.iteration} iterations trained so far` : '');
    moveExplanation.textContent = s.move_explanation || '';
    renderMctsStats(s.mcts_stats);
    renderTrainingMath(s.training_math);
    renderOutcomeLog(s.outcome_log || []);
    renderInitialWeights(s.initial_weights);
    drawLossChart(s.iteration_history || []);
    drawWinChart(s.iteration_history || []);

    let status = s.stage || 'idle';
    status += ` | buffer=${s.buffer_size}`;
    statusText.textContent = status;
  } catch (e) {
    statusText.textContent = 'disconnected';
  }
  setTimeout(poll, 300);
}

poll();
