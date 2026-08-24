const boardEl = document.getElementById('board');
const statusText = document.getElementById('statusText');
const gameCounter = document.getElementById('gameCounter');
const moveExplanation = document.getElementById('moveExplanation');
const qUpdateEl = document.getElementById('qUpdate');
const outcomeLogEl = document.getElementById('outcomeLog');
const initialWeightsEl = document.getElementById('initialWeights');
const weightFeedEl = document.getElementById('weightFeed');
const speedSlider = document.getElementById('speedSlider');
const speedLabel = document.getElementById('speedLabel');
const chart = document.getElementById('chart');
const ctx = chart.getContext('2d');
const lossChart = document.getElementById('lossChart');
const lossCtx = lossChart.getContext('2d');

// Slider 0 (slow) -> 100 (fast) maps to a delay between MAX_DELAY and MIN_DELAY seconds,
// applied to both move_delay and update_delay live, no page reload needed.
// IMPORTANT: the slider must never fire on page load/poll - only on a real
// user drag - otherwise it silently overwrites move_delay/update_delay that
// were set some other way (e.g. typed directly into the config fields).
const MIN_DELAY = 0.05;
const MAX_DELAY = 2.0;

function delayFromSlider(value) {
  const t = value / 100;
  return MAX_DELAY - t * (MAX_DELAY - MIN_DELAY);
}

function sliderValueFromDelay(delay) {
  const t = (MAX_DELAY - delay) / (MAX_DELAY - MIN_DELAY);
  return Math.max(0, Math.min(100, t * 100));
}

function applySpeed() {
  const delay = delayFromSlider(parseFloat(speedSlider.value));
  speedLabel.textContent = delay.toFixed(2) + 's/step';
  postJSON('/api/update_params', { move_delay: delay, update_delay: delay });
}

speedSlider.oninput = () => { speedLabel.textContent = delayFromSlider(parseFloat(speedSlider.value)).toFixed(2) + 's/step'; };
speedSlider.onchange = applySpeed;  // only fires on an actual user interaction, never programmatically

const NUMERIC_CONFIG_FIELDS = [
  'learning_rate', 'discount_factor', 'epsilon_start', 'epsilon_end',
  'epsilon_decay_games', 'mcts_iterations', 'games_per_run',
  'move_delay', 'update_delay', 'checkpoint_interval',
  'checkpoint_eval_games', 'checkpoint_mcts_iterations', 'draw_reward',
];
const CONFIG_FIELDS = ['opponent', 'reward_shaping', 'td0', ...NUMERIC_CONFIG_FIELDS];

// Single dropdown in the UI, maps to two independent backend flags:
// sparse -> reward_shaping=0, td0=0 | shaped -> reward_shaping=1, td0=0 | td0 -> td0=1
function readConfigFromForm() {
  const updateRule = document.getElementById('cfg_update_rule').value;
  const cfg = {
    opponent: document.getElementById('cfg_opponent').value,
    reward_shaping: updateRule === 'shaped' ? 1 : 0,
    td0: updateRule === 'td0' ? 1 : 0,
  };
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

function renderQUpdate(u) {
  if (!u) {
    qUpdateEl.textContent = 'Waiting for a game to finish...';
    return;
  }
  const modeTag = u.mode === 'td0' ? 'TD(0) online' : (u.shaped ? 'potential-shaped' : 'sparse');
  qUpdateEl.innerHTML =
    (u.mode === 'td0' ? `Online update (immediate, no backward pass)` : `Backward step ${u.step} / ${u.total_steps}`) +
    ` <span class="shaped-tag">[${modeTag}]</span>` + `\n` +
    `state: ${u.state_hash}\n` +
    `action: ${JSON.stringify(u.action)}\n` +
    `old Q(s,a): ${u.old_q}\n` +
    `learning rate (alpha): ${u.learning_rate}\n` +
    `target return (G): ${u.target_G}\n` +
    `<span class="formula">${u.formula}</span>`;
}

function renderInitialWeights(w) {
  if (!w) {
    initialWeightsEl.innerHTML = '';
    return;
  }
  initialWeightsEl.innerHTML =
    `<strong>Initial weights (fresh agent):</strong>\n` +
    `learning rate (alpha) = ${w.learning_rate}\n` +
    `discount factor (gamma) = ${w.discount_factor}\n` +
    `epsilon: ${w.epsilon_start} -> ${w.epsilon_end} over ${w.epsilon_decay_games} games\n` +
    `Q-table entries = ${w.q_table_entries}\n` +
    `${w.note}`;
}

function renderWeightFeed(log) {
  if (!log || log.length === 0) {
    weightFeedEl.innerHTML = '';
    return;
  }
  weightFeedEl.innerHTML = log.map(u => {
    const direction = u.new_q > u.old_q ? 'up' : (u.new_q < u.old_q ? 'down' : 'flat');
    return `<div class="weight-row ${direction}">` +
      `#${u.game_num} step ${u.step}/${u.total_steps} &nbsp; ` +
      `${JSON.stringify(u.action)} &nbsp; ` +
      `${u.old_q.toFixed(3)} &rarr; <strong>${u.new_q.toFixed(3)}</strong>` +
      `</div>`;
  }).join('');
}

function renderOutcomeLog(log) {
  outcomeLogEl.innerHTML = log.map(g =>
    `<div class="${g.outcome}">#${g.game_num} - Q played ${g.q_plays} - ${g.outcome}` +
    ` <span class="opponent-tag">(vs ${g.opponent === 'perfect' ? 'perfect solver' : 'MCTS'})</span></div>`
  ).join('');
}

function drawChart(history) {
  ctx.clearRect(0, 0, chart.width, chart.height);
  ctx.strokeStyle = '#333';
  ctx.beginPath();
  ctx.moveTo(40, 10); ctx.lineTo(40, 230); ctx.lineTo(470, 230);
  ctx.stroke();

  ctx.fillStyle = '#6b7280';
  ctx.font = '11px sans-serif';
  ctx.fillText('100%', 4, 14);
  ctx.fillText('50%', 8, 122);
  ctx.fillText('0%', 12, 230);

  ctx.strokeStyle = '#374151';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(40, 120); ctx.lineTo(470, 120);
  ctx.stroke();
  ctx.setLineDash([]);

  if (history.length === 0) return;

  const maxGame = history[history.length - 1].game_num;
  const xFor = g => 40 + (g / maxGame) * 420;
  const yFor = pct => 230 - (pct / 100) * 220;

  ctx.strokeStyle = '#3b82f6';
  ctx.beginPath();
  history.forEach((pt, i) => {
    const x = xFor(pt.game_num), y = yFor(pt.win_rate);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  history.forEach(pt => {
    ctx.fillStyle = pt.opponent === 'perfect' ? '#a855f7' : '#3b82f6';
    ctx.beginPath();
    ctx.arc(xFor(pt.game_num), yFor(pt.win_rate), 4, 0, 7);
    ctx.fill();
  });

  const last = history[history.length - 1];
  const oppLabel = last.opponent === 'perfect' ? 'perfect solver' : 'MCTS';
  ctx.fillStyle = '#e8e8ea';
  ctx.font = '12px sans-serif';
  ctx.fillText(`latest: ${last.win_rate.toFixed(1)}% win rate vs ${oppLabel} after ${last.game_num} games`, 45, 20);
  ctx.fillStyle = '#3b82f6'; ctx.fillText('●', 340, 20);
  ctx.fillStyle = '#9aa0aa'; ctx.fillText('MCTS', 350, 20);
  ctx.fillStyle = '#a855f7'; ctx.fillText('●', 400, 20);
  ctx.fillStyle = '#9aa0aa'; ctx.fillText('perfect', 410, 20);
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
    lossCtx.fillText('No checkpoints with value_error yet - wait for the next one', 45, 120);
    return;
  }

  const maxGame = points[points.length - 1].game_num;
  const maxError = Math.max(...points.map(p => p.value_error), 0.01);
  const xFor = g => 40 + (g / maxGame) * 420;
  const yFor = err => 230 - (err / maxError) * 220;

  lossCtx.fillStyle = '#6b7280';
  lossCtx.font = '11px sans-serif';
  lossCtx.fillText(maxError.toFixed(3), 4, 14);
  lossCtx.fillText('0', 12, 230);

  lossCtx.strokeStyle = '#f59e0b';
  lossCtx.beginPath();
  points.forEach((pt, i) => {
    const x = xFor(pt.game_num), y = yFor(pt.value_error);
    if (i === 0) lossCtx.moveTo(x, y); else lossCtx.lineTo(x, y);
  });
  lossCtx.stroke();

  points.forEach(pt => {
    lossCtx.fillStyle = '#f59e0b';
    lossCtx.beginPath();
    lossCtx.arc(xFor(pt.game_num), yFor(pt.value_error), 3, 0, 7);
    lossCtx.fill();
  });

  const last = points[points.length - 1];
  lossCtx.fillStyle = '#e8e8ea';
  lossCtx.font = '12px sans-serif';
  lossCtx.fillText(`latest: mean |error| = ${last.value_error.toFixed(4)} after ${last.game_num} games`, 45, 20);
}

let formSynced = false;

function syncFormFromConfig(cfg) {
  if (formSynced || !cfg) return;
  CONFIG_FIELDS.forEach(key => {
    const el = document.getElementById('cfg_' + key);
    if (el && cfg[key] !== undefined) el.value = cfg[key];
  });
  if (cfg.td0 !== undefined && cfg.reward_shaping !== undefined) {
    document.getElementById('cfg_update_rule').value = cfg.td0 ? 'td0' : (cfg.reward_shaping ? 'shaped' : 'sparse');
  }
  // Read-only sync, matches the real current delay instead of a stale
  // default - setting .value here does NOT fire onchange/oninput, so this
  // can never accidentally re-POST and overwrite anything.
  if (cfg.move_delay !== undefined) {
    speedSlider.value = sliderValueFromDelay(cfg.move_delay);
    speedLabel.textContent = cfg.move_delay.toFixed(2) + 's/step';
  }
  formSynced = true;
}

async function poll() {
  try {
    const res = await fetch('/api/state');
    const s = await res.json();

    syncFormFromConfig(s.config);
    renderBoard(s.board);
    gameCounter.textContent = s.games_target
      ? `- game ${s.game_num}/${s.games_target} (Q playing ${s.q_plays || '-'})`
      : (s.game_num ? `- ${s.game_num} games trained so far` : '');
    moveExplanation.textContent = s.move_explanation || '';
    renderQUpdate(s.q_update);
    renderWeightFeed(s.q_update_log || []);
    renderOutcomeLog(s.outcome_log || []);
    renderInitialWeights(s.initial_weights);
    drawChart(s.checkpoint_history || []);
    drawLossChart(s.checkpoint_history || []);

    let status = s.stage || 'idle';
    if (s.epsilon !== null && s.epsilon !== undefined) {
      status += ` | epsilon=${s.epsilon} | Q-table size=${s.q_table_size}`;
    }
    statusText.textContent = status;
  } catch (e) {
    statusText.textContent = 'disconnected';
  }
  setTimeout(poll, 300);
}

poll();

// --- Play vs human --------------------------------------------------------
const humanBoardEl = document.getElementById('humanBoard');
const humanExplanationEl = document.getElementById('humanExplanation');
const humanMoveLogEl = document.getElementById('humanMoveLog');
const humanStatusText = document.getElementById('humanStatusText');
const humanSymbolSelect = document.getElementById('humanSymbol');

let humanSelectedFrom = null;  // mid-selection for the sliding phase: piece picked up, destination not chosen yet
let humanStateCache = null;

document.getElementById('humanNewGameBtn').onclick = () => {
  humanSelectedFrom = null;
  postJSON('/api/human/new', { human_symbol: humanSymbolSelect.value }).then(r => {
    if (!r.ok) { humanStatusText.textContent = r.error; return; }
    humanStatusText.textContent = '';
    renderHumanState(r.state);
  });
};

function movesEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function submitHumanMove(move) {
  humanSelectedFrom = null;
  postJSON('/api/human/move', { move }).then(r => {
    if (!r.ok) { humanStatusText.textContent = r.error; return; }
    humanStatusText.textContent = '';
    renderHumanState(r.state);
  });
}

function renderHumanState(s) {
  humanStateCache = s;
  humanBoardEl.innerHTML = '';
  const isHumanTurn = s.active && !s.game_over && s.current_player === s.human_symbol;

  s.board.forEach((v, i) => {
    const cell = document.createElement('div');
    cell.className = 'cell' + (v === 1 ? ' x' : v === 2 ? ' o' : '');
    cell.textContent = v === 1 ? 'X' : v === 2 ? 'O' : '';

    if (isHumanTurn && s.phase === 'placement' && s.valid_moves.includes(i)) {
      cell.classList.add('playable');
      cell.onclick = () => submitHumanMove(i);
    } else if (isHumanTurn && s.phase === 'sliding') {
      if (humanSelectedFrom === null) {
        const canPickUp = s.valid_moves.some(m => m[0] === i);
        if (canPickUp) {
          cell.classList.add('playable');
          cell.onclick = () => { humanSelectedFrom = i; renderHumanState(s); };
        }
      } else if (humanSelectedFrom === i) {
        cell.classList.add('selected');
        cell.onclick = () => { humanSelectedFrom = null; renderHumanState(s); };
      } else {
        const isDestination = s.valid_moves.some(m => movesEqual(m, [humanSelectedFrom, i]));
        if (isDestination) {
          cell.classList.add('playable');
          cell.onclick = () => submitHumanMove([humanSelectedFrom, i]);
        }
      }
    }
    humanBoardEl.appendChild(cell);
  });

  let statusMsg = '';
  if (!s.active) {
    statusMsg = 'Click "New Game" to play.';
  } else if (s.game_over) {
    statusMsg = s.winner === 'draw' ? 'Game over: draw.'
      : (s.winner === s.human_symbol ? 'Game over: you win!' : 'Game over: agent wins.');
  } else if (isHumanTurn) {
    statusMsg = s.phase === 'sliding' && humanSelectedFrom !== null
      ? 'Pick a destination for the selected piece.'
      : 'Your move.';
  } else {
    statusMsg = "Agent's turn...";
  }
  humanStatusText.textContent = statusMsg;
  humanExplanationEl.textContent = s.agent_move_explanation || '';
  humanMoveLogEl.innerHTML = (s.move_log || []).map(m =>
    `<div>${m.mover === 'human' ? 'You' : 'Agent'} played ${JSON.stringify(m.move)}</div>`
  ).join('');
}

async function pollHuman() {
  try {
    const res = await fetch('/api/human/state');
    const s = await res.json();
    if (humanSelectedFrom !== null && (!s.active || s.game_over)) humanSelectedFrom = null;
    renderHumanState(s);
  } catch (e) {
    // leave last known state on screen rather than blanking it on a transient error
  }
  setTimeout(pollHuman, 500);
}

pollHuman();
