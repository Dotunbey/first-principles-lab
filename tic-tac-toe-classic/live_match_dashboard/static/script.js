const boardEl = document.getElementById('board');
const statusText = document.getElementById('statusText');
const gameCounter = document.getElementById('gameCounter');
const moveExplanation = document.getElementById('moveExplanation');
const scoreBoardEl = document.getElementById('scoreBoard');
const sourceInfoEl = document.getElementById('sourceInfo');
const outcomeLogEl = document.getElementById('outcomeLog');
const scoreChart = document.getElementById('scoreChart');
const scoreCtx = scoreChart.getContext('2d');

const NUMERIC_CONFIG_FIELDS = ['mcts_iterations', 'move_delay', 'games_per_run'];

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

document.getElementById('applyBtn').onclick = () => postJSON('/api/update_params', readConfigFromForm());
document.getElementById('startBtn').onclick = () => postJSON('/api/start', readConfigFromForm()).then(r => {
  if (!r.ok) statusText.textContent = r.error;
});
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

function renderScore(score, netPlays) {
  const total = score.net + score.qagent + score.draws;
  const pct = n => total ? (n / total * 100).toFixed(1) : '0.0';
  scoreBoardEl.innerHTML =
    `<table><tbody>` +
    `<tr><td>Network</td><td>${score.net} wins</td><td>${pct(score.net)}%</td></tr>` +
    `<tr><td>Q-agent</td><td>${score.qagent} wins</td><td>${pct(score.qagent)}%</td></tr>` +
    `<tr><td>Draws</td><td>${score.draws}</td><td>${pct(score.draws)}%</td></tr>` +
    `</tbody></table>` +
    `<p style="margin-top:10px;color:#9aa0aa;">Total games: ${total}${netPlays ? ` &middot; network is playing ${netPlays} this game` : ''}</p>`;
}

function renderSourceInfo(netIter, qagentGames) {
  if (netIter === null && qagentGames === null) { sourceInfoEl.innerHTML = ''; return; }
  sourceInfoEl.innerHTML =
    `<strong>Current checkpoints in this match:</strong>\n` +
    `network: iteration ${netIter ?? '?'} of its own training\n` +
    `Q-agent: ${qagentGames ?? '?'} games of its own training`;
}

function renderOutcomeLog(log) {
  outcomeLogEl.innerHTML = log.map(o => {
    const label = o.outcome === 'draw' ? 'draw' : (o.outcome === 'net_win' ? 'network won' : 'Q-agent won');
    return `<div>#${o.game_num} - network played ${o.net_plays} - ${label}</div>`;
  }).join('');
}

function drawScoreChart(history) {
  scoreCtx.clearRect(0, 0, scoreChart.width, scoreChart.height);
  scoreCtx.strokeStyle = '#333';
  scoreCtx.beginPath();
  scoreCtx.moveTo(40, 10); scoreCtx.lineTo(40, 230); scoreCtx.lineTo(470, 230);
  scoreCtx.stroke();

  if (history.length === 0) return;

  const maxGame = history[history.length - 1].game_num;
  const xFor = g => 40 + (g / maxGame) * 420;
  const yFor = pct => 230 - (pct / 100) * 220;

  scoreCtx.fillStyle = '#6b7280';
  scoreCtx.font = '11px sans-serif';
  scoreCtx.fillText('100%', 4, 14);
  scoreCtx.fillText('0%', 12, 230);

  const series = [
    { key: 'net_win_rate', color: '#3b82f6', label: 'network' },
    { key: 'qagent_win_rate', color: '#ef4444', label: 'Q-agent' },
    { key: 'draw_rate', color: '#9aa0aa', label: 'draws' },
  ];
  series.forEach(s => {
    scoreCtx.strokeStyle = s.color;
    scoreCtx.beginPath();
    history.forEach((pt, i) => {
      const x = xFor(pt.game_num), y = yFor(pt[s.key]);
      if (i === 0) scoreCtx.moveTo(x, y); else scoreCtx.lineTo(x, y);
    });
    scoreCtx.stroke();
  });

  let legendX = 300;
  series.forEach(s => {
    scoreCtx.fillStyle = s.color;
    scoreCtx.fillText('●', legendX, 20);
    scoreCtx.fillStyle = '#9aa0aa';
    scoreCtx.fillText(s.label, legendX + 10, 20);
    legendX += 70;
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
    gameCounter.textContent = s.games_target
      ? `- game ${s.game_num}/${s.games_target}`
      : (s.game_num ? `- ${s.game_num} games played so far` : '');
    moveExplanation.textContent = s.move_explanation || '';
    renderScore(s.score, s.net_plays);
    renderSourceInfo(s.net_source_iteration, s.qagent_source_games);
    renderOutcomeLog(s.outcome_log || []);
    drawScoreChart(s.score_history || []);

    statusText.textContent = s.stage || 'idle';
  } catch (e) {
    statusText.textContent = 'disconnected';
  }
  setTimeout(poll, 300);
}

poll();
