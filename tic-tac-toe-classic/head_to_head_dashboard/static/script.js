const boardEl = document.getElementById('board');
const statusText = document.getElementById('statusText');
const gameCounter = document.getElementById('gameCounter');
const moveExplanation = document.getElementById('moveExplanation');
const sideTag = document.getElementById('sideTag');
const aLabel = document.getElementById('aLabel');
const bLabel = document.getElementById('bLabel');
const aWinsEl = document.getElementById('aWins');
const bWinsEl = document.getElementById('bWins');
const drawsEl = document.getElementById('draws');
const tableSizes = document.getElementById('tableSizes');
const outcomeLogEl = document.getElementById('outcomeLog');
const rateChart = document.getElementById('rateChart');
const rateCtx = rateChart.getContext('2d');

function postJSON(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  }).then(r => r.json());
}

function readConfigFromForm() {
  return {
    move_delay: parseFloat(document.getElementById('cfg_move_delay').value),
    games_per_run: parseInt(document.getElementById('cfg_games_per_run').value, 10),
    rolling_window: parseInt(document.getElementById('cfg_rolling_window').value, 10),
  };
}

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

function renderOutcomeLog(log) {
  outcomeLogEl.innerHTML = log.map(o =>
    `<div class="${o.outcome}">#${o.game_count} - ${o.outcome === 'draw' ? 'draw' : (o.outcome === 'a' ? 'A wins' : 'B wins')} (A played ${o.a_side})</div>`
  ).join('');
}

function drawRateChart(history) {
  rateCtx.clearRect(0, 0, rateChart.width, rateChart.height);
  rateCtx.strokeStyle = '#333';
  rateCtx.beginPath();
  rateCtx.moveTo(40, 10); rateCtx.lineTo(40, 230); rateCtx.lineTo(470, 230);
  rateCtx.stroke();

  rateCtx.fillStyle = '#6b7280';
  rateCtx.font = '11px sans-serif';
  rateCtx.fillText('100%', 4, 14);
  rateCtx.fillText('50%', 8, 122);
  rateCtx.fillText('0%', 12, 230);

  rateCtx.strokeStyle = '#374151';
  rateCtx.setLineDash([3, 3]);
  rateCtx.beginPath();
  rateCtx.moveTo(40, 120); rateCtx.lineTo(470, 120);
  rateCtx.stroke();
  rateCtx.setLineDash([]);

  if (history.length === 0) return;

  const maxGame = history[history.length - 1].game_count;
  const xFor = g => 40 + (g / maxGame) * 420;
  const yFor = pct => 230 - (pct / 100) * 220;

  const series = [
    { key: 'a_win_rate', color: '#3b82f6' },
    { key: 'b_win_rate', color: '#ef4444' },
    { key: 'draw_rate', color: '#9aa0aa' },
  ];
  series.forEach(s => {
    rateCtx.strokeStyle = s.color;
    rateCtx.beginPath();
    history.forEach((pt, i) => {
      const x = xFor(pt.game_count), y = yFor(pt[s.key]);
      if (i === 0) rateCtx.moveTo(x, y); else rateCtx.lineTo(x, y);
    });
    rateCtx.stroke();
  });

  const last = history[history.length - 1];
  rateCtx.fillStyle = '#e8e8ea';
  rateCtx.font = '12px sans-serif';
  rateCtx.fillText(`A=${last.a_win_rate.toFixed(0)}% B=${last.b_win_rate.toFixed(0)}% draw=${last.draw_rate.toFixed(0)}% after game ${last.game_count}`, 45, 20);
}

async function poll() {
  try {
    const res = await fetch('/api/state');
    const s = await res.json();

    aLabel.textContent = s.a_label;
    bLabel.textContent = s.b_label;
    renderBoard(s.board);
    gameCounter.textContent = s.games_target ? `- game ${s.game_count}/${s.games_target}` : (s.game_count ? `- ${s.game_count} games played` : '');
    sideTag.textContent = s.a_side ? `(A is ${s.a_side} this game)` : '';
    moveExplanation.textContent = s.move_explanation || '';
    aWinsEl.textContent = s.scoreboard.a_wins;
    bWinsEl.textContent = s.scoreboard.b_wins;
    drawsEl.textContent = s.scoreboard.draws;
    tableSizes.textContent = `A table size: ${s.a_table_size} | B table size: ${s.b_table_size}`;
    renderOutcomeLog(s.outcome_log || []);
    drawRateChart(s.history || []);

    statusText.textContent = s.stage || 'idle';
  } catch (e) {
    statusText.textContent = 'disconnected';
  }
  setTimeout(poll, 300);
}

poll();
