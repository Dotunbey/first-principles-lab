const statusText = document.getElementById('statusText');
const recentLog = document.getElementById('recentLog');
const errorChart = document.getElementById('errorChart');
const errorCtx = errorChart.getContext('2d');
const lossChart = document.getElementById('lossChart');
const lossCtx = lossChart.getContext('2d');
const outcomeChart = document.getElementById('outcomeChart');
const outcomeCtx = outcomeChart.getContext('2d');

function drawAxes(ctx, canvas, yLabel) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#333';
  ctx.beginPath();
  ctx.moveTo(45, 10); ctx.lineTo(45, 230); ctx.lineTo(470, 230);
  ctx.stroke();
}

function drawLineChart(ctx, canvas, rows, series, colors, maxYOverride) {
  drawAxes(ctx, canvas);
  if (rows.length === 0) return;

  const maxIter = rows[rows.length - 1].iteration;
  let maxY = maxYOverride;
  if (!maxY) {
    maxY = 0.01;
    series.forEach(key => rows.forEach(r => { if (r[key] > maxY) maxY = r[key]; }));
  }

  const xFor = it => 45 + (it / maxIter) * 415;
  const yFor = val => 230 - (val / maxY) * 220;

  ctx.fillStyle = '#6b7280';
  ctx.font = '11px sans-serif';
  ctx.fillText(maxY.toFixed(2), 4, 14);
  ctx.fillText('0', 30, 230);

  series.forEach((key, si) => {
    ctx.strokeStyle = colors[si];
    ctx.beginPath();
    rows.forEach((r, i) => {
      const x = xFor(r.iteration), y = yFor(r[key]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  ctx.font = '11px sans-serif';
  series.forEach((key, si) => {
    ctx.fillStyle = colors[si];
    ctx.fillText('● ' + key, 50 + si * 150, 20);
  });
}

function renderRecentLog(rows) {
  const recent = rows.slice(-15).reverse();
  recentLog.innerHTML = recent.map(r =>
    `<div>#${r.iteration}: X=${r.wins_X} O=${r.wins_O} draw=${r.draws} | ` +
    `value_err=${r.value_error_vs_solver.toFixed(4)} | ${r.elapsed_sec.toFixed(1)}s</div>`
  ).join('');
}

async function poll() {
  try {
    const res = await fetch('/api/log');
    const data = await res.json();
    const rows = data.rows;

    if (rows.length === 0) {
      statusText.textContent = 'Waiting for training log to appear...';
    } else {
      const last = rows[rows.length - 1];
      statusText.textContent =
        `Iteration ${last.iteration} | latest value_error_vs_solver = ${last.value_error_vs_solver.toFixed(4)} | ` +
        `value_loss = ${last.value_loss.toFixed(4)} | policy_loss = ${last.policy_loss.toFixed(4)}`;
    }

    drawLineChart(errorCtx, errorChart, rows, ['value_error_vs_solver'], ['#f59e0b']);
    drawLineChart(lossCtx, lossChart, rows, ['value_loss', 'policy_loss'], ['#3b82f6', '#a855f7']);
    drawLineChart(outcomeCtx, outcomeChart, rows, ['wins_X', 'wins_O', 'draws'], ['#3b82f6', '#ef4444', '#9aa0aa'], 20);
    renderRecentLog(rows);
  } catch (e) {
    statusText.textContent = 'disconnected';
  }
  setTimeout(poll, 2000);
}

poll();
