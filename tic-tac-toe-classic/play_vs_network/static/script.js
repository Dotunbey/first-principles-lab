const boardEl = document.getElementById('board');
const explanationEl = document.getElementById('explanation');
const moveLogEl = document.getElementById('moveLog');
const statusText = document.getElementById('statusText');
const variantSelect = document.getElementById('variantSelect');
const humanSymbolSelect = document.getElementById('humanSymbol');
const newGameBtn = document.getElementById('newGameBtn');

function postJSON(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  }).then(r => r.json());
}

async function loadVariants() {
  const res = await fetch('/api/variants');
  const data = await res.json();
  variantSelect.innerHTML = data.variants.map(v => `<option value="${v}">${v}</option>`).join('');
}

newGameBtn.onclick = () => {
  postJSON('/api/new', { variant: variantSelect.value, human_symbol: humanSymbolSelect.value }).then(r => {
    if (!r.ok) { statusText.textContent = r.error; return; }
    render(r.state);
  });
};

function submitMove(move) {
  postJSON('/api/move', { move }).then(r => {
    if (!r.ok) { statusText.textContent = r.error; return; }
    render(r.state);
  });
}

function render(s) {
  boardEl.innerHTML = '';
  const isHumanTurn = s.active && !s.game_over && s.current_player === s.human_symbol;

  s.board.forEach((v, i) => {
    const cell = document.createElement('div');
    cell.className = 'cell' + (v === 1 ? ' x' : v === 2 ? ' o' : '');
    cell.textContent = v === 1 ? 'X' : v === 2 ? 'O' : '';

    if (isHumanTurn && s.valid_moves.includes(i)) {
      cell.classList.add('playable');
      cell.onclick = () => submitMove(i);
    }
    boardEl.appendChild(cell);
  });

  let msg = '';
  if (!s.active) {
    msg = 'Pick a variant and side, then click "New Game".';
  } else if (s.game_over) {
    msg = s.winner === 'draw' ? 'Game over: draw.'
      : (s.winner === s.human_symbol ? 'Game over: you win!' : `Game over: ${s.variant} wins.`);
  } else if (isHumanTurn) {
    msg = 'Your move.';
  } else {
    msg = `${s.variant}'s turn...`;
  }
  statusText.textContent = msg;
  explanationEl.textContent = s.agent_move_explanation || '';
  moveLogEl.innerHTML = (s.move_log || []).map(m =>
    `<div>${m.mover === 'human' ? 'You' : s.variant} played cell ${m.move}</div>`
  ).join('');
}

loadVariants().then(() => render({ active: false, board: [0,0,0,0,0,0,0,0,0] }));
