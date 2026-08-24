const boardEl = document.getElementById('board');
const explanationEl = document.getElementById('explanation');
const moveLogEl = document.getElementById('moveLog');
const statusText = document.getElementById('statusText');
const variantSelect = document.getElementById('variantSelect');
const humanSymbolSelect = document.getElementById('humanSymbol');
const newGameBtn = document.getElementById('newGameBtn');

let selectedFrom = null;  // mid-selection during the sliding phase

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
  selectedFrom = null;
  postJSON('/api/new', { variant: variantSelect.value, human_symbol: humanSymbolSelect.value }).then(r => {
    if (!r.ok) { statusText.textContent = r.error; return; }
    render(r.state);
  });
};

function movesEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function submitMove(move) {
  selectedFrom = null;
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

    if (isHumanTurn && s.phase === 'placement' && s.valid_moves.includes(i)) {
      cell.classList.add('playable');
      cell.onclick = () => submitMove(i);
    } else if (isHumanTurn && s.phase === 'sliding') {
      if (selectedFrom === null) {
        const canPickUp = s.valid_moves.some(m => m[0] === i);
        if (canPickUp) {
          cell.classList.add('playable');
          cell.onclick = () => { selectedFrom = i; render(s); };
        }
      } else if (selectedFrom === i) {
        cell.classList.add('selected');
        cell.onclick = () => { selectedFrom = null; render(s); };
      } else {
        const isDestination = s.valid_moves.some(m => movesEqual(m, [selectedFrom, i]));
        if (isDestination) {
          cell.classList.add('playable');
          cell.onclick = () => submitMove([selectedFrom, i]);
        }
      }
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
    msg = s.phase === 'sliding' && selectedFrom !== null
      ? 'Pick a destination for the selected piece.'
      : 'Your move.';
  } else {
    msg = `${s.variant}'s turn...`;
  }
  statusText.textContent = msg;
  explanationEl.textContent = s.agent_move_explanation || '';
  moveLogEl.innerHTML = (s.move_log || []).map(m =>
    `<div>${m.mover === 'human' ? 'You' : s.variant} played ${JSON.stringify(m.move)}</div>`
  ).join('');
}

loadVariants().then(() => render({ active: false, board: [0,0,0,0,0,0,0,0,0] }));
