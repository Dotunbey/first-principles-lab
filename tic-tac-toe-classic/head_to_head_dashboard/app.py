"""
Live head-to-head scoreboard between the two classic tic-tac-toe Q-learning
agents this project has been comparing:
  - "5013 / current"  - the one warm-started from the sliding game (later
    proven to have started WORSE than fresh - see ../research/
    warm_start_experiment.md), now also learning from human games via
    ../web_dashboard/app.py's play-vs-human feature.
  - "5014 / fresh"    - the one trained from scratch, still self-playing.

This dashboard does NOT hold its own copy of either agent - it reloads both
Q-tables fresh from their live session files (../web_dashboard/live_session.pkl
and live_session_fresh.pkl) before every single game, so every match reflects
whatever either agent has ACTUALLY learned by that moment, including human-play
updates and ongoing background self-play. It never writes to either agent's
file - purely a read-only, live-scoring spectator.

Run with: python app.py [port]
Then open http://127.0.0.1:<port>
"""

import os
import pickle
import queue
import sys
import threading
import time
from collections import defaultdict

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game_engine import TicTacToe, X, O  # noqa: E402
from q_learning import QLearningAgent  # noqa: E402

app = Flask(__name__)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5016
SESSION_FILE = os.path.join(os.path.dirname(__file__), 'head_to_head_session.pkl')

AGENT_A_PATH = os.path.join(os.path.dirname(__file__), '..', 'web_dashboard', 'live_session.pkl')
AGENT_B_PATH = os.path.join(os.path.dirname(__file__), '..', 'web_dashboard', 'live_session_fresh.pkl')
AGENT_A_LABEL = '5013 / current (warm-started, +human play)'
AGENT_B_LABEL = '5014 / fresh (self-play)'

DEFAULT_CONFIG = {
    'move_delay': 0.3,
    'games_per_run': 20,
    'rolling_window': 20,  # win-rate chart is computed over this many most-recent games
}
CONFIG = dict(DEFAULT_CONFIG)

state_lock = threading.Lock()
state = {
    'running': False,
    'paused': False,
    'game_count': 0,
    'games_target': 0,
    'stage': 'idle',
    'board': [0] * 9,
    'current_player': None,
    'move_explanation': '',
    'a_label': AGENT_A_LABEL,
    'b_label': AGENT_B_LABEL,
    'a_side': None,  # 'X' or 'O' for the CURRENT game
    'scoreboard': {'a_wins': 0, 'b_wins': 0, 'draws': 0},
    'outcome_log': [],
    'history': [],  # [{game_count, a_win_rate, b_win_rate, draw_rate}] over the rolling window
    'a_table_size': 0,
    'b_table_size': 0,
    'config': dict(CONFIG),
}

game_count = 0
scoreboard = {'a_wins': 0, 'b_wins': 0, 'draws': 0}
recent_outcomes = []  # rolling list of 'a', 'b', or 'draw', capped at a generous length
stop_flag = threading.Event()
pause_flag = threading.Event()
training_thread = None

_save_queue = queue.Queue(maxsize=3)


def _saver_thread_loop():
    while True:
        data = _save_queue.get()
        try:
            tmp_path = SESSION_FILE + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f)
            os.replace(tmp_path, SESSION_FILE)
        except OSError:
            pass


threading.Thread(target=_saver_thread_loop, daemon=True).start()


def save_own_session():
    with state_lock:
        outcome_log = list(state['outcome_log'])
        history = list(state['history'])
    data = {
        'game_count': game_count,
        'scoreboard': dict(scoreboard),
        'recent_outcomes': list(recent_outcomes),
        'outcome_log': outcome_log,
        'history': history,
        'config': dict(CONFIG),
    }
    try:
        _save_queue.put_nowait(data)
    except queue.Full:
        pass


def load_own_session():
    global game_count, scoreboard, recent_outcomes
    if not os.path.exists(SESSION_FILE):
        return
    with open(SESSION_FILE, 'rb') as f:
        data = pickle.load(f)
    game_count = data['game_count']
    scoreboard = data['scoreboard']
    recent_outcomes = data['recent_outcomes']
    CONFIG.update(data.get('config', {}))
    push_state(game_count=game_count, scoreboard=dict(scoreboard),
               outcome_log=data['outcome_log'], history=data['history'], config=dict(CONFIG))


def push_state(**kwargs):
    with state_lock:
        state.update(kwargs)


def load_agent_snapshot(path):
    """Reconstruct a QLearningAgent from whatever the live dashboard has
    most recently autosaved - read-only, never writes back."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    agent = QLearningAgent(
        learning_rate=data['learning_rate'],
        discount_factor=data['discount_factor'],
        exploration_rate=data['exploration_rate'],
    )
    agent.q_table = defaultdict(lambda: defaultdict(float))
    for state_hash, actions in data['q_table'].items():
        for action, value in actions.items():
            agent.q_table[state_hash][action] = value
    return agent


def symbol(player):
    return {X: 'X', O: 'O'}.get(player)


def wait_while_paused():
    if pause_flag.is_set():
        push_state(stage='paused', paused=True)
        save_own_session()
    while pause_flag.is_set() and not stop_flag.is_set():
        time.sleep(0.1)
    if not stop_flag.is_set():
        push_state(paused=False)


def interruptible_sleep(duration):
    step = 0.05
    elapsed = 0.0
    while elapsed < duration:
        if stop_flag.is_set():
            return
        wait_while_paused()
        time.sleep(step)
        elapsed += step


def play_one_game(agent_a, agent_b, a_plays_x):
    """Both agents play fully greedy (best-known move, no exploration) -
    their strongest available play right now, exactly like the decisive
    tests used throughout this project."""
    game = TicTacToe()
    a_symbol = X if a_plays_x else O
    b_symbol = O if a_plays_x else X

    push_state(stage='playing', board=game.board.copy(),
               current_player=symbol(game.current_player), a_side=symbol(a_symbol))

    while not game.game_over:
        wait_while_paused()
        if stop_flag.is_set():
            return None

        mover_agent = agent_a if game.current_player == a_symbol else agent_b
        mover_label = 'A' if game.current_player == a_symbol else 'B'
        move = mover_agent.choose_action(game, greedy=True)
        game.make_move(move)

        push_state(board=game.board.copy(),
                   current_player=symbol(game.current_player) if not game.game_over else None,
                   move_explanation=f"{mover_label} ({state['a_label'] if mover_label == 'A' else state['b_label']}) played {move}")
        interruptible_sleep(CONFIG['move_delay'])

    if game.winner == -1:
        return 'draw'
    return 'a' if game.winner == a_symbol else 'b'


def training_loop(games_this_run):
    global game_count

    push_state(running=True, games_target=game_count + games_this_run)

    for _ in range(games_this_run):
        wait_while_paused()
        if stop_flag.is_set():
            break

        # Reload both agents fresh from disk before every game, so the match
        # always reflects current knowledge, not a frozen snapshot from
        # whenever this dashboard started.
        try:
            agent_a = load_agent_snapshot(AGENT_A_PATH)
            agent_b = load_agent_snapshot(AGENT_B_PATH)
        except (FileNotFoundError, EOFError):
            interruptible_sleep(1.0)  # the other dashboard may be mid-save - retry shortly
            continue

        push_state(a_table_size=len(agent_a.q_table), b_table_size=len(agent_b.q_table))

        game_count += 1
        a_plays_x = (game_count % 2 == 1)  # alternate who's X each game, for fairness
        outcome = play_one_game(agent_a, agent_b, a_plays_x)
        if outcome is None:  # stopped mid-game
            break

        if outcome == 'a':
            scoreboard['a_wins'] += 1
        elif outcome == 'b':
            scoreboard['b_wins'] += 1
        else:
            scoreboard['draws'] += 1
        recent_outcomes.append(outcome)
        del recent_outcomes[:-500]  # cap memory, keep plenty for the rolling window

        window = CONFIG['rolling_window']
        recent = recent_outcomes[-window:]
        a_rate = recent.count('a') / len(recent) * 100
        b_rate = recent.count('b') / len(recent) * 100
        draw_rate = recent.count('draw') / len(recent) * 100

        with state_lock:
            log = [{'game_count': game_count, 'outcome': outcome, 'a_side': symbol(X if a_plays_x else O)}] \
                + state['outcome_log']
            state['outcome_log'] = log[:100]
            history = state['history'] + [{
                'game_count': game_count, 'a_win_rate': a_rate, 'b_win_rate': b_rate, 'draw_rate': draw_rate,
            }]
            state['history'] = history[-500:]

        push_state(game_count=game_count, scoreboard=dict(scoreboard))
        save_own_session()

    save_own_session()
    push_state(running=False, paused=False, stage='stopped' if stop_flag.is_set() else 'idle')


@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html')) as f:
        return f.read()


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), filename)


@app.route('/api/state')
def api_state():
    with state_lock:
        return jsonify(state)


@app.route('/api/start', methods=['POST'])
def api_start():
    global training_thread
    with state_lock:
        if state['running']:
            return jsonify({'ok': False, 'error': 'already running'}), 400

    body = request.get_json(silent=True) or {}
    for key in DEFAULT_CONFIG:
        if key in body:
            CONFIG[key] = type(DEFAULT_CONFIG[key])(body[key])
    push_state(config=dict(CONFIG))

    stop_flag.clear()
    pause_flag.clear()
    training_thread = threading.Thread(target=training_loop, args=(CONFIG['games_per_run'],), daemon=True)
    training_thread.start()
    return jsonify({'ok': True})


@app.route('/api/pause', methods=['POST'])
def api_pause():
    pause_flag.set()
    return jsonify({'ok': True})


@app.route('/api/resume', methods=['POST'])
def api_resume():
    pause_flag.clear()
    return jsonify({'ok': True})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    stop_flag.set()
    pause_flag.clear()
    return jsonify({'ok': True})


load_own_session()

if __name__ == '__main__':
    print(f"Session file: {SESSION_FILE}")
    print(f"Agent A ({AGENT_A_LABEL}): {AGENT_A_PATH}")
    print(f"Agent B ({AGENT_B_LABEL}): {AGENT_B_PATH}")
    app.run(debug=False, port=PORT)
