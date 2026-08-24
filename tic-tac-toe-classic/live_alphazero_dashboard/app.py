"""
Live, fully manual AlphaZero-style training dashboard for classic
tic-tac-toe - same design as ../../sliding-tic-tac-toe/live_alphazero_dashboard/,
adapted for this project's simpler action space (no sliding phase, every
move is a plain cell index). Same UX as ../web_dashboard/app.py (initialize,
Start/Pause/Resume/Stop, live math on screen, adjustable hyperparameters,
autosave/resume across restarts), but for the neural-network + guided-MCTS
self-play loop instead of tabular Q-learning. This is deliberately a
SEPARATE, freshly-initialized network from the ../alphazero_selfplay.py
batch run ('base') - it never touches that checkpoint, so there's no risk
of two processes fighting over the same file.

What's shown live, per move during self-play (unlike the batch script, which
just blasts through games as fast as possible):
  - the actual board, updated after every move
  - the MCTS root's per-candidate-move stats (visit count, prior probability,
    PUCT score) - "the math" behind which move got chosen
  - once a training step runs (after a batch of self-play games): the actual
    value/policy loss numbers and the Frobenius-norm size of each weight
    matrix's update - a concrete, visible stand-in for "the network changed
    by this much just now," the same spirit as the Q-dashboard's live
    weight-change feed, adapted for a matrix of weights instead of one
    scalar Q-value.

Run with: python app.py [port] [session_filename]
Then open http://127.0.0.1:<port>
"""

import os
import pickle
import queue
import random
import sys
import threading
import time
from collections import deque

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game_engine import TicTacToe, X, O  # noqa: E402
from neural_net import TicTacToeNet, encode_action, encode_state  # noqa: E402
from neural_mcts import neural_mcts_search, visit_count_policy  # noqa: E402
from perfect_solver import negamax_value, enumerate_all_states  # noqa: E402
import numpy as np  # noqa: E402

app = Flask(__name__)

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5031
SESSION_FILENAME = sys.argv[2] if len(sys.argv) > 2 else 'live_alphazero_session.pkl'
SESSION_FILE = os.path.join(os.path.dirname(__file__), SESSION_FILENAME)

DEFAULT_CONFIG = {
    'mcts_iterations': 60,          # kept modest so a live move doesn't take too long to display
    'temperature_moves': 6,
    'noise_frac': 0.25,
    'learning_rate': 0.001,
    'batch_size': 64,
    'games_per_iteration': 6,       # self-play games before each training step - small, so an "iteration" is watchable
    'iterations_per_run': 20,       # how many training iterations one "Start" click runs
    'move_delay': 0.4,              # seconds paused after each self-play move, so the board is actually watchable
    'checkpoint_mcts_iterations': None,  # unused, kept for config-panel symmetry with the Q-dashboard
    'replay_buffer_size': 4000,
}

CONFIG = dict(DEFAULT_CONFIG)

state_lock = threading.Lock()
state = {
    'running': False,
    'paused': False,
    'initialized': False,
    'iteration': 0,
    'iterations_target': 0,
    'stage': 'idle',   # idle | initialized | selfplay | training | paused | stopped
    'board': [0] * 9,
    'phase': None,
    'current_player': None,
    'move_explanation': '',
    'mcts_stats': [],       # [{move, visits, prior, puct, value}] for the just-computed move, sorted by visits desc
    'game_in_iteration': 0,
    'games_this_iteration_target': 0,
    'buffer_size': 0,
    'outcome_log': [],
    'training_math': None,  # {value_loss, policy_loss, weight_update_norms, formula strings}
    'iteration_history': [],  # [{iteration, value_error, value_error_X, value_error_O, wins_X, wins_O, draws}]
    'initial_weights': None,
    'config': dict(CONFIG),
}

net = None
replay_buffer = None
iterations_run_total = 0
training_thread = None
stop_flag = threading.Event()
pause_flag = threading.Event()

print("Enumerating solver states for value-error benchmarking (once, at startup)...", flush=True)
_SOLVER_VALUES = enumerate_all_states()
print(f"  {len(_SOLVER_VALUES)} solved states loaded.", flush=True)

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


def save_session():
    if net is None:
        return
    with state_lock:
        outcome_log = list(state['outcome_log'])
        iteration_history = list(state['iteration_history'])
    net_params = {p: getattr(net, p) for p in net.params}
    data = {
        'net_params': net_params,
        'net_t': net.t, 'net_m': net.m, 'net_v': net.v,
        'replay_buffer': list(replay_buffer),
        'iterations_run_total': iterations_run_total,
        'config': dict(CONFIG),
        'outcome_log': outcome_log,
        'iteration_history': iteration_history,
    }
    try:
        _save_queue.put_nowait(data)
    except queue.Full:
        pass


def load_session():
    global net, replay_buffer, iterations_run_total
    if not os.path.exists(SESSION_FILE):
        return False

    with open(SESSION_FILE, 'rb') as f:
        data = pickle.load(f)

    CONFIG.update(data['config'])
    restored_net = TicTacToeNet(hidden_size=32, seed=42)
    for p in restored_net.params:
        setattr(restored_net, p, data['net_params'][p])
    restored_net.t = data['net_t']
    restored_net.m = data['net_m']
    restored_net.v = data['net_v']

    net = restored_net
    replay_buffer = deque(data['replay_buffer'], maxlen=CONFIG['replay_buffer_size'])
    iterations_run_total = data['iterations_run_total']

    push_state(
        initialized=True, stage='restored', iteration=iterations_run_total,
        buffer_size=len(replay_buffer),
        outcome_log=data['outcome_log'], iteration_history=data['iteration_history'],
        config=dict(CONFIG),
        initial_weights={
            'hidden_size': 32,
            'learning_rate': CONFIG['learning_rate'],
            'note': f'Restored from disk: {iterations_run_total} training iterations already '
                    f'completed before this server was last stopped. Nothing was lost.',
        },
    )
    return True


def symbol(player):
    return {X: 'X', O: 'O'}.get(player)


def push_state(**kwargs):
    with state_lock:
        state.update(kwargs)


def wait_while_paused():
    if pause_flag.is_set():
        push_state(stage='paused', paused=True)
        save_session()
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


def mcts_stats_from_root(root):
    """Extract the live 'math' behind the move MCTS just picked - every
    candidate's visit count, prior probability, and PUCT score - sorted by
    visit count descending (the same ordering MCTS itself uses to pick the
    final move)."""
    stats = []
    for move, child in root.children.items():
        stats.append({
            'move': list(move) if isinstance(move, tuple) else move,
            'visits': child.visit_count,
            'prior': round(child.prior, 4),
            'value': round(child.value(), 4),
            'puct': round(child.puct_score(root.visit_count), 4),
        })
    stats.sort(key=lambda s: -s['visits'])
    return stats


def play_one_selfplay_game_live(net, iteration_num, game_in_iteration, games_target):
    game = TicTacToe()
    examples = []
    move_number = 0

    push_state(stage='selfplay', board=game.board.copy(), phase=game.phase,
                current_player=symbol(game.current_player),
                game_in_iteration=game_in_iteration, games_this_iteration_target=games_target)

    while not game.game_over:
        wait_while_paused()
        if stop_flag.is_set():
            return examples, None

        move, root = neural_mcts_search(game, net, iterations=CONFIG['mcts_iterations'],
                                         add_noise=True, noise_frac=CONFIG['noise_frac'])
        mcts_stats = mcts_stats_from_root(root)

        temperature = 1.0 if move_number < CONFIG['temperature_moves'] else 0.1
        policy_target = visit_count_policy(root, temperature=temperature)

        x = encode_state(game.board, game.phase, game.current_player)
        examples.append((x, policy_target, game.current_player))

        if temperature > 0.5:
            legal_moves = list(root.children.keys())
            legal_indices = [encode_action(m) for m in legal_moves]
            probs = np.array([policy_target[i] for i in legal_indices])
            probs = probs / probs.sum()
            move = legal_moves[np.random.choice(len(legal_moves), p=probs)]
            explanation = f"Sampled (temperature={temperature}) - move {move}"
        else:
            explanation = f"Greedy (temperature={temperature}) - move {move}"

        game.make_move(move)
        move_number += 1

        push_state(board=game.board.copy(), phase=game.phase,
                    current_player=symbol(game.current_player) if not game.game_over else None,
                    move_explanation=explanation, mcts_stats=mcts_stats)
        interruptible_sleep(CONFIG['move_delay'])

    winner = game.winner
    training_examples = []
    for x, policy_target, mover in examples:
        if winner == -1:
            value_target = 0.0
        else:
            value_target = 1.0 if winner == mover else -1.0
        training_examples.append((x, policy_target, value_target, mover))
    return training_examples, winner


def balance_by_mover(examples):
    by_mover = {X: [], O: []}
    for example in examples:
        by_mover[example[3]].append(example)
    if not by_mover[X] or not by_mover[O]:
        return examples
    majority_side = X if len(by_mover[X]) >= len(by_mover[O]) else O
    minority_side = O if majority_side == X else X
    target_count = len(by_mover[majority_side])
    oversampled_minority = random.choices(by_mover[minority_side], k=target_count)
    return by_mover[majority_side] + oversampled_minority


def train_on_examples_live(net, examples):
    """Same training step as alphazero_selfplay.py's train_on_examples, but
    also snapshots each weight matrix before/after so the UI can show a real
    number for 'how much did the network just change' - the Frobenius norm
    of the update, ||W_after - W_before||, per matrix."""
    before = {p: getattr(net, p).copy() for p in net.params}

    balanced = balance_by_mover(examples)
    random.shuffle(balanced)
    total_value_loss = 0.0
    total_policy_loss = 0.0
    n_batches = 0
    batch_size = CONFIG['batch_size']

    for i in range(0, len(balanced), batch_size):
        batch = balanced[i:i + batch_size]
        x_batch = np.array([e[0] for e in batch])
        policy_batch = np.array([e[1] for e in batch])
        value_batch = np.array([[e[2]] for e in batch])
        legal_mask = (policy_batch > 0).astype(float)
        value_loss, policy_loss = net.train_step(x_batch, value_batch, policy_batch, legal_mask,
                                                   CONFIG['learning_rate'])
        total_value_loss += value_loss
        total_policy_loss += policy_loss
        n_batches += 1

    weight_update_norms = {
        p: round(float(np.linalg.norm(getattr(net, p) - before[p])), 5) for p in net.params
    }

    return (total_value_loss / n_batches, total_policy_loss / n_batches, weight_update_norms) \
        if n_batches else (None, None, weight_update_norms)


def value_error_vs_solver_by_mover(net, sample_size=200):
    keys = random.sample(list(_SOLVER_VALUES.keys()), min(sample_size, len(_SOLVER_VALUES)))
    errors = {X: [], O: []}
    for board, phase, player in keys:
        game = TicTacToe()
        game.board = list(board)
        game.phase = phase
        game.current_player = player
        true_value = negamax_value(game)
        pred_value, _ = net.predict(board, phase, player)
        errors[player].append(abs(pred_value - true_value))
    err_x = sum(errors[X]) / len(errors[X]) if errors[X] else None
    err_o = sum(errors[O]) / len(errors[O]) if errors[O] else None
    return err_x, err_o


def training_loop(iterations_this_run):
    global iterations_run_total

    push_state(running=True, iterations_target=iterations_run_total + iterations_this_run)

    for _ in range(iterations_this_run):
        wait_while_paused()
        if stop_flag.is_set():
            break

        iterations_run_total += 1
        iteration_num = iterations_run_total
        results = {'X': 0, 'O': 0, 'draw': 0}

        for game_i in range(CONFIG['games_per_iteration']):
            wait_while_paused()
            if stop_flag.is_set():
                break
            examples, winner = play_one_selfplay_game_live(
                net, iteration_num, game_i + 1, CONFIG['games_per_iteration'])
            if winner is None:  # stopped mid-game
                break
            replay_buffer.extend(examples)
            if winner == -1:
                results['draw'] += 1
            elif winner == X:
                results['X'] += 1
            else:
                results['O'] += 1

        if stop_flag.is_set():
            break

        push_state(stage='training')
        value_loss, policy_loss, weight_norms = train_on_examples_live(net, list(replay_buffer))
        err_x, err_o = value_error_vs_solver_by_mover(net)
        err = (err_x + err_o) / 2 if err_x is not None and err_o is not None else None

        training_math = {
            'value_loss': round(value_loss, 4) if value_loss is not None else None,
            'policy_loss': round(policy_loss, 4) if policy_loss is not None else None,
            'weight_update_norms': weight_norms,
            'value_formula': f"value_loss = mean((v_pred - v_target)^2) = {value_loss:.4f}" if value_loss is not None else '',
            'policy_formula': f"policy_loss = -mean(sum(target * log(pred))) = {policy_loss:.4f}" if policy_loss is not None else '',
        }

        with state_lock:
            history = state['iteration_history'] + [{
                'iteration': iteration_num,
                'value_error': err, 'value_error_X': err_x, 'value_error_O': err_o,
                'wins_X': results['X'], 'wins_O': results['O'], 'draws': results['draw'],
            }]
            state['iteration_history'] = history[-500:]
            log = [{'iteration': iteration_num, 'wins_X': results['X'], 'wins_O': results['O'],
                    'draws': results['draw']}] + state['outcome_log']
            state['outcome_log'] = log[:100]

        push_state(training_math=training_math, buffer_size=len(replay_buffer), iteration=iteration_num)
        save_session()

    save_session()
    push_state(running=False, paused=False, stage='stopped' if stop_flag.is_set() else 'idle')


def _make_fresh_net():
    return TicTacToeNet(hidden_size=32, seed=42)


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


def _apply_config_from_request():
    body = request.get_json(silent=True) or {}
    for key in DEFAULT_CONFIG:
        if key in body and DEFAULT_CONFIG[key] is not None:
            CONFIG[key] = type(DEFAULT_CONFIG[key])(body[key])
        elif key in body:
            CONFIG[key] = body[key]
    push_state(config=dict(CONFIG))


@app.route('/api/init', methods=['POST'])
def api_init():
    global net, replay_buffer, iterations_run_total
    with state_lock:
        if state['running']:
            return jsonify({'ok': False, 'error': 'stop or pause current run first'}), 400

    _apply_config_from_request()
    net = _make_fresh_net()
    replay_buffer = deque(maxlen=CONFIG['replay_buffer_size'])
    iterations_run_total = 0
    stop_flag.clear()
    pause_flag.clear()

    push_state(
        running=False, paused=False, initialized=True, stage='initialized',
        iteration=0, iterations_target=0, board=[0] * 9, phase=None,
        current_player=None, move_explanation='', mcts_stats=[],
        game_in_iteration=0, games_this_iteration_target=0, buffer_size=0,
        outcome_log=[], training_math=None, iteration_history=[],
        initial_weights={
            'hidden_size': 32,
            'learning_rate': CONFIG['learning_rate'],
            'note': 'Freshly initialized (He init) - a brand new network, no training yet.',
        },
    )
    save_session()
    return jsonify({'ok': True, 'config': CONFIG})


@app.route('/api/update_params', methods=['POST'])
def api_update_params():
    _apply_config_from_request()
    return jsonify({'ok': True, 'config': CONFIG})


@app.route('/api/start', methods=['POST'])
def api_start():
    global net, replay_buffer, training_thread
    with state_lock:
        if state['running']:
            return jsonify({'ok': False, 'error': 'already running'}), 400

    _apply_config_from_request()
    if net is None:
        net = _make_fresh_net()
        replay_buffer = deque(maxlen=CONFIG['replay_buffer_size'])
        push_state(initialized=True)

    stop_flag.clear()
    pause_flag.clear()
    training_thread = threading.Thread(target=training_loop, args=(CONFIG['iterations_per_run'],), daemon=True)
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


load_session()

if __name__ == '__main__':
    print(f"Session file: {SESSION_FILE}")
    app.run(debug=False, port=PORT)
