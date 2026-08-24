"""
Live match between the two classic-tic-tac-toe agents this project has
running side by side: the Q-learning table (../web_dashboard, self-play
session) and the AlphaZero-style network (../live_alphazero_dashboard).
Same game, two completely different architectures - the Q-agent has no
generalization at all (an unvisited state is pure zero) and no search at
decision time; the network generalizes to unseen states and searches via
MCTS every move.

Deliberately READ-ONLY toward both source agents: before every game, this
process reloads each agent's LATEST checkpoint directly from the other
dashboards' own session files, in memory, without touching their live
Flask processes at all. Since both those dashboards are actively training
in the background, the match automatically reflects each agent's current
strength as the games are played, not a frozen snapshot from whenever this
was launched.

Run with: python app.py [port]
Then open http://127.0.0.1:<port>
"""

import csv
import os
import pickle
import sys
import threading
import time
from collections import defaultdict

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game_engine import TicTacToe, X, O  # noqa: E402
from neural_net import TicTacToeNet  # noqa: E402
from neural_mcts import neural_mcts_search  # noqa: E402
from q_learning import QLearningAgent  # noqa: E402

app = Flask(__name__)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5041

NET_SESSION_FILE = os.path.join(os.path.dirname(__file__), '..', 'live_alphazero_dashboard',
                                 'live_alphazero_session.pkl')
QAGENT_SESSION_FILE = os.path.join(os.path.dirname(__file__), '..', 'web_dashboard',
                                    'live_session_fresh.pkl')
MATCH_LOG_FILE = os.path.join(os.path.dirname(__file__), 'match_log.csv')

DEFAULT_CONFIG = {
    'mcts_iterations': 100,     # the network's search budget per move during the match
    'move_delay': 0.3,          # seconds paused after each move, so the board is watchable
    'games_per_run': 200,       # how many more games one "Start" click plays
}
CONFIG = dict(DEFAULT_CONFIG)

state_lock = threading.Lock()
state = {
    'running': False,
    'paused': False,
    'initialized': False,
    'game_num': 0,
    'games_target': 0,
    'board': [0] * 9,
    'current_player': None,
    'net_plays': None,       # 'X' or 'O' - alternates every game
    'stage': 'idle',
    'move_explanation': '',
    'score': {'net': 0, 'qagent': 0, 'draws': 0},
    'net_source_iteration': None,     # which iteration of training the CURRENT net checkpoint is from
    'qagent_source_games': None,      # how many games the CURRENT q-agent checkpoint has trained on
    'outcome_log': [],
    'score_history': [],   # [{game_num, net_win_rate, qagent_win_rate, draw_rate}] - rolling, for a chart
    'config': dict(CONFIG),
}

stop_flag = threading.Event()
pause_flag = threading.Event()
training_thread = None


def symbol(player):
    return {X: 'X', O: 'O'}.get(player)


def push_state(**kwargs):
    with state_lock:
        state.update(kwargs)


def load_current_net():
    with open(NET_SESSION_FILE, 'rb') as f:
        data = pickle.load(f)
    net = TicTacToeNet(hidden_size=32, seed=42)
    for p in net.params:
        setattr(net, p, data['net_params'][p])
    net.t, net.m, net.v = data['net_t'], data['net_m'], data['net_v']
    return net, data.get('iterations_run_total', 0)


def load_current_qagent():
    with open(QAGENT_SESSION_FILE, 'rb') as f:
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
    return agent, data.get('games_run_total', 0)


def _append_match_log_row(row):
    is_new = not os.path.exists(MATCH_LOG_FILE)
    with open(MATCH_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['game_num', 'net_plays', 'outcome', 'net_source_iteration', 'qagent_source_games'])
        writer.writerow(row)


def _load_match_log():
    """Restore score/history/outcome log from disk on startup, so restarting
    this dashboard doesn't wipe out the running tally - same 'survives a
    restart' standard as every other dashboard in this project."""
    if not os.path.exists(MATCH_LOG_FILE):
        return
    with open(MATCH_LOG_FILE) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    score = {'net': 0, 'qagent': 0, 'draws': 0}
    history = []
    outcome_log = []
    for row in rows:
        game_num = int(row['game_num'])
        outcome = row['outcome']
        if outcome == 'draw':
            score['draws'] += 1
        elif outcome == 'net_win':
            score['net'] += 1
        else:
            score['qagent'] += 1
        total = score['net'] + score['qagent'] + score['draws']
        history.append({
            'game_num': game_num,
            'net_win_rate': score['net'] / total * 100,
            'qagent_win_rate': score['qagent'] / total * 100,
            'draw_rate': score['draws'] / total * 100,
        })
        outcome_log.append({'game_num': game_num, 'net_plays': row['net_plays'], 'outcome': outcome})

    last_row = rows[-1]
    push_state(
        initialized=True, stage='restored', game_num=int(last_row['game_num']),
        score=score, score_history=history[-1000:], outcome_log=list(reversed(outcome_log))[:100],
        net_source_iteration=int(last_row['net_source_iteration']) if last_row['net_source_iteration'] else None,
        qagent_source_games=int(last_row['qagent_source_games']) if last_row['qagent_source_games'] else None,
    )


def wait_while_paused():
    if pause_flag.is_set():
        push_state(stage='paused', paused=True)
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


def play_one_match_game(net, agent, net_plays, game_num, games_target):
    game = TicTacToe()
    push_state(stage='playing', board=game.board.copy(), current_player=symbol(game.current_player),
               net_plays=symbol(net_plays), game_num=game_num, games_target=games_target)

    while not game.game_over:
        wait_while_paused()
        if stop_flag.is_set():
            return None

        if game.current_player == net_plays:
            move, _ = neural_mcts_search(game, net, iterations=CONFIG['mcts_iterations'], add_noise=False)
            explanation = f"Network (greedy MCTS, {CONFIG['mcts_iterations']} sims): cell {move}"
        else:
            move = agent.choose_action(game, greedy=True)
            explanation = f"Q-agent (greedy, best known move): cell {move}"

        game.make_move(move)
        push_state(board=game.board.copy(),
                   current_player=symbol(game.current_player) if not game.game_over else None,
                   move_explanation=explanation)
        interruptible_sleep(CONFIG['move_delay'])

    return game.winner


def match_loop(games_this_run):
    global_game_start = state['game_num']
    push_state(running=True, games_target=global_game_start + games_this_run)

    for i in range(games_this_run):
        wait_while_paused()
        if stop_flag.is_set():
            break

        net, net_iter = load_current_net()
        agent, agent_games = load_current_qagent()

        game_num = global_game_start + i + 1
        net_plays = X if game_num % 2 == 1 else O  # alternate who starts each game

        winner = play_one_match_game(net, agent, net_plays, game_num, global_game_start + games_this_run)
        if winner is None:  # stopped mid-game
            break

        with state_lock:
            score = dict(state['score'])
        if winner == -1:
            score['draws'] += 1
            outcome = 'draw'
        elif winner == net_plays:
            score['net'] += 1
            outcome = 'net_win'
        else:
            score['qagent'] += 1
            outcome = 'qagent_win'

        total = score['net'] + score['qagent'] + score['draws']
        with state_lock:
            history = state['score_history'] + [{
                'game_num': game_num,
                'net_win_rate': score['net'] / total * 100,
                'qagent_win_rate': score['qagent'] / total * 100,
                'draw_rate': score['draws'] / total * 100,
            }]
            state['score_history'] = history[-1000:]
            log = [{'game_num': game_num, 'net_plays': symbol(net_plays), 'outcome': outcome}] + state['outcome_log']
            state['outcome_log'] = log[:100]

        _append_match_log_row([game_num, symbol(net_plays), outcome, net_iter, agent_games])
        push_state(score=score, game_num=game_num,
                   net_source_iteration=net_iter, qagent_source_games=agent_games)

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


def _apply_config_from_request():
    body = request.get_json(silent=True) or {}
    for key in DEFAULT_CONFIG:
        if key in body:
            CONFIG[key] = type(DEFAULT_CONFIG[key])(body[key])
    push_state(config=dict(CONFIG))


@app.route('/api/start', methods=['POST'])
def api_start():
    global training_thread
    with state_lock:
        if state['running']:
            return jsonify({'ok': False, 'error': 'already running'}), 400
        if not os.path.exists(NET_SESSION_FILE):
            return jsonify({'ok': False, 'error': f'network session not found yet: {NET_SESSION_FILE}'}), 400
        if not os.path.exists(QAGENT_SESSION_FILE):
            return jsonify({'ok': False, 'error': f'Q-agent session not found yet: {QAGENT_SESSION_FILE}'}), 400

    _apply_config_from_request()
    push_state(initialized=True)
    stop_flag.clear()
    pause_flag.clear()
    training_thread = threading.Thread(target=match_loop, args=(CONFIG['games_per_run'],), daemon=True)
    training_thread.start()
    return jsonify({'ok': True})


@app.route('/api/update_params', methods=['POST'])
def api_update_params():
    _apply_config_from_request()
    return jsonify({'ok': True, 'config': CONFIG})


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


_load_match_log()

if __name__ == '__main__':
    app.run(debug=False, port=PORT)
