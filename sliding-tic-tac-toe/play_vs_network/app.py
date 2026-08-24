"""
Play against any trained AlphaZero-style network variant from this project,
yourself, in the browser.

All 9 variants (research/VARIANT_REGISTRY.md) are loaded once at startup and
kept in memory; pick one from the dropdown, choose your side, and play. The
network always moves greedily (neural_mcts_search with add_noise=False,
150 MCTS iterations) - its strongest available play, no exploration
randomness, matching how every decisive test and the round-robin tournament
(../tournament.py) evaluated these networks.

Run with: python app.py [port]
Then open http://127.0.0.1:<port>
"""

import os
import sys

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game_engine import SlidingTicTacToe, X, O  # noqa: E402
from neural_net import TicTacToeNet  # noqa: E402
from neural_mcts import neural_mcts_search  # noqa: E402

app = Flask(__name__)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5020
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..')

MCTS_ITERATIONS = 150


def _load(path, seed=42):
    net = TicTacToeNet(hidden_size=64, seed=seed)
    net.load(os.path.join(MODELS_DIR, path))
    return net


def _build_variants():
    """Returns {variant_name: {X: net, O: net}} - single-net variants use the
    same net object for both sides, dual-net variants use their own trained
    net per role."""
    variants = {}

    single_net = {
        'unbalanced': 'selfplay_trained_net.pkl',
        'balanced': 'selfplay_trained_net_balanced.pkl',
        'teacher': 'selfplay_trained_net_teacher.pkl',
        'exploration': 'selfplay_trained_net_exploration.pkl',
        'o_compute': 'selfplay_trained_net_o_compute.pkl',
        'imitate': 'selfplay_trained_net_imitate.pkl',
    }
    for name, path in single_net.items():
        net = _load(path)
        variants[name] = {X: net, O: net}

    dual_net = {
        'twonet': ('selfplay_trained_net_twonet_X.pkl', 'selfplay_trained_net_twonet_O.pkl'),
        'catchup': ('selfplay_trained_net_catchup_X.pkl', 'selfplay_trained_net_catchup_O.pkl'),
        'catchup_wr': ('selfplay_trained_net_catchup_wr_X.pkl', 'selfplay_trained_net_catchup_wr_O.pkl'),
    }
    for name, (x_path, o_path) in dual_net.items():
        variants[name] = {X: _load(x_path, seed=42), O: _load(o_path, seed=43)}

    return variants


print("Loading all network variants (once, at startup)...", flush=True)
VARIANTS = _build_variants()
print(f"Loaded: {list(VARIANTS.keys())}", flush=True)

# --- Ordered so the strongest-by-tournament-record variants show up first
# in the dropdown (research/VARIANT_REGISTRY.md tournament results) -------
VARIANT_ORDER = ['o_compute', 'twonet', 'unbalanced', 'exploration', 'catchup',
                  'balanced', 'catchup_wr', 'teacher', 'imitate']

game = None
human_symbol = None
current_variant = None
move_log = []


def symbol(player):
    return {X: 'X', O: 'O'}.get(player)


def parse_symbol(s):
    return X if s == 'X' else O


def valid_moves_json():
    if game is None or game.game_over:
        return []
    moves = game.get_valid_moves()
    if game.phase == 'placement':
        return moves
    return [list(m) for m in moves]


def build_state(agent_explanation=''):
    if game is None:
        return {'active': False, 'board': [0] * 9, 'phase': None, 'human_symbol': None,
                'variant': None, 'current_player': None, 'valid_moves': [], 'game_over': False,
                'winner': None, 'agent_move_explanation': '', 'move_log': []}
    winner = None
    if game.game_over:
        winner = 'draw' if game.winner == -1 else symbol(game.winner)
    return {
        'active': True,
        'board': game.board.copy(),
        'phase': game.phase,
        'human_symbol': symbol(human_symbol),
        'variant': current_variant,
        'current_player': symbol(game.current_player) if not game.game_over else None,
        'valid_moves': valid_moves_json(),
        'game_over': game.game_over,
        'winner': winner,
        'agent_move_explanation': agent_explanation,
        'move_log': move_log[:50],
    }


def agent_take_turn():
    agent_side = O if human_symbol == X else X
    net = VARIANTS[current_variant][agent_side]
    move, _ = neural_mcts_search(game, net, iterations=MCTS_ITERATIONS, add_noise=False)
    game.make_move(move)
    move_log.insert(0, {'mover': 'agent', 'move': list(move) if isinstance(move, tuple) else move})
    return f"{current_variant} (greedy MCTS, {MCTS_ITERATIONS} sims): {move}"


@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html')) as f:
        return f.read()


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), filename)


@app.route('/api/variants')
def api_variants():
    return jsonify({'variants': VARIANT_ORDER})


@app.route('/api/state')
def api_state():
    return jsonify(build_state())


@app.route('/api/new', methods=['POST'])
def api_new():
    global game, human_symbol, current_variant, move_log
    body = request.get_json(silent=True) or {}
    variant = body.get('variant')
    if variant not in VARIANTS:
        return jsonify({'ok': False, 'error': f'unknown variant {variant}'}), 400

    current_variant = variant
    human_symbol = parse_symbol(body.get('human_symbol', 'X'))
    game = SlidingTicTacToe()
    move_log = []

    explanation = ''
    if game.current_player != human_symbol:
        explanation = agent_take_turn()
    return jsonify({'ok': True, 'state': build_state(explanation)})


@app.route('/api/move', methods=['POST'])
def api_move():
    if game is None:
        return jsonify({'ok': False, 'error': 'no game in progress - click "New Game" first'}), 400
    if game.game_over:
        return jsonify({'ok': False, 'error': 'game is over - start a new one'}), 400
    if game.current_player != human_symbol:
        return jsonify({'ok': False, 'error': 'not your turn'}), 400

    body = request.get_json(silent=True) or {}
    raw_move = body.get('move')
    move = tuple(raw_move) if isinstance(raw_move, list) else raw_move
    if move not in game.get_valid_moves():
        return jsonify({'ok': False, 'error': 'illegal move'}), 400

    game.make_move(move)
    move_log.insert(0, {'mover': 'human', 'move': list(move) if isinstance(move, tuple) else move})

    explanation = ''
    if not game.game_over:
        explanation = agent_take_turn()
    return jsonify({'ok': True, 'state': build_state(explanation)})


if __name__ == '__main__':
    app.run(debug=False, port=PORT)
