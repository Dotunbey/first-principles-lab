"""
Play against a trained AlphaZero-style network for classic tic-tac-toe.

Simpler than the sliding project's version (../../sliding-tic-tac-toe/play_vs_network/):
no sliding phase, so every move is just a single cell click - no two-step
piece-then-destination selection needed. Currently only one variant exists
('base' - see ../alphazero_selfplay.py) since classic tic-tac-toe has no known
X/O asymmetry to fight, so none of the sliding project's fix-attempt variants
apply here; the dropdown is written to support more if that ever changes.

The network's checkpoint is reloaded from disk each time you click "New
Game" against it, so playing against 'base' always uses its current
training progress - no need to restart this server as training continues.

Run with: python app.py [port]
Then open http://127.0.0.1:<port>
"""

import os
import sys

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game_engine import TicTacToe, X, O  # noqa: E402
from neural_net import TicTacToeNet  # noqa: E402
from neural_mcts import neural_mcts_search  # noqa: E402

app = Flask(__name__)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5021
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..')

MCTS_ITERATIONS = 150

VARIANT_FILES = {
    'base': 'selfplay_trained_net_base.pkl',
}

game = None
human_symbol = None
current_variant = None
current_net = None
move_log = []


def symbol(player):
    return {X: 'X', O: 'O'}.get(player)


def parse_symbol(s):
    return X if s == 'X' else O


def load_variant(name):
    net = TicTacToeNet(hidden_size=32, seed=42)
    net.load(os.path.join(MODELS_DIR, VARIANT_FILES[name]))
    return net


def build_state(agent_explanation=''):
    if game is None:
        return {'active': False, 'board': [0] * 9, 'human_symbol': None, 'variant': None,
                'current_player': None, 'valid_moves': [], 'game_over': False, 'winner': None,
                'agent_move_explanation': '', 'move_log': []}
    winner = None
    if game.game_over:
        winner = 'draw' if game.winner == -1 else symbol(game.winner)
    return {
        'active': True,
        'board': game.board.copy(),
        'human_symbol': symbol(human_symbol),
        'variant': current_variant,
        'current_player': symbol(game.current_player) if not game.game_over else None,
        'valid_moves': [] if game.game_over else game.get_valid_moves(),
        'game_over': game.game_over,
        'winner': winner,
        'agent_move_explanation': agent_explanation,
        'move_log': move_log[:50],
    }


def agent_take_turn():
    move, _ = neural_mcts_search(game, current_net, iterations=MCTS_ITERATIONS, add_noise=False)
    game.make_move(move)
    move_log.insert(0, {'mover': 'agent', 'move': move})
    return f"{current_variant} (greedy MCTS, {MCTS_ITERATIONS} sims): cell {move}"


@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html')) as f:
        return f.read()


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), filename)


@app.route('/api/variants')
def api_variants():
    available = [name for name, path in VARIANT_FILES.items()
                 if os.path.exists(os.path.join(MODELS_DIR, path))]
    return jsonify({'variants': available})


@app.route('/api/state')
def api_state():
    return jsonify(build_state())


@app.route('/api/new', methods=['POST'])
def api_new():
    global game, human_symbol, current_variant, current_net, move_log
    body = request.get_json(silent=True) or {}
    variant = body.get('variant')
    if variant not in VARIANT_FILES:
        return jsonify({'ok': False, 'error': f'unknown variant {variant}'}), 400

    current_variant = variant
    current_net = load_variant(variant)  # reload from disk - picks up latest training progress
    human_symbol = parse_symbol(body.get('human_symbol', 'X'))
    game = TicTacToe()
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
    move = body.get('move')
    if move not in game.get_valid_moves():
        return jsonify({'ok': False, 'error': 'illegal move'}), 400

    game.make_move(move)
    move_log.insert(0, {'mover': 'human', 'move': move})

    explanation = ''
    if not game.game_over:
        explanation = agent_take_turn()
    return jsonify({'ok': True, 'state': build_state(explanation)})


if __name__ == '__main__':
    app.run(debug=False, port=PORT)
