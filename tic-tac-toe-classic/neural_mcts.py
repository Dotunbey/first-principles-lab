"""
MCTS guided by the neural network (neural_net.py), AlphaZero-style - same
algorithm as ../sliding-tic-tac-toe/neural_mcts.py, unchanged except for
importing this project's smaller POLICY_SIZE (9, vs the sliding game's 90).

  - Selection uses PUCT (Q + c_puct * prior * sqrt(N_parent) / (1 + N)) instead
    of plain UCB1 - the policy network's prior P(s,a) biases which branches
    get explored, instead of every move starting out equally interesting.
  - A leaf is expanded ALL AT ONCE (every legal child gets a prior from one
    network call), not one untried move at a time.
  - No rollout at all: the value network's output at the leaf IS the estimate.
  - Values flow up the tree with a sign flip at every level (negamax
    convention, same one used throughout this whole project).
"""

import math

import numpy as np

from neural_net import encode_action, POLICY_SIZE

C_PUCT = 1.5  # AlphaZero's typical range is 1-5; controls explore/exploit balance


class NeuralNode:
    def __init__(self, game_state, parent=None, move=None, prior=0.0, player_just_moved=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.player_just_moved = player_just_moved  # who moved TO create this node - see _terminal_value
        self.children = {}  # move -> NeuralNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def value(self):
        """Average value from THIS node's own mover's perspective."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def puct_score(self, parent_visit_count, c_puct=C_PUCT):
        exploitation = -self.value()
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return exploitation + exploration

    def select_child(self, c_puct=C_PUCT):
        return max(self.children.items(), key=lambda item: item[1].puct_score(self.visit_count, c_puct))


def _terminal_value(node):
    """Exact value at a real game-over node, from 'whoever would move next'
    perspective. Uses player_just_moved (captured before the move was
    applied) rather than game_state.current_player - see the sliding
    project's neural_mcts.py / 04_bugs_lessons_and_limitations.md for why
    trusting current_player at a terminal node is a real, previously-hit
    bug (game_engine.py's move methods don't flip current_player on a win)."""
    game_state = node.game_state
    mover = node.player_just_moved
    if game_state.winner == -1:
        value_to_mover = 0.0
    elif game_state.winner == mover:
        value_to_mover = 1.0
    else:
        value_to_mover = -1.0
    return -value_to_mover


def _net_for(net, player):
    """net may be a single network (shared) OR a dict {X: net_x, O: net_o} -
    picks whichever is responsible for this position's mover."""
    if isinstance(net, dict):
        return net[player]
    return net


def _expand(node, net):
    game = node.game_state
    active_net = _net_for(net, game.current_player)
    value, policy_probs = active_net.predict(game.board, game.phase, game.current_player)

    legal_moves = game.get_valid_moves()
    legal_indices = np.array([encode_action(m) for m in legal_moves], dtype=int)
    legal_probs = policy_probs[legal_indices]
    total = legal_probs.sum()
    legal_probs = legal_probs / total if total > 1e-9 else np.full(len(legal_moves), 1.0 / len(legal_moves))

    mover = game.current_player  # capture BEFORE the move - see _terminal_value
    for move, prior in zip(legal_moves, legal_probs):
        child_state = game.copy()
        child_state.make_move(move)
        node.children[move] = NeuralNode(child_state, parent=node, move=move, prior=float(prior),
                                          player_just_moved=mover)

    node.is_expanded = True
    return value


def _backup(node, value):
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        value = -value  # flip: one level up is the opponent's perspective
        node = node.parent


def _add_dirichlet_noise(root, alpha=0.3, frac=0.25):
    """Standard AlphaZero self-play trick: mix random noise into the root's
    priors so self-play doesn't always search the exact same moves."""
    if not root.children:
        return
    noise = np.random.dirichlet([alpha] * len(root.children))
    for (move, child), n in zip(root.children.items(), noise):
        child.prior = child.prior * (1 - frac) + n * frac


def neural_mcts_search(root_state, net, iterations=200, c_puct=C_PUCT, add_noise=False, noise_frac=0.25):
    """Run guided MCTS from root_state. Returns (best_move, root)."""
    root = NeuralNode(root_state.copy())
    _expand(root, net)
    if add_noise:
        _add_dirichlet_noise(root, frac=noise_frac)

    for _ in range(iterations):
        node = root
        while node.is_expanded and not node.game_state.game_over and node.children:
            move, node = node.select_child(c_puct)

        if node.game_state.game_over:
            value = _terminal_value(node)
        else:
            value = _expand(node, net)

        _backup(node, value)

    best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
    return best_move, root


def visit_count_policy(root, temperature=1.0):
    """Root's children visit counts -> a probability distribution over the
    POLICY_SIZE-slot policy space, suitable as a training target."""
    policy = np.zeros(POLICY_SIZE)
    for move, child in root.children.items():
        policy[encode_action(move)] = child.visit_count
    if temperature != 1.0:
        policy = policy ** (1.0 / temperature)
    total = policy.sum()
    return policy / total if total > 0 else policy
