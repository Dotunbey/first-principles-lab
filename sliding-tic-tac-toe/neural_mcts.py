"""
MCTS guided by the neural network (neural_net.py), AlphaZero-style - the
"Stage 2" gap-closing piece: replaces plain-Python mcts.py's UCB1 selection
and rollout-based leaf evaluation with a policy-network prior and an instant
value-network evaluation.

Differences from mcts.py, precisely:
  - Selection uses PUCT (Q + c_puct * prior * sqrt(N_parent) / (1 + N)) instead
    of plain UCB1 - the policy network's prior P(s,a) biases which branches
    get explored, instead of every move starting out equally interesting.
  - A leaf is expanded ALL AT ONCE (every legal child gets a prior from one
    network call), not one untried move at a time.
  - No rollout at all: the value network's output at the leaf IS the estimate,
    used directly - no simulated game is ever played out.
  - Values flow up the tree with a sign flip at every level (negamax
    convention, same one used throughout this whole project - the solver,
    plain MCTS's backprop, and TD(0)'s bootstrap all do the same thing).
"""

import math

import numpy as np

from neural_net import encode_action

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
        # Flip sign: a child's value is from the CHILD's mover's perspective,
        # but the parent needs "how good is this for MY mover" - same negamax
        # flip used everywhere else in this project.
        exploitation = -self.value()
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return exploitation + exploration

    def select_child(self, c_puct=C_PUCT):
        return max(self.children.items(), key=lambda item: item[1].puct_score(self.visit_count, c_puct))


def _terminal_value(node):
    """Exact value at a real game-over node, converted to the same 'value to
    whoever would move next' convention every non-terminal node uses.

    IMPORTANT: game_engine.py's move methods return early on a win, BEFORE
    flipping current_player - so after a win, current_player is still the
    WINNER, not 'whoever's turn would be next.' Relying on that field here
    would silently invert every win/loss. Instead, use player_just_moved
    (captured before the move was applied, in _expand below) to compute
    the value relative to the mover, then negate to match the "next mover"
    convention - the same defensive pattern perfect_solver.py already uses
    for exactly this reason."""
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
    """net may be a single network (shared, business as usual) OR a dict
    {X: net_x, O: net_o} (the two-specialist-networks experiment) - picks
    whichever one is actually responsible for this position's mover, so
    every other part of the search stays completely unaware of the
    difference."""
    if isinstance(net, dict):
        return net[player]
    return net


def _expand(node, net):
    """Ask the network once for this leaf's value and a prior over every
    legal move, then create all children at once. Returns the leaf's own
    value (from its own mover's perspective) for backup."""
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
    """Standard AlphaZero self-play trick: mix some random noise into the
    root's priors so self-play doesn't always search the exact same
    'obviously good' moves - without this, exploration during self-play
    relies entirely on the network already being right, which is circular
    early in training."""
    if not root.children:
        return
    noise = np.random.dirichlet([alpha] * len(root.children))
    for (move, child), n in zip(root.children.items(), noise):
        child.prior = child.prior * (1 - frac) + n * frac


def neural_mcts_search(root_state, net, iterations=200, c_puct=C_PUCT, add_noise=False, noise_frac=0.25):
    """Run guided MCTS from root_state. Returns (best_move, root) - root is
    returned too so callers (self-play) can read off the visit-count
    distribution as a training target for the policy head."""
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
    full 90-slot policy space, suitable as a training target. temperature<1
    sharpens toward the most-visited move; temperature=1 uses raw counts."""
    policy = np.zeros(90)
    for move, child in root.children.items():
        policy[encode_action(move)] = child.visit_count
    if temperature != 1.0:
        policy = policy ** (1.0 / temperature)
    total = policy.sum()
    return policy / total if total > 0 else policy
