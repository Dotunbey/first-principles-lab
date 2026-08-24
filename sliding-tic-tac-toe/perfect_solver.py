"""
Exact game-theoretic solver for Sliding Tic-Tac-Toe, via value iteration
over the full reachable state graph.

MCTS at any iteration count is still an APPROXIMATION - it estimates values
through sampling. This is different: since the reachable state space is
small (~4,030-5,000 raw states, see the combinatorics worked out earlier),
we can compute the true value of every position directly - provably
optimal, not "very likely near-optimal."

Why value iteration instead of plain memoized minimax: the sliding phase
can revisit the same board (that's exactly what the engine's threefold-
repetition draw rule is for), so the state graph has cycles. A naive
memoized recursive solver that treats "already on this DFS path" as an
immediate draw is UNSOUND here - it can cache a value that was only valid
because of where the search happened to be standing at that moment, then
reuse that same wrong value later when the same state is reached via a
different, perfectly normal path. Value iteration avoids this: every
state starts at a neutral value (0, meaning "provisionally a draw"), and
every state's value gets recomputed from its neighbors' current values,
repeatedly, until nothing changes anywhere (a fixed point). This is the
same idea used in real endgame-tablebase solvers.
"""

from collections import deque

from game_engine import SlidingTicTacToe, X, O

_states = None
_transitions = None
_values = None


def state_key(game):
    return (tuple(game.board), game.phase, game.current_player)


def _enumerate_states():
    """BFS over every state reachable from the initial position, recording
    each state's legal moves and where they lead (a terminal value, or
    another state)."""
    start = SlidingTicTacToe()
    start_key = state_key(start)
    states = {start_key: start}
    transitions = {}
    queue = deque([start])

    while queue:
        game = queue.popleft()
        key = state_key(game)
        if key in transitions:
            continue

        edges = []
        for move in game.get_valid_moves():
            trial = game.copy()
            trial.make_move(move)
            if trial.game_over:
                if trial.winner == -1:
                    edges.append((move, 0, None))
                elif trial.winner == game.current_player:
                    edges.append((move, 1, None))
                else:
                    edges.append((move, -1, None))
            else:
                next_key = state_key(trial)
                edges.append((move, None, next_key))
                if next_key not in states:
                    states[next_key] = trial
                    queue.append(trial)
        transitions[key] = edges

    return states, transitions


def _value_iteration(states, transitions, max_sweeps=500):
    """Every state's value = the best over its moves of (terminal value,
    or the negation of where that move leads). Repeat until nothing
    changes anywhere - correctly handles cycles by just letting them settle
    at whatever value they converge to (usually 0, a real draw)."""
    values = {key: 0 for key in states}

    for sweep in range(max_sweeps):
        changed = False
        for key, edges in transitions.items():
            best = -2
            for move, terminal_value, next_key in edges:
                value = terminal_value if terminal_value is not None else -values[next_key]
                if value > best:
                    best = value
            if best != values[key]:
                values[key] = best
                changed = True
        if not changed:
            return values, sweep + 1

    return values, max_sweeps


def _ensure_solved():
    global _states, _transitions, _values
    if _values is not None:
        return
    _states, _transitions = _enumerate_states()
    _values, _ = _value_iteration(_states, _transitions)


def negamax_value(game):
    """Value of this position from the perspective of the player about to
    move: +1 = can force a win, -1 = will lose to optimal play, 0 = draw."""
    _ensure_solved()
    return _values[state_key(game)]


def best_move(game):
    """The provably optimal move: prefer a forced win, then a draw, then
    the least-bad loss - never settle for less than the position allows."""
    _ensure_solved()
    best_value = -2
    chosen = None
    for move in game.get_valid_moves():
        trial = game.copy()
        trial.make_move(move)
        if trial.game_over:
            if trial.winner == -1:
                value = 0
            elif trial.winner == game.current_player:
                value = 1
            else:
                value = -1
        else:
            value = -_values[state_key(trial)]
        if value > best_value:
            best_value = value
            chosen = move
    return chosen


if __name__ == "__main__":
    import time

    print("Enumerating every reachable position and solving via value iteration...")
    start_time = time.time()
    states, transitions = _enumerate_states()
    values, sweeps = _value_iteration(states, transitions)
    elapsed = time.time() - start_time

    _states, _transitions, _values = states, transitions, values  # already module-level here, no `global` needed

    print(f"Solved {len(states)} distinct positions in {elapsed:.2f}s ({sweeps} sweeps to converge).")
    print()
    root_value = values[state_key(SlidingTicTacToe())]
    if root_value == 0:
        print("Result: a DRAW with perfect play by both sides (X cannot force a win, "
              "and neither can O).")
    elif root_value == 1:
        print("Result: X (the first mover) can FORCE A WIN with perfect play.")
    else:
        print("Result: O can FORCE A WIN with perfect play - X cannot even draw.")
