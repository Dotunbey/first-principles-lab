"""
Live, fully manual training dashboard: you initialize the agent (and see its
starting hyperparameters / empty Q-table before anything runs), you press
Start, you can Pause and Resume at any point mid-game, and you can change
the hyperparameters live from the UI - nothing auto-starts.

Run with: python app.py [port] [session_filename]
Then open http://127.0.0.1:<port>
Defaults: port=5000, session_filename=live_session.pkl - so two completely
independent runs (e.g. one sparse/shaped agent, one TD(0) agent) can be
trained side by side by launching this twice with different arguments;
each process has its own globals and its own save file, so they never
interfere with each other.
"""

import ast
import os
import pickle
import queue
import random
import sys
import threading
import time
from collections import defaultdict

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game_engine import SlidingTicTacToe, X, O  # noqa: E402
from mcts import mcts_search  # noqa: E402
from q_learning import QLearningAgent  # noqa: E402
from perfect_solver import best_move as perfect_solver_move  # noqa: E402
from perfect_solver import negamax_value  # noqa: E402

app = Flask(__name__)

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SESSION_FILENAME = sys.argv[2] if len(sys.argv) > 2 else 'live_session.pkl'
SESSION_FILE = os.path.join(os.path.dirname(__file__), SESSION_FILENAME)

DEFAULT_CONFIG = {
    'opponent': 'mcts',            # 'mcts' or 'perfect' - switch anytime, live
    'reward_shaping': 0,           # 0 = off (sparse baseline), 1 = on (potential-based, via perfect solver) - ignored when td0=1
    'td0': 0,                      # 0 = Monte Carlo backward pass (sparse or shaped, per reward_shaping)
                                    # 1 = true online TD(0): update every move immediately, bootstrapped
                                    #     off the agent's OWN Q-values, no solver involved at all
    'draw_reward': -0.2,           # terminal reward for a draw - negative to make drawing mildly
                                    # unattractive instead of neutral, directly targeting the
                                    # "drawing trap" (see research/08_td0_online_learning.md)
    'learning_rate': 0.1,
    'discount_factor': 0.9,
    'epsilon_start': 0.4,
    'epsilon_end': 0.05,
    'epsilon_decay_games': 300,   # epsilon reaches epsilon_end after this many TOTAL games
    'mcts_iterations': 150,
    'games_per_run': 50,          # how many more games one "Start" click plays
    'move_delay': 0.5,
    'update_delay': 0.6,
    'checkpoint_interval': 25,
    'checkpoint_eval_games': 10,
    'checkpoint_mcts_iterations': 80,
}

CONFIG = dict(DEFAULT_CONFIG)

state_lock = threading.Lock()
state = {
    'running': False,
    'paused': False,
    'initialized': False,
    'game_num': 0,               # cumulative games trained so far (this agent's lifetime)
    'games_target': 0,           # game_num we're heading to at the end of the current run
    'q_plays': None,
    'phase': None,
    'board': [0] * 9,
    'current_player': None,
    'stage': 'idle',             # idle | initialized | playing | updating_q | evaluating | paused | stopped
    'move_explanation': '',
    'q_update': None,
    'epsilon': None,
    'q_table_size': 0,
    'outcome_log': [],
    'checkpoint_history': [],
    'initial_weights': None,
    'config': dict(CONFIG),
    'q_update_log': [],   # persistent scrolling history of weight changes, not just the current one
}

agent = None
games_run_total = 0
training_thread = None
stop_flag = threading.Event()
pause_flag = threading.Event()

# Autosaving writes to disk on every single completed game (by design - not
# reduced, per instruction). The risk that showed up in practice: Windows
# (antivirus / indexer / similar) can occasionally hold a lock on the
# destination file just long enough that os.replace() sits there waiting,
# which - if done on the training thread - freezes training entirely even
# though nothing has actually crashed. Fix: never do the disk write on the
# training thread. Build the snapshot (fast, in-memory, must stay on the
# training thread to avoid racing with live Q-table mutation), then hand it
# to a dedicated saver thread that does the slow/risky part. Worst case, a
# save is delayed or silently dropped one round if Windows is being
# difficult - training itself never stalls for it.
_save_queue = queue.Queue(maxsize=3)


def _saver_thread_loop():
    while True:
        data = _save_queue.get()
        try:
            tmp_path = SESSION_FILE + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f)
            os.replace(tmp_path, SESSION_FILE)  # atomic - never leaves a half-written file behind
        except OSError:
            pass  # this save is skipped; the next one will try again in a moment


threading.Thread(target=_saver_thread_loop, daemon=True).start()


def save_session():
    """Snapshot the current agent/session state (fast, synchronous, safe to
    call from the training thread) and queue it for the saver thread to
    actually write to disk."""
    if agent is None:
        return
    with state_lock:
        outcome_log = list(state['outcome_log'])
        checkpoint_history = list(state['checkpoint_history'])
    data = {
        'q_table': {s: dict(actions) for s, actions in agent.q_table.items()},
        'learning_rate': agent.learning_rate,
        'discount_factor': agent.discount_factor,
        'exploration_rate': agent.exploration_rate,
        'games_played': agent.games_played,
        'wins': agent.wins,
        'draws': agent.draws,
        'games_run_total': games_run_total,
        'config': dict(CONFIG),
        'outcome_log': outcome_log,
        'checkpoint_history': checkpoint_history,
    }
    try:
        _save_queue.put_nowait(data)
    except queue.Full:
        pass  # a save is already pending; this snapshot is slightly stale-able, next game's will catch up


def load_session():
    """Restore a previous session on server startup, if one was saved."""
    global agent, games_run_total
    if not os.path.exists(SESSION_FILE):
        return False

    with open(SESSION_FILE, 'rb') as f:
        data = pickle.load(f)

    CONFIG.update(data['config'])
    restored_agent = QLearningAgent(
        learning_rate=data['learning_rate'],
        discount_factor=data['discount_factor'],
        exploration_rate=data['exploration_rate'],
    )
    restored_agent.q_table = defaultdict(lambda: defaultdict(float))
    for state_hash, actions in data['q_table'].items():
        for action, value in actions.items():
            restored_agent.q_table[state_hash][action] = value
    restored_agent.games_played = data['games_played']
    restored_agent.wins = data['wins']
    restored_agent.draws = data['draws']

    agent = restored_agent
    games_run_total = data['games_run_total']

    push_state(
        initialized=True, stage='restored', game_num=games_run_total,
        q_table_size=len(agent.q_table), epsilon=round(agent.exploration_rate, 4),
        outcome_log=data['outcome_log'], checkpoint_history=data['checkpoint_history'],
        config=dict(CONFIG),
        initial_weights={
            'learning_rate': agent.learning_rate,
            'discount_factor': agent.discount_factor,
            'epsilon_start': CONFIG['epsilon_start'],
            'epsilon_end': CONFIG['epsilon_end'],
            'epsilon_decay_games': CONFIG['epsilon_decay_games'],
            'q_table_entries': len(agent.q_table),
            'note': f'Restored from disk: {games_run_total} games already trained '
                    f'before this server was last stopped. Nothing was lost.',
        },
    )
    return True


def symbol(player):
    return {X: 'X', O: 'O'}.get(player)


def parse_symbol(s):
    return X if s == 'X' else O


# --- Play vs human -------------------------------------------------------
# A completely separate game session from the training loop, so a human can
# play the CURRENT agent without interfering with the background training
# loop. Only allowed while training is fully stopped (not just paused) -
# choose_action() reads agent.q_table via a defaultdict, which silently
# INSERTS missing keys as a side effect of reading, and the training thread
# may be mid-way through iterating that same dict (e.g. in
# compute_value_error) - so playing while training is active is a real
# "dictionary changed size during iteration" risk, not just a UX nicety.
#
# The agent DOES learn from these games - every move it makes gets a real
# TD(0) update (the exact same online rule used against MCTS during
# training), deferred until the consequence is known: either the game ends
# right after, or the human's reply reveals the next state to bootstrap
# off of. A human opponent is a genuinely new kind of training partner this
# project hasn't had before - every other opponent (MCTS, perfect solver,
# self-play) was some form of automated play.
human_game = None  # SlidingTicTacToe instance, or None if no game in progress
human_symbol = None
human_agent_symbol = None
human_game_counter = 0
pending_human_update = None  # (state_hash, move) for the agent's own last move, not yet resolved
human_state = {
    'active': False,
    'board': [0] * 9,
    'phase': None,
    'human_symbol': None,
    'agent_symbol': None,
    'current_player': None,
    'valid_moves': [],
    'game_over': False,
    'winner': None,
    'agent_move_explanation': '',
    'move_log': [],
}


def _human_valid_moves_json():
    """JSON-friendly valid moves for whoever's turn it is right now - list of
    ints during placement, list of [from, to] pairs during sliding."""
    if human_game is None or human_game.game_over:
        return []
    moves = human_game.get_valid_moves()
    if human_game.phase == 'placement':
        return moves
    return [list(m) for m in moves]


def _push_human_state(**kwargs):
    human_state.update(kwargs)


def _sync_human_state_from_game(agent_explanation=''):
    game = human_game
    winner = None
    if game.game_over:
        winner = 'draw' if game.winner == -1 else symbol(game.winner)
    _push_human_state(
        active=True,
        board=game.board.copy(),
        phase=game.phase,
        current_player=symbol(game.current_player) if not game.game_over else None,
        valid_moves=_human_valid_moves_json(),
        game_over=game.game_over,
        winner=winner,
        agent_move_explanation=agent_explanation,
    )
    # The agent's Q-table may have just grown from a real learning update
    # (see _agent_take_turn/_resolve_pending_human_update) - keep the
    # displayed table size honest, not just a stale number from the last
    # background-training checkpoint.
    push_state(q_table_size=len(agent.q_table))


def _resolve_pending_human_update():
    """The agent's PREVIOUS move now has a known consequence - either the
    game just ended, or it's genuinely the agent's turn again after the
    human's reply, so its own next-best-value can be bootstrapped off of -
    the same deferred-update pattern _apply_td0_update was built for."""
    global pending_human_update
    if pending_human_update is None:
        return
    state_hash, move = pending_human_update
    pending_human_update = None
    _apply_td0_update(agent, state_hash, move, human_game, human_agent_symbol,
                       game_num=f'human-{human_game_counter}')


def _agent_take_turn():
    """Agent plays its own move (greedy - best known, so its play stays
    predictable to a human opponent), then defers a REAL Q-table update for
    this move until the consequence is known (see _resolve_pending_human_update)."""
    state_hash = agent.hash_state(human_game)
    move = agent.choose_action(human_game, greedy=True)
    human_game.make_move(move)
    explanation = f"Agent (greedy, best known move): {move}"
    with state_lock:
        human_state['move_log'] = ([{
            'mover': 'agent', 'move': list(move) if isinstance(move, tuple) else move,
        }] + human_state['move_log'])[:50]

    global pending_human_update
    if human_game.game_over:
        _apply_td0_update(agent, state_hash, move, human_game, human_agent_symbol,
                           game_num=f'human-{human_game_counter}')
    else:
        pending_human_update = (state_hash, move)
    return explanation


@app.route('/api/human/state')
def api_human_state():
    return jsonify(human_state)


@app.route('/api/human/new', methods=['POST'])
def api_human_new():
    global human_game, human_symbol, human_agent_symbol, human_game_counter, pending_human_update
    with state_lock:
        if state['running']:
            return jsonify({'ok': False, 'error': 'stop training first - human play and training '
                                                    'cannot safely share the agent at the same time'}), 400
    if agent is None:
        return jsonify({'ok': False, 'error': 'initialize or load an agent first'}), 400

    body = request.get_json(silent=True) or {}
    human_symbol = parse_symbol(body.get('human_symbol', 'X'))
    human_agent_symbol = O if human_symbol == X else X
    human_game = SlidingTicTacToe()
    human_game_counter += 1
    pending_human_update = None  # a fresh game - nothing carries over from the last one
    _push_human_state(move_log=[], human_symbol=symbol(human_symbol), agent_symbol=symbol(human_agent_symbol))

    explanation = ''
    if human_game.current_player != human_symbol:
        explanation = _agent_take_turn()
    _sync_human_state_from_game(explanation)
    save_session()  # the agent's Q-table may have just changed - persist it like any other update
    return jsonify({'ok': True, 'state': human_state})


@app.route('/api/human/move', methods=['POST'])
def api_human_move():
    if human_game is None or not human_state['active']:
        return jsonify({'ok': False, 'error': 'no human game in progress - click "New Game" first'}), 400
    if human_game.game_over:
        return jsonify({'ok': False, 'error': 'game is over - start a new one'}), 400
    if human_game.current_player != human_symbol:
        return jsonify({'ok': False, 'error': "not your turn"}), 400

    body = request.get_json(silent=True) or {}
    raw_move = body.get('move')
    move = tuple(raw_move) if isinstance(raw_move, list) else raw_move
    if move not in human_game.get_valid_moves():
        return jsonify({'ok': False, 'error': 'illegal move'}), 400

    human_game.make_move(move)
    with state_lock:
        human_state['move_log'] = ([{'mover': 'human', 'move': list(move) if isinstance(move, tuple) else move}]
                                    + human_state['move_log'])[:50]

    # The human's reply is exactly the missing piece needed to resolve the
    # agent's own PREVIOUS move: either it just won/lost/drew the game, or
    # it's genuinely the agent's turn again and its own next options are
    # now knowable.
    _resolve_pending_human_update()

    explanation = ''
    if not human_game.game_over:
        explanation = _agent_take_turn()
    _sync_human_state_from_game(explanation)
    save_session()
    return jsonify({'ok': True, 'state': human_state})


def push_state(**kwargs):
    with state_lock:
        state.update(kwargs)


def wait_while_paused():
    if pause_flag.is_set():
        push_state(stage='paused', paused=True)
        # Safe to save here even though a game may be mid-flight: this runs on
        # the training thread itself, so nothing else can be mutating the
        # Q-table at the same instant (unlike calling save_session from a
        # Flask request thread while training is still active elsewhere).
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


def opponent_move(game, mcts_iterations):
    """Whichever opponent is currently selected: fast-but-approximate MCTS,
    or the slow-once-then-instant, provably unbeatable perfect solver."""
    if CONFIG['opponent'] == 'perfect':
        return perfect_solver_move(game), "Perfect solver: provably optimal move (unbeatable)"
    return (mcts_search(game, iterations=mcts_iterations),
            f"MCTS running {mcts_iterations} simulations to pick this move")


def evaluate_vs_opponent(agent, mcts_iterations, num_games):
    wins = losses = draws = 0
    for i in range(num_games):
        q_plays = X if i % 2 == 0 else O
        game = SlidingTicTacToe()
        while not game.game_over:
            if game.current_player == q_plays:
                move = agent.choose_action(game, greedy=True)
            else:
                move, _ = opponent_move(game, mcts_iterations)
            game.make_move(move)
        if game.winner == -1:
            draws += 1
        elif game.winner == q_plays:
            wins += 1
        else:
            losses += 1
    return {'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': wins / num_games * 100, 'draw_rate': draws / num_games * 100}


def current_epsilon():
    progress = min(1.0, games_run_total / max(1, CONFIG['epsilon_decay_games']))
    return CONFIG['epsilon_start'] + (CONFIG['epsilon_end'] - CONFIG['epsilon_start']) * progress


def _apply_td0_update(agent, state_hash, move, current_game, q_plays, game_num):
    """Resolve one deferred TD(0) update, now that we actually know what
    happened: either the game truly ended (real win/loss/draw, not a guess),
    or it's genuinely the agent's own turn again and we can bootstrap off its
    own best-known option from here - no sign-flip needed, since this is
    already 'my turn,' the same perspective the original move was made from."""
    old_q = agent.get_q_value(state_hash, move)

    if current_game.game_over:
        if current_game.winner == -1:
            target = CONFIG['draw_reward']
        elif current_game.winner == q_plays:
            target = 1.0
        else:
            target = -1.0
    else:
        next_state_hash = agent.hash_state(current_game)
        next_moves = current_game.get_valid_moves()
        best_own_value = max(
            (agent.get_q_value(next_state_hash, a) for a in next_moves),
            default=0.0,
        )
        target = agent.discount_factor * best_own_value

    new_q = old_q + agent.learning_rate * (target - old_q)
    agent.q_table[state_hash][move] = new_q

    q_update = {
        'game_num': game_num,
        'step': 1,
        'total_steps': 1,
        'state_hash': state_hash,
        'action': list(move) if isinstance(move, tuple) else move,
        'old_q': round(old_q, 4),
        'learning_rate': agent.learning_rate,
        'target_G': round(target, 4),
        'new_q': round(new_q, 4),
        'shaped': False,
        'mode': 'td0',
        'formula': (f"Q(s,a) = {old_q:.4f} + {agent.learning_rate} x "
                    f"[{target:.4f} - {old_q:.4f}] = {new_q:.4f}"),
    }
    with state_lock:
        state['q_update_log'] = ([q_update] + state['q_update_log'])[:60]
    push_state(stage='updating_q', q_update=q_update)


def phi_agent(game, q_plays):
    """Potential function for reward shaping: the exact solved value of
    this position (from negamax_value, always 'value to whoever is about
    to move'), re-expressed from the Q-agent's own FIXED perspective so it
    means the same thing regardless of whose turn it currently is -
    required for the potential-shaping guarantee to hold."""
    value = negamax_value(game)
    return value if game.current_player == q_plays else -value


def compute_value_error(agent):
    """The real 'training loss' for this project: how far off is the agent's
    own best-known value at each state from the solver's EXACT value at that
    same state - averaged over every state the agent has ever visited.
    Unlike win-rate, this needs no opponent, no sampling, and has zero noise:
    every term is an exact comparison against ground truth. Should trend
    toward 0 as the Q-table genuinely improves, regardless of which opponent
    is currently being played.

    Must only be called from the training thread (it iterates agent.q_table,
    which the training thread itself mutates - calling this from a different
    thread while training is live risks the same dict-mutated-during-
    iteration hazard discussed for save_session)."""
    total_error = 0.0
    count = 0
    for state_hash, actions in agent.q_table.items():
        if not actions:
            continue
        board_str, phase, player_str = state_hash.rsplit('_', 2)
        game = SlidingTicTacToe()
        game.board = list(ast.literal_eval(board_str))
        game.phase = phase
        game.current_player = int(player_str)

        true_value = negamax_value(game)
        agent_value = max(actions.values())
        total_error += abs(agent_value - true_value)
        count += 1

    return total_error / count if count else None


def training_loop(games_this_run):
    global games_run_total

    push_state(running=True, stage='playing', games_target=games_run_total + games_this_run)

    for _ in range(games_this_run):
        wait_while_paused()
        if stop_flag.is_set():
            break

        agent.exploration_rate = current_epsilon()
        games_run_total += 1
        game_num = games_run_total

        q_plays = X if game_num % 2 == 0 else O  # iterated, not fixed
        game = SlidingTicTacToe()
        q_history = []
        pending_td0_update = None  # (state_hash, move) waiting to see what happens next
        pending_selfplay_update = {X: None, O: None}  # same idea, one slot per side in self-play

        push_state(game_num=game_num, q_plays=symbol(q_plays), stage='playing',
                   board=game.board.copy(), phase=game.phase,
                   current_player=symbol(game.current_player),
                   epsilon=round(agent.exploration_rate, 4),
                   q_table_size=len(agent.q_table), q_update=None)

        while not game.game_over:
            wait_while_paused()
            if stop_flag.is_set():
                break

            player = game.current_player
            move_already_applied = False

            if CONFIG['opponent'] == 'self':
                # True self-play: the SAME agent picks moves for BOTH sides, and
                # BOTH sides' moves get a real TD(0) update - not just one "q_plays"
                # role. This is the automatic-curriculum idea: early on both sides
                # are equally weak, so wins/losses are common and informative, and
                # difficulty naturally scales up as the agent's own opponent (itself)
                # improves alongside it.
                epsilon_now = agent.exploration_rate
                is_random = random.random() < epsilon_now
                state_hash = agent.hash_state(game)
                move = agent.choose_action(game)
                explanation = (f"Self-play EXPLORING (epsilon={epsilon_now:.3f}), player {symbol(player)}"
                                if is_random else
                                f"Self-play EXPLOITING, player {symbol(player)}")

                game.make_move(move)
                move_already_applied = True

                if game.game_over:
                    _apply_td0_update(agent, state_hash, move, game, player, game_num)
                else:
                    pending_selfplay_update[player] = (state_hash, move)

                other_player = O if player == X else X
                if pending_selfplay_update[other_player] is not None:
                    prev_state_hash, prev_move = pending_selfplay_update[other_player]
                    pending_selfplay_update[other_player] = None
                    _apply_td0_update(agent, prev_state_hash, prev_move, game, other_player, game_num)
            elif player == q_plays:
                epsilon_now = agent.exploration_rate
                is_random = random.random() < epsilon_now
                state_hash = agent.hash_state(game)
                phi_before = phi_agent(game, q_plays) if CONFIG['reward_shaping'] else None
                move = agent.choose_action(game)
                explanation = (f"Q-agent EXPLORING (epsilon={epsilon_now:.3f}): random legal move"
                                if is_random else
                                "Q-agent EXPLOITING: best known move by current Q-value")

                if CONFIG['td0']:
                    # True online TD(0), done correctly: DON'T update yet - we don't
                    # know the real outcome until we see what the opponent does next.
                    # Updating immediately (using the opponent-turn state's own
                    # stored values) only works if that exact state has been trained
                    # before - which it almost never has this early, so it silently
                    # bootstraps off zeros forever. Instead, wait one half-move: once
                    # we see either the true terminal result or the agent's own next
                    # turn, THEN apply the update with real information.
                    game.make_move(move)
                    move_already_applied = True
                    if game.game_over:
                        _apply_td0_update(agent, state_hash, move, game, q_plays, game_num)
                    else:
                        pending_td0_update = (state_hash, move)
                else:
                    q_history.append((state_hash, move, phi_before))
            else:
                move, explanation = opponent_move(game, CONFIG['mcts_iterations'])

            if not move_already_applied:
                game.make_move(move)

            if CONFIG['td0'] and player != q_plays and pending_td0_update is not None:
                # The opponent just moved - now we know what actually happened to
                # the agent's last move: either the game ended for real (a true
                # win/loss/draw, not a guess), or it's genuinely the agent's turn
                # again and we can bootstrap off its OWN next options (no sign-flip
                # needed here, since it's already "my turn" - same perspective).
                prev_state_hash, prev_move = pending_td0_update
                pending_td0_update = None
                _apply_td0_update(agent, prev_state_hash, prev_move, game, q_plays, game_num)

            push_state(board=game.board.copy(), phase=game.phase,
                       current_player=symbol(game.current_player) if not game.game_over else None,
                       move_explanation=explanation, stage='playing')
            interruptible_sleep(CONFIG['move_delay'])

        if stop_flag.is_set():
            break

        outcome = 'draw' if game.winner == -1 else ('q_win' if game.winner == q_plays else 'q_loss')
        if game.winner == q_plays:
            terminal_reward = 1
        elif game.winner == -1:
            terminal_reward = CONFIG['draw_reward']
        else:
            terminal_reward = -1

        shaping_on = bool(CONFIG['reward_shaping'])
        total_steps = len(q_history)
        # Unshaped: seed G at the terminal reward, step reward is always 0.
        # Shaped: the terminal reward is folded INTO the potential function
        # instead (Φ(terminal) = terminal_reward), so G starts at 0 and every
        # step gets its own informative reward = γΦ(s') - Φ(s). Same telescoping
        # math either way - shaping only changes how quickly credit propagates,
        # never what the optimal move ends up being (Ng, Harada & Russell 1999).
        future_return = 0.0 if shaping_on else terminal_reward

        for i in range(total_steps):
            wait_while_paused()
            if stop_flag.is_set():
                break
            j = total_steps - 1 - i  # chronological index of the move being updated
            state_hash, move, phi_before = q_history[j]

            if shaping_on:
                phi_next = q_history[j + 1][2] if j + 1 < total_steps else terminal_reward
                step_reward = agent.discount_factor * phi_next - phi_before
                future_return = step_reward + agent.discount_factor * future_return
            # else: future_return was already fully computed by the previous
            # iteration's discounting below - nothing extra to add here.

            old_q = agent.get_q_value(state_hash, move)
            new_q = old_q + agent.learning_rate * (future_return - old_q)
            agent.q_table[state_hash][move] = new_q

            q_update = {
                'game_num': game_num,
                'step': i + 1,
                'total_steps': total_steps,
                'state_hash': state_hash,
                'action': list(move) if isinstance(move, tuple) else move,
                'old_q': round(old_q, 4),
                'learning_rate': agent.learning_rate,
                'target_G': round(future_return, 4),
                'new_q': round(new_q, 4),
                'shaped': shaping_on,
                'formula': (f"Q(s,a) = {old_q:.4f} + {agent.learning_rate} x "
                            f"[{future_return:.4f} - {old_q:.4f}] = {new_q:.4f}"),
            }
            with state_lock:
                state['q_update_log'] = ([q_update] + state['q_update_log'])[:60]
            push_state(stage='updating_q', q_update=q_update)
            interruptible_sleep(CONFIG['update_delay'])
            if not shaping_on:
                future_return = agent.discount_factor * future_return
            # (shaped case already computed the fully-discounted G_j above - don't discount twice)

        agent.games_played += 1
        if game.winner == -1:
            agent.draws += 1
        else:
            agent.wins[game.winner] += 1

        with state_lock:
            log = [{'game_num': game_num, 'q_plays': symbol(q_plays), 'outcome': outcome,
                    'opponent': CONFIG['opponent']}] + state['outcome_log']
            state['outcome_log'] = log[:100]
        push_state(q_table_size=len(agent.q_table))

        if game_num % CONFIG['checkpoint_interval'] == 0:
            push_state(stage='evaluating')
            results = evaluate_vs_opponent(agent, CONFIG['checkpoint_mcts_iterations'], CONFIG['checkpoint_eval_games'])
            value_error = compute_value_error(agent)  # exact, opponent-independent, zero-noise
            with state_lock:
                history = state['checkpoint_history'] + [{
                    'game_num': game_num,
                    'win_rate': results['win_rate'],
                    'draw_rate': results['draw_rate'],
                    'opponent': CONFIG['opponent'],
                    'value_error': value_error,
                }]
                state['checkpoint_history'] = history

        save_session()  # autosave after every completed game - at most one in-progress game is ever at risk

    save_session()  # also save right when the loop exits (e.g. Stop mid-game) - still the training thread, still race-free
    push_state(running=False, paused=False, stage='stopped' if stop_flag.is_set() else 'idle')


def _make_fresh_agent():
    return QLearningAgent(
        learning_rate=CONFIG['learning_rate'],
        discount_factor=CONFIG['discount_factor'],
        exploration_rate=CONFIG['epsilon_start'],
    )


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


@app.route('/api/init', methods=['POST'])
def api_init():
    global agent, games_run_total
    with state_lock:
        if state['running']:
            return jsonify({'ok': False, 'error': 'stop or pause current run first'}), 400

    _apply_config_from_request()
    agent = _make_fresh_agent()
    games_run_total = 0
    stop_flag.clear()
    pause_flag.clear()

    push_state(
        running=False, paused=False, initialized=True, stage='initialized',
        game_num=0, games_target=0, q_plays=None, board=[0] * 9, phase=None,
        current_player=None, move_explanation='', q_update=None, q_update_log=[],
        outcome_log=[], checkpoint_history=[],
        epsilon=round(agent.exploration_rate, 4), q_table_size=0,
        initial_weights={
            'learning_rate': agent.learning_rate,
            'discount_factor': agent.discount_factor,
            'epsilon_start': CONFIG['epsilon_start'],
            'epsilon_end': CONFIG['epsilon_end'],
            'epsilon_decay_games': CONFIG['epsilon_decay_games'],
            'q_table_entries': 0,
            'note': 'Every state defaults to Q=0.0 until the agent actually visits it - '
                    'there is nothing else to show yet, the table is empty.',
        },
    )
    save_session()  # overwrite any old saved session immediately - a fresh reset should stick across restarts too
    return jsonify({'ok': True, 'config': CONFIG})


@app.route('/api/update_params', methods=['POST'])
def api_update_params():
    """Change hyperparameters live - takes effect on the very next move/update,
    even mid-run."""
    _apply_config_from_request()
    if agent is not None:
        agent.learning_rate = CONFIG['learning_rate']
        agent.discount_factor = CONFIG['discount_factor']
    return jsonify({'ok': True, 'config': CONFIG})


@app.route('/api/start', methods=['POST'])
def api_start():
    global agent, training_thread
    with state_lock:
        if state['running']:
            return jsonify({'ok': False, 'error': 'already running'}), 400

    _apply_config_from_request()
    if agent is None:
        agent = _make_fresh_agent()
        push_state(initialized=True)

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
    pause_flag.clear()  # don't leave it stuck waiting on a pause that will never lift
    return jsonify({'ok': True})


load_session()  # restore whatever was last saved, if anything - runs once at process startup

if __name__ == '__main__':
    print(f"Session file: {SESSION_FILE}")
    app.run(debug=False, port=PORT)
