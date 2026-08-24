# TD(0): A "Pure" Dense-Reward Alternative to Solver-Based Shaping

## Motivation

The reward-shaping experiment (`06_reward_shaping_experiment.md`) established a real
tradeoff: sparse reward is slow but safe; potential-based shaping via the perfect solver
is fast but only safe when the opponent's behavior matches the solver's "assume optimal
play" assumption - which MCTS does not satisfy, causing a measured 60% -> 0% win-rate
collapse.

This raised a natural question: is there a way to get dense, every-move feedback *without*
depending on an external oracle that assumes anything about the opponent? The answer is
yes, and it is not a new invention - it is the original Bellman/Q-learning update itself,
applied online (every move, immediately) rather than as a Monte Carlo backward pass at the
end of the game:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max Q(s',a') - Q(s,a) ]
```

The key property that makes this "pure": `max Q(s',a')` is the agent's **own current
belief** about the next state, not an external solver's opinion of what should happen
under idealized play. It is self-referential - it only ever reflects what the agent has
actually learned from real games against whatever opponent it is actually facing.

## The two-player wrinkle

In a single-agent MDP, "the next state" is unambiguous. In this adversarial, alternating-
move game, the state immediately after the agent's own move has the **opponent** to move,
and the Q-table's convention throughout this whole project is "value to whoever's turn it
currently is." So naively taking `max` over that immediate next state would measure the
opponent's best outcome, not the agent's - the sign is backwards relative to the agent's
own perspective.

Two ways to resolve this were considered:
1. **Negate immediately**: bootstrap off the very next state, but flip the sign (the same
   negamax trick used throughout - solver, MCTS backprop, and the `phi_agent` shaping
   potential).
2. **Wait one half-move**: let the opponent's real move actually happen, then bootstrap off
   the state where it is genuinely the agent's own turn again (or the true terminal
   outcome) - no sign flip needed, since it is already "my turn" by construction.

## First implementation (option 1) and the bug it hid

The first implementation used option 1: immediately after the agent's move, if the game
wasn't over, bootstrap by looking up the opponent-turn state's own stored Q-values and
negating the best one.

This shipped, was verified mechanically (update log showed live formula output), and a
fresh agent was trained against MCTS from game 0. After 550 games, a direct look at the
"weight changes" feed (not just the aggregate charts) showed something wrong: **every
single logged update read exactly `0.000 -> 0.000`.** Nothing had moved at all.

### Root cause

The bug: a real win/loss/draw target was only ever produced when the **agent's own move**
directly ended the game. But in most real games, it is the **opponent's subsequent move**
that actually delivers the win - and in that far more common case, the code bootstrapped
off the opponent-turn state's stored values, which are essentially always still exactly 0
this early (those specific "opponent to move here" table entries are never trained
directly - the agent's own side alternates, so an exact match is rare). The result: the
agent almost never received a real, informative target at all. Not "slow learning" -
effectively no learning.

This was caught by inspecting the raw update log directly, not by trusting the theory or
the aggregate charts - the same methodological lesson as the earlier bugs in this project
(`07_summary_and_open_questions.md`'s "recurring methodological pattern" section).

### The fix (option 2)

Switched to the "wait one half-move" approach: the agent's move is applied, but its
Q-update is deferred (`pending_td0_update`) unless it directly ends the game. Once the
opponent's actual reply is known, the deferred update is resolved using either the true
terminal outcome or the agent's own next-turn options - correct information either way,
and no sign-flip required since the bootstrap always happens from the agent's own
perspective at the moment it's genuinely evaluated.

Re-running from a fresh agent immediately showed real signal:

```
Q(s,a) = 0.0000 + 0.1 x [-1.0000 - 0.0000] = -0.1000
```

Real `-1.0` targets, correctly assigning blame to the agent's move right before the
opponent's actual winning reply - exactly the case the original version could never see.

## Early results (fresh agent, TD(0), vs. MCTS 400 iterations)

Checkpoints from games 1300-1750 (post-fix):

```
draw_rate: consistently 85-95%
win_rate: mostly 0%, one blip to 5%
value_error: 0.299 -> 0.324 (slowly RISING, not falling)
```

**The encouraging part**: a combined non-loss rate (win+draw) of 85-95% against MCTS this
early is a strong result - noticeably better than what sparse or shaped Monte Carlo showed
at a comparable stage.

**The open puzzle**: value error (the exact metric from `07_summary_and_open_questions.md`
/ the dashboard's `compute_value_error`) is drifting slightly upward rather than toward
zero, even as game outcomes improve. Working interpretation: TD(0) may be learning a good
*defensive policy* (which moves avoid losing) faster than it is learning accurate
*values* - the same behavior-vs-values gap first noticed with the shaped agent's perfect
50/50/0 plateau against the solver despite a non-zero, non-shrinking value error there too.
Whether value error eventually turns over and starts decreasing, or whether this is a
stable "correct enough to act, not yet accurate" pattern, is not yet resolved.

## Infrastructure: two agents training in parallel

Rather than repeatedly switching one shared agent between experimental conditions (which
is what made the reward-shaping recovery experiment harder to read cleanly - see
`06_reward_shaping_experiment.md`), the dashboard server (`web_dashboard/app.py`) was made
to accept a port and a session filename as command-line arguments:

```
python app.py [port] [session_filename]
```

Two independent processes are now running side by side, each with its own memory and save
file, never interfering with each other:
- **Port 5000** (`live_session.pkl`): the original sparse/shaped-history agent, ~47,000
  games.
- **Port 5001** (`live_session_td0.pkl`): the fresh TD(0) agent described above.

This makes future comparisons directly observable in real time rather than requiring
careful before/after bookkeeping on a single shared table.

## The value-error puzzle resolved: a "drawing trap"

By game ~1,850 the pattern had sharpened into something unambiguous: draw rate steady at
80-95%, win rate essentially 0% (one blip to 5%), and value error climbing smoothly and
consistently across 15 checkpoints (0.284 -> 0.335 across games 1150-1850) rather than
settling.

The initial hypothesis - the agent specifically undervalues the positions where it plays
X (the side proven able to force a win) - was tested directly by splitting mean value error
by whose turn it is at each state. That split did not cleanly confirm the hypothesis (X's
states showed only modestly higher error, 0.368 vs. 0.310 for O), because averaging over
*all* visited states mixes together many different position types. Splitting instead by
the *true* value of the position (using the solver's exact `negamax_value`) was far more
informative:

```
Winning positions (true value = +1.0): agent's own estimate ~= 0.004  -> mean |error| ~= 0.996
Drawish positions (true value ~= 0.0): agent's own estimate ~= 0.000  -> mean |error| ~= 0.000
Losing positions  (true value = -1.0): agent's own estimate ~= -0.09  -> mean |error| ~= 0.910
```

**The agent has essentially perfectly learned to recognize drawish positions (n=353, ~64%
of the visited table, error ~0), but has learned almost nothing about decisively winning
or losing positions (n=74 and n=127 respectively, error ~0.9-1.0, close to the maximum
possible)** - it still values a genuinely winning position at roughly the same as a dead
draw.

This single result explains every observation from this run at once:
- **High draw rate**: defensive values are excellent and well-calibrated - the agent is
  very good at recognizing and steering into drawn positions.
- **Near-zero win rate**: since a winning position and a drawish position look the same to
  the agent (both valued near 0), there is no internal signal telling it to specifically
  pursue or press a winning line even when one is available.
- **Rising value error**: as the Q-table grows and picks up more of these rare,
  still-uncalibrated decisive-outcome entries, each contributing ~0.9-1.0 error, the
  average rises even while the much more numerous drawish entries stay essentially exact.

**The likely mechanism - a self-reinforcing "drawing trap"**: winning and losing are
comparatively rare events under a policy that already draws 80-95% of the time, so the
specific (state, action) pairs that lead toward decisive outcomes receive far fewer real
TD(0) updates than the ones leading to draws. Those decisive-outcome states therefore stay
poorly calibrated, which removes any incentive (via greedy action selection) to steer
toward them over the "safe," already-well-known drawish alternative - reinforcing the same
drawing behavior. This is a specific instance of a well-known general RL phenomenon: without
some form of directed exploration bonus for rare-but-potentially-high-value regions, an
agent can settle into a self-perpetuating, locally-stable-but-globally-suboptimal
equilibrium.

This is a materially different failure mode than the reward-shaping opponent-mismatch
problem in `06_reward_shaping_experiment.md`. That one was a bias actively pushing the
agent's beliefs in a wrong direction. This one is closer to insufficient exploration
pressure toward rare, high-value outcomes - the values it has learned for what it actually
experiences are accurate (the drawish-position error is ~0), it just hasn't experienced
enough of the decisive-outcome states to calibrate them.

## Testing the fix: does a weaker opponent break the trap?

The "drawing trap" mechanism implies a direct lever: if wins are rare because the opponent
is strong, a weaker opponent should make real wins common enough to seed the table with
genuine positive experience before the safe-drawing equilibrium can set in. This was
tested directly: a fresh TD(0) agent, identical in every other respect, trained against
MCTS at only 50 iterations (down from 400) - a separate parallel dashboard instance
(port 5002, `live_session_td0_easy_mcts.pkl`) so as not to disturb either of the other two
running experiments.

At a comparable stage (games 50-800), the character of the run was immediately and
obviously different from the 400-iteration version:

```
win_rate: real and substantial throughout (5-25%, mostly 10-25%) - never stuck at 0
value_error: 0.530 -> 0.448 across 16 checkpoints - DECREASING, not rising
```

The same winning/drawish/losing bucket breakdown (at game 1,249, n=714 states) shows a
more nuanced picture than "problem solved":

```
Winning (true=+1.0): agent's own estimate ~= 0.060  -> mean |error| ~= 0.940
Drawish (true~=0.0):  agent's own estimate ~= 0.005  -> mean |error| ~= 0.005
Losing (true=-1.0):   agent's own estimate ~= -0.063 -> mean |error| ~= 0.937
```

Compare to the 400-iteration run's same breakdown: winning ~=0.004, drawish ~=0.000, losing
~=-0.090. The raw error magnitude in the winning bucket is still large in *both* runs
(~0.94-0.996) - TD(0)'s one-step credit propagation is inherently slow to fully calibrate
rare states regardless of opponent strength, and that has not been fixed by this change.

**What has changed is the separation between buckets.** In the 400-iteration run, the
agent's average value for winning positions (0.004) was statistically indistinguishable
from drawish positions (0.000) - the agent had no way to tell a winning position from a
merely drawing one. Here, winning positions average 0.060 versus drawish positions' 0.005 -
a real, if still small, gap. That gap is precisely what a greedy policy needs in order to
prefer the winning line over the safe one, which is consistent with the healthy (real wins,
falling value error) aggregate trend above.

**Verdict: opponent strength is a real, meaningfully contributing factor to the drawing
trap - but not the whole story.** Weakening the opponent clearly helps the agent start
accumulating genuine positive experience and avoid the runaway one-sided drift seen before.
It does not, by itself, fully solve TD(0)'s underlying slowness at calibrating rare
decisive-outcome states precisely - that limitation shows up in both runs and is a
separate, still-open target for a fix like eligibility traces / TD(lambda) (see below) or
optimistic initialization, rather than something opponent difficulty alone resolves.

## Status

Two levers have now been identified for the drawing trap, both worth pursuing rather than
treating as alternatives: (1) opponent/curriculum strength, confirmed above to have a real
effect, and (2) the credit-propagation speed itself, which weakening the opponent does not
fix and which techniques like eligibility traces (TD(lambda) - interpolating between this
project's `td0` and `sparse` modes via a single parameter, see discussion in the
conversation record) or optimistic initialization directly target. Natural next step:
implement one of these two techniques on top of the current TD(0) mode and see whether the
winning-bucket error actually closes, rather than just separating from the drawish bucket.
See `07_summary_and_open_questions.md` for the updated open-questions list.
