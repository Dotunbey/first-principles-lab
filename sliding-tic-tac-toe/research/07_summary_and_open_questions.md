# Summary Timeline and Open Questions

## Chronological summary of concrete results

1. **State space predicted by hand**: standard tic-tac-toe = 6,046 raw combinatorial
   boards (5,478 after removing unreachable post-win continuations); sliding tic-tac-toe =
   4,030 raw combinatorial boards, before any code was written.
2. **Game engine built and stress-tested**: 2,000 random self-play games, zero deadlocks,
   zero runaway games, max total moves observed = 46 (exactly matching the design: 6
   placement + 40 slide-cap).
3. **MCTS built, one bug fixed pre-launch** (terminal nodes falsely reporting untried
   moves), then validated at 20/20 wins vs. a random opponent with zero training.
4. **Q-learning agent built separately, trained via true self-play**: 5,000 games, final
   Q-table size 4,424 states (matching the by-hand combinatorial prediction), 99.5% win
   rate vs. random.
5. **MCTS's rollout policy strengthened** from uniform-random to a one-ply-lookahead
   heuristic - roughly 30x slower per iteration, but far more informative per simulation.
6. **A perfect solver built, found to be unsound via empirical testing** (lost 9/20 games
   to MCTS despite claiming optimality), root-caused to a memoization/cycle-detection
   flaw, and corrected via value iteration over the fully enumerated state graph (4,974
   states, 9 sweeps, 0.32s) - re-validated at 12 wins / 8 draws / **0 losses** against
   MCTS, and the corrected solve determined that **X can force a win** with perfect play
   under this project's exact rules (the initial, buggy solver had incorrectly reported a
   draw).
7. **A full live training dashboard built**, including two real bugs found and fixed during
   actual use: a UI state-sync bug (the speed slider silently overwriting manually-set
   delay values) and a genuine background-thread freeze (a Windows file-lock stalling the
   autosave's disk write), the latter recovered from twice with zero data loss and fixed
   architecturally by moving disk I/O off the training thread entirely.
8. **Potential-based reward shaping implemented using the solver's exact values**,
   validated in isolation with a concrete demonstration of sharper credit assignment
   (a -1.9 penalty pinpointing an actual blunder, vs. a flat +0.1 for already-decided
   moves) - then found, via a real extended training run, to cause a sustained win-rate
   decline (60% -> 0% over ~700 games) when the opponent was MCTS (an imperfect player),
   and to instead produce the exact theoretical performance ceiling (50% win / 50% draw /
   0% loss) when the opponent was switched to the perfect solver - directly confirming
   that the shaping technique's policy-invariance guarantee depends on an assumption
   (the opponent also plays optimally) that does not hold against MCTS.
9. **An exact, opponent-independent "training loss" metric added**: mean absolute error
   between the agent's own best-known value at each state and the solver's exact value
   there, averaged over the whole Q-table - zero sampling noise, unlike win-rate. Used to
   confirm (via two independent, agreeing metrics) that reverting shaping back to sparse
   against MCTS produces genuine, if gradual, recovery rather than noise.
10. **TD(0) implemented as a third, "pure" dense-reward option** (no solver dependency,
    bootstrapped only off the agent's own Q-values) - the first version shipped with a real
    bug (only detected a real outcome when the agent's *own* move ended the game, missing
    the far more common case of the *opponent's* move ending it, leaving the table stuck
    at all zeros for 550 games), caught by inspecting the raw update log directly, and
    fixed by deferring each update until the true next outcome is known. See
    `08_td0_online_learning.md`.
11. **A mini-AlphaZero built from scratch (numpy only)**: a small neural network (value +
    policy heads), MCTS guided by it (PUCT selection, instant value-network leaf
    evaluation instead of rollouts), and a closed self-play training loop where the
    search's own outputs become the network's next training targets. Validated the
    network generalizes to never-before-seen states (test MAE 0.566 -> 0.206 on held-out
    solved positions - something no table-based approach in this project could ever do),
    then found and fixed a real bug in the guided search (losing 14/20 to random before
    the fix, 20/20 after) caused by trusting `game_engine.py`'s `current_player` field
    immediately after a win, when it does not actually flip in that case. See
    `10_alphazero_style_implementation.md`.

## Recurring methodological pattern

Several of the most significant findings here came from treating a component's own claims
as testable predictions rather than trusting the implementation on inspection alone:

- The perfect solver's claim of optimality was checked by trying to beat it with an
  independent opponent (MCTS), which is what caught the cycle-detection bug - code review
  alone had not caught it.
- The reward-shaping regression was caught by watching the actual win-rate trend over an
  extended live run rather than assuming the theorem's guarantee applied unconditionally.
- The dashboard "hang" was diagnosed by comparing live process state across a time gap
  (to distinguish "frozen" from "just slow") and by inspecting actual file contents on
  disk (to distinguish "corrupted" from "one step behind"), rather than guessing.

## Open questions / natural next experiments

1. **Controlled shaping comparison against MCTS**: with the same (already-trained) Q-table,
   turn `reward_shaping` back off while keeping the opponent as MCTS, and confirm the
   win-rate recovers over the following checkpoints - the direct before/after comparison
   that would fully close the loop on the finding in
   `06_reward_shaping_experiment.md`, beyond the theoretical diagnosis and the
   perfect-solver confirmation already gathered.
2. **A from-scratch, matched-conditions comparison**: train two fresh agents under
   identical schedules (same number of games, same opponent, same epsilon schedule) -
   one with sparse reward throughout, one with shaping throughout - and compare their
   full win-rate-over-training curves, rather than switching schemes mid-stream on an
   already-partially-trained table (which is what happened here, and which itself
   introduced a transient adjustment period that was hard to cleanly separate from the
   opponent-mismatch effect).
3. **Is there a "partial" or opponent-aware shaping** that degrades gracefully against an
   imperfect opponent instead of actively misleading the agent - for instance, shaping only
   moves toward a position's value where that value is *close to* what's still achievable
   against a fallible opponent, rather than the pure "assume perfect resistance" value?
   Not attempted here, but a natural follow-up suggested directly by the diagnosis in
   `06_reward_shaping_experiment.md`.
4. **Does the same reward-shaping mismatch generalize** to the earlier Oware project's
   shaped rewards (capture bonuses, center-control bonuses)? Those bonuses are not
   potential-based (they don't have the Ng/Harada/Russell policy-invariance guarantee at
   all), so they were always a different, less rigorous category of shaping than what was
   built here - but the general question of "does this bonus assume something about the
   opponent that isn't true" is worth re-examining there too.
5. **MCTS strength ceiling vs. training practicality**: the heuristic rollout made MCTS
   meaningfully stronger per iteration but ~30x more expensive; whether a further-improved
   rollout policy (e.g. also preferring moves that create a capture/threat, not just
   avoiding an immediate loss) would shift that tradeoff favorably has not been measured.
6. **RESOLVED: why was TD(0)'s value error rising?** Diagnosed directly by splitting error
   by the position's true value (`08_td0_online_learning.md`): the agent had learned
   drawish positions almost perfectly (error ~0) but had learned almost nothing about
   decisively winning or losing ones (error ~0.9-1.0, i.e. still valuing them near a draw).
   Named the "drawing trap" - winning/losing are rare under an already-drawing policy, so
   those states get little training signal, which removes any incentive to pursue them,
   which keeps the policy drawing.
6b. **PARTIALLY CONFIRMED: opponent strength is a real contributing factor.** A fresh TD(0)
   agent trained against a much weaker MCTS (50 vs. 400 iterations) showed real, substantial
   win rates (10-25% vs. ~0%) and *decreasing* value error, plus a genuine (if still small)
   separation between its average value for winning vs. drawish positions - absent in the
   400-iteration run. But raw calibration error for winning/losing states remained large in
   both runs (~0.94-0.996) - weakening the opponent helps the agent start accumulating real
   positive experience, but does not by itself fix TD(0)'s inherently slow one-step credit
   propagation to rare states. See `08_td0_online_learning.md` for the full comparison.
7. **Head-to-head TD(0) vs. sparse vs. shaped, matched conditions**: two independent
   dashboard processes are now running in parallel (`08_td0_online_learning.md`,
   "Infrastructure" section) specifically to make this comparison observable directly
   rather than through sequential before/after bookkeeping on one shared table.
