# Would the Agent Still Work Without the Engine's Draw Rules?

A natural question once it was established that threefold-repetition and the 40-move cap
are engineering conventions, not official rules (`01_game_definition_and_state_space.md`):
if this trained agent were deployed against a real opponent under the "real," unmodified
game (no repetition rule, no move cap), would it still function?

## The agent's knowledge does not depend on either rule

The Q-table is indexed purely by `(board, phase, player to move)` - nothing about move
count or repetition count is part of any state the agent reasons about. Removing either
rule doesn't remove information the agent needs or introduce states it has never seen; the
same roughly 4,000-ish reachable positions exist regardless of what convention (if any)
eventually stops a real match.

## What the agent learned from those rules remains valid without them

When a training game ended in a draw via repetition or the move cap, the agent learned
"this position/sequence is worth 0 (non-decisive)." That lesson is still true in a
ruleless real-world context: if two players loop the same position with neither able to
force a win, that genuinely *is* a non-decisive situation whether or not anyone formally
calls it a draw. The learned value continues to correctly describe reality.

## Where the real risk actually lives: the game, not the agent

Without *some* termination rule, sliding tic-tac-toe has no guaranteed end at all - this is
the same fundamental problem the rules were built to solve in the first place
(`01_game_definition_and_state_space.md`). Deployed against a real opponent with no
draw-forcing convention, a well-matched defensive game could in principle continue
indefinitely, exactly as it could during training without the cap. That is a property of
the underlying game, not a limitation introduced by, or specific to, this trained agent.

## Practical takeaway

The agent would still make sensible, well-informed decisions on every single turn with or
without these rules in place. But any real deployment (e.g. letting a human actually play a
full match against it) would still want *some* stopping convention purely so the match has
a defined end - not because the agent's decision-making requires it.
