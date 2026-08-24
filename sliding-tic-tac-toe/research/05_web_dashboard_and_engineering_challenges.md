# The Live Training Dashboard, and Two Real Engineering Bugs

## Motivation

Running training scripts from the terminal and reading printed statistics after the fact
left the actual moment-to-moment process of learning invisible. A Flask-based live
dashboard (`web_dashboard/`) was built specifically so that every part of training could be
watched and controlled as it happened, not summarized afterward.

## Feature set, as it was built up

1. **Live board + move explanation** - the board re-renders after every move, with a plain
   -language reason for that move ("Q-agent EXPLORING (epsilon=0.37): random legal move",
   "MCTS running 150 simulations to pick this move", "Perfect solver: provably optimal move
   (unbeatable)").
2. **The actual backward-pass math, live** - after every game, the real Q-value update
   streams to the browser one step at a time, with real numbers substituted into the
   formula (e.g. `Q(s,a) = 0.0000 + 0.1 x [-1.0000 - 0.0000] = -0.1000`), verified directly
   via DOM inspection during testing, not just assumed from the underlying code.
3. **Side alternation** - which side (X or O) the learning agent controls alternates every
   game (`q_plays = X if game_num % 2 == 0 else O`), so the agent isn't always stuck with
   or without the first-move advantage; verified by inspecting the outcome log's per-game
   side tags.
4. **Full manual control** - Initialize / Start / Pause / Resume / Stop, all
   user-triggered; nothing auto-starts. A `pause_flag` / `stop_flag` pair (Python
   `threading.Event`s) are checked at fine-grained points inside the training loop
   (`wait_while_paused`, `interruptible_sleep`) so pausing takes effect within a fraction of
   a second rather than only between whole games.
5. **"Initial weights" visibility** - clicking Initialize shows the fresh agent's
   hyperparameters and confirms the Q-table is empty, before a single game is played.
6. **Live-editable hyperparameters** - learning rate, discount factor, epsilon
   schedule, MCTS iterations, checkpoint settings, and more, all changeable via "Apply
   Params Live" without stopping or resetting training.
7. **A live speed control** and a **persistent "weight changes" feed** (not just the single
   current update, a scrolling history of recent ones) - added after explicit user feedback
   that the single-panel view made it feel like changes were "hidden" rather than watched.
8. **Opponent selector** - MCTS (adjustable iteration count) or the perfect solver,
   switchable live mid-run; both the outcome log and the win-rate chart tag which opponent
   was faced at each point, since the two are not comparable difficulty levels (see
   `06_reward_shaping_experiment.md` for why this distinction became important).
9. **The win-rate ("closing the gap") chart** - checkpointed every N games, plotted with a
   hand-drawn canvas chart (no external charting library dependency), color-coded by which
   opponent was active for that checkpoint.
10. **Disk persistence** - see below.

## Bug 1: the speed slider silently overwriting manually-set values

**Symptom**: delays were explicitly set to 0 and "the speed was increased," but training
still appeared frozen/very slow.

**Diagnosis**: checking the live server's config directly (`/api/state`) showed
`move_delay` and `update_delay` back at ~1.22 seconds, not 0. The speed slider was wired to
unconditionally POST its own mapped delay value both on page load and on every interaction
- so loading or refreshing the page, or touching the slider even once, would silently
overwrite a delay value set any other way (typed directly into the delay fields, or
persisted from a previous session).

**Fix**: the slider no longer fires automatically on page load; instead it reads its
starting position *from* the live config (a one-time, read-only sync that doesn't trigger
a POST, since setting an element's `.value` in JavaScript does not fire `change`/`input`
events), and only sends an update on a genuine user drag (`onchange`).

## Bug 2: a genuine background-thread freeze caused by a Windows file-lock

This was more serious, since it risked losing real training progress (thousands of games
at stake), and went through two escalating rounds of diagnosis.

### Round 1

**Symptom**: "it is hanging" reported after enabling autosave-heavy settings.

**Diagnosis process**: polled `/api/state` twice across a several-second gap and found the
*exact same* game number and the *exact same* in-progress Q-update recorded both times -
confirming a genuine freeze, not just slow progress. Checking the dashboard's save
directory turned up a leftover `live_session.pkl.tmp` file sitting alongside the real
`live_session.pkl` - the autosave writes to a temp file and atomically renames it into
place (`os.replace`) specifically so a crash never leaves a half-written save file behind;
a stray `.tmp` file left over is a strong signal that this final rename step was the exact
point execution got stuck.

**Hypothesis**: on Windows, antivirus real-time scanning or a file indexer can transiently
hold a lock on a file being written or replaced, causing `os.replace` to block. The
project folder was checked and confirmed **not** to be under OneDrive sync (a very common
cause of exactly this symptom) via the OneDrive environment variable, the actual
"Documents" known-folder path, and the absence of any cloud-placeholder file attributes -
ruling that specific cause out, leaving a generic antivirus/indexer file-lock as the
remaining likely explanation.

**Mitigation applied at the time**: a Windows Defender exclusion added for the project
folder (via Settings > Windows Security > Virus & threat protection > Exclusions, or
`Add-MpPreference -ExclusionPath` from an elevated PowerShell).

### Round 2 - it recurred, at much higher stakes (5,920 games trained)

The same signature appeared again later: identical frozen state across a time gap, and a
`.tmp` file sitting next to the real save file. This time, given how much progress was on
the line, both files' actual pickle contents were inspected directly rather than just their
existence:

```
live_session.pkl:     games_run_total=5919, q_table entries=1451, VALID
live_session.pkl.tmp: games_run_total=5920, q_table entries=1451, VALID
```

Both files loaded successfully - the `.tmp` file was not corrupt, it was simply one game
*more* advanced than the committed save, frozen at the final rename step. Rather than
discard it, it was manually promoted (`os.replace('live_session.pkl.tmp',
'live_session.pkl')`) before restarting the server, recovering that extra game with
**zero data loss** (confirmed post-restart: `games_run_total=5920`).

**Real fix** (the Defender exclusion alone was not sufficient - the recurrence proved that):
the underlying design flaw was that the actual disk write (`pickle.dump` + `os.replace`)
was happening directly on the training thread, so if the OS ever stalled that operation for
any reason, the entire training loop stalled with it. The fix was architectural rather than
just retrying harder: the fast, in-memory part of taking a snapshot (which *must* run on
the training thread, to avoid racing with live Q-table mutation from a second thread) was
separated from the slow, failure-prone disk write, which now runs on a dedicated background
"saver" thread fed via a small bounded queue (`queue.Queue(maxsize=3)`, non-blocking
`put_nowait`, silently skip-and-retry-next-time on any I/O error). Save *frequency* was
deliberately left unchanged (still every completed game, per explicit instruction not to
reduce it) - what changed is that a stalled or failed save can now only delay/skip that one
save, never freeze training itself.

Verified after the fix: restarted cleanly (0 games lost), then ran a fresh 30-game batch to
completion with no freeze (5,920 -> 5,950 games, finished naturally).

### Why this bug matters methodologically

Both rounds were diagnosed the same disciplined way: don't assume a cause, check the live
process state directly (comparing snapshots across a time gap to distinguish "frozen" from
"just slow"), then check the filesystem evidence (`.tmp` file presence, then its actual
contents) before deciding on a fix. The first round's mitigation (an OS-level exclusion)
treated a symptom of the environment; the second round's fix addressed the actual software
design flaw (blocking I/O on a thread that must never block) that made the symptom possible
in the first place, and directly caused by the constraint that save frequency was not
allowed to be reduced as a workaround.

## Disk persistence, in general

Every completed game triggers a snapshot of the full session (Q-table, hyperparameters,
outcome log, checkpoint history, cumulative game count) queued for the background saver.
On startup, the server automatically loads the last saved session, if any. This was
explicitly tested by fully killing and restarting the server process (not just pausing) to
simulate a real shutdown, and confirming the restored state matched exactly what had been
trained before the kill - both early on (5 games) and much later (5,920 games, as part of
diagnosing Bug 2 above). The one acknowledged limit: a game that is still in progress at
the exact instant of a kill (not paused, not yet completed) is not saved - only fully
completed games are guaranteed durable.
