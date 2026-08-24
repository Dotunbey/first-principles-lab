# The Neural Network: Full Math and Architecture

Code: `neural_net.py`. Read `00_overview.md` first if you haven't.

## 1. Why a network at all - the generalization argument, concretely

Before any math, the single most important idea: a table memorizes, a network
generalizes.

Imagine the Q-table agent from earlier in this project sees this position for the very
first time:

```
X . O          Never-seen-before board.
. X .          Table lookup: Q(this exact state, any action) = 0 (default)
O . .          The table knows LITERALLY NOTHING about this position.
```

It has no opinion at all - every action defaults to the same value (0), regardless of
whether one move is a brilliant winning strike and another loses instantly. It can only
ever have an opinion about a state after having visited that *exact* state before.

A neural network, by contrast, takes the board as *input* and computes an answer through
arithmetic - the same arithmetic it would use for a similar-but-not-identical position. If
it has learned "a piece in the center combined with two of my pieces in a row is usually
good," it can apply that pattern to a position it has literally never seen, because that
pattern was encoded into numbers (weights) shared across all positions, not into one
table entry per exact position.

This was directly measured in this project (see `05_results_summary.md` for the full
numbers): trained on 80% of the exactly-solved positions, tested on the 20% it never saw,
the network's average error dropped from ~0.57 (about as good as a random guess) to ~0.21
- real, substantial accuracy on positions it was never shown during training.

## 2. Representing a board as numbers: the input encoding

A neural network only understands numbers - vectors of them. So the first job is turning
a board position into a fixed-length list of numbers.

### The canonicalization trick

A naive encoding might be: "cell 0 is X, cell 1 is empty, cell 2 is O, ..." - but this
forces the network to separately learn two entire mirror-image concepts: "good positions
when I am X" and "good positions when I am O." That doubles the learning problem for no
reason, since strategically, "my pieces vs. the opponent's pieces" is what actually matters,
not the arbitrary X/O label.

So instead, the board is always encoded **from the perspective of whoever is about to
move** - "where are MY pieces" and "where are the OPPONENT's pieces," regardless of
whether "I" am X or O in this particular position. This is the same trick the real
AlphaZero uses.

### The exact encoding (19 numbers)

```
Input vector (19 numbers total):

  [0..8]   9 numbers: 1.0 if I have a piece in that cell, else 0.0
  [9..17]  9 numbers: 1.0 if the OPPONENT has a piece in that cell, else 0.0
  [18]     1 number:  0.0 if placement phase, 1.0 if sliding phase
```

Concretely, if it's O's turn and the board is:

```
Cell:    0  1  2  3  4  5  6  7  8
Piece:   X  .  O  .  X  .  O  .  .
```

Since O is about to move, "my pieces" = O's cells (2, 6), "opponent pieces" = X's cells
(0, 4):

```
input[0..8]  = [0,0,1,0,0,0,1,0,0]   (my pieces: cells 2, 6)
input[9..17] = [1,0,0,0,1,0,0,0,0]   (opponent pieces: cells 0, 4)
input[18]    = 0.0 or 1.0             (phase)
```

This is implemented in `neural_net.py`'s `encode_state()`.

### Encoding a move: the policy output space

The network's policy head needs to output a number for every possible move, in a fixed
order (neural networks have a fixed-size output, they can't grow one output slot per
game). This game has two kinds of moves - placement (pick 1 of 9 cells) and sliding (pick
a from/to pair) - so both are packed into one 90-slot vector:

```
Policy vector (90 numbers total):

  [0..8]    9 numbers: "place a piece in cell i"
  [9..89]   81 numbers: "slide from cell f to cell t", packed as index = 9 + f*9 + t
```

Most of those 81 sliding slots are never actually legal (the adjacency graph only allows
sliding between neighboring cells - see `../01_game_definition_and_state_space.md`), but
using every from/to combination keeps the encoding simple and fixed-size; illegal slots are
always masked out (forced to zero probability) whenever the policy is actually used for a
decision - see `encode_action()` / `decode_action_index()` in `neural_net.py`.

## 3. The network's architecture

```
                     INPUT (19 numbers)
                            │
                            ▼
                  ┌───────────────────┐
                  │  Hidden layer 1   │   64 neurons, ReLU activation
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │  Hidden layer 2   │   64 neurons, ReLU activation
                  └─────────┬─────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                                ▼
   ┌─────────────────┐              ┌─────────────────┐
   │   VALUE HEAD     │              │   POLICY HEAD    │
   │  1 number, tanh   │              │  90 numbers,      │
   │  range: -1 to +1  │              │  raw logits        │
   └─────────────────┘              └─────────────────┘
   "how good is this          "which of the 90 possible
    position for me?"          moves looks promising?"
```

Both heads share the same two hidden layers ("the trunk") - this is a deliberate design
choice, not just an efficiency shortcut: whatever internal representation of "board
understanding" the network builds in its hidden layers benefits both heads simultaneously,
since useful features for judging a position (like "do I have two in a row") are also
useful for deciding which move to make.

## 4. The forward pass - the actual arithmetic, step by step

Given input vector `x` (19 numbers), and weight matrices/biases for each layer:

```
z1 = x @ W1 + b1        # matrix-multiply x by the first layer's weights, add bias
a1 = ReLU(z1)            # ReLU(v) = max(0, v) - keeps positive signals, zeroes out negative ones

z2 = a1 @ W2 + b2
a2 = ReLU(z2)

value        = tanh(a2 @ W_value + b_value)      # squashes to exactly [-1, 1]
policy_logits = a2 @ W_policy + b_policy          # raw scores, one per possible action
```

**ReLU** (Rectified Linear Unit) is simply `max(0, v)` - if the pre-activation value is
negative, output 0; otherwise pass it through unchanged. It's the standard choice for
hidden layers because it's cheap to compute and its derivative (needed for training,
covered below) is trivial: 1 where the input was positive, 0 where it wasn't.

**tanh** squashes any real number into the range (-1, 1) - exactly the range this whole
project has used for "value" everywhere (the solver, the Q-table, plain MCTS), which is
why it's used specifically for the value head's final output.

**Policy logits** are left as raw, unsquashed numbers - they get converted into actual
probabilities separately (next section), because the conversion needs to account for which
moves are legal in this specific position, which varies state to state.

### From logits to probabilities: softmax

A raw logit vector like `[2.1, -0.5, 0.3, ...]` isn't yet a probability distribution (it
doesn't sum to 1, and can contain negative numbers). The **softmax** function fixes this:

```
softmax(logits)_i = exp(logits_i) / sum_j( exp(logits_j) )
```

In plain language: exponentiate every value (making everything positive, and amplifying
the gap between large and small values), then divide each by the total, so everything
sums to exactly 1 - a genuine probability distribution. A small numerical-stability trick
(subtracting the max logit before exponentiating) is used in the actual code
(`_softmax()`), which doesn't change the math, just avoids overflowing to infinity for
large inputs.

When only *legal* moves should get any probability, illegal ones are forced to `-1e9`
(effectively negative infinity) before the softmax - `exp(-1e9)` is indistinguishable from
0, so illegal moves end up with exactly 0 probability, and the remaining probability mass
is redistributed proportionally among the legal moves.

## 5. Training: what "correct" means, and how the network learns it

### The two loss functions

**Value loss - Mean Squared Error (MSE)**: if the network predicted `value` and the true
target (real game outcome, or the solver's exact answer during the isolated validation
test) is `target_value`:

```
value_loss = mean( (value - target_value)^2 )
```

Squaring the error does two things: it makes the loss always positive (errors in either
direction count as "bad"), and it penalizes large errors disproportionately more than
small ones - being off by 1.0 is 100x worse than being off by 0.1 in this loss, not 10x,
which pushes training to prioritize fixing the biggest mistakes first.

**Policy loss - Cross-Entropy**: if the network's predicted probability distribution is
`policy_probs` and the target distribution (from MCTS's own visit counts - see
`03_self_play_training_loop.md`) is `target_policy`:

```
policy_loss = -mean( sum(target_policy * log(policy_probs)) )
```

In plain language: for every move the target distribution says is good, look at how much
probability the network currently assigns it, take the log (which is very negative for
tiny probabilities and close to 0 for probabilities near 1), and average this penalty
across the batch. This specifically punishes confidently assigning low probability to a
move the search discovered was actually good.

### Backpropagation - deriving every gradient by hand

"Training" means: figure out, for every single weight in the network, whether nudging it
up or down would reduce the loss, and by how much - then take a small step in that
direction. This requires the derivative of the loss with respect to every weight, computed
via the chain rule, working backward from the output to the input (hence
"back-propagation").

This was implemented by hand (no autodiff library) in `train_step()`. Here is the full
derivation, one layer at a time, working backward:

**Step 1 - value head gradient.** Since `value = tanh(pre_value)` and
`value_loss = mean((value - target)^2)`:

```
d(loss)/d(value)     = 2 * (value - target) / N            # derivative of squared error
d(value)/d(pre_value) = 1 - tanh(pre_value)^2 = 1 - value^2  # derivative of tanh
d(loss)/d(pre_value)  = 2 * (value - target) * (1 - value^2) / N
```

(`N` = batch size, since the loss is a mean over the batch.)

**Step 2 - policy head gradient.** For softmax combined with cross-entropy specifically,
the combined gradient has a famously clean form (this cancellation is why softmax and
cross-entropy are almost always paired in practice):

```
d(loss)/d(policy_logits) = (policy_probs - target_policy) / N
```

(masked to zero for illegal actions, since those never contribute to the loss).

**Step 3 - propagate back through the output layers' weights.** For any layer computing
`output = input @ W + b`, standard matrix-calculus rules give:

```
d(loss)/d(W) = input^T @ d(loss)/d(output)
d(loss)/d(b) = sum over batch of d(loss)/d(output)
d(loss)/d(input) = d(loss)/d(output) @ W^T
```

Applied to both heads, then **summed together** at the shared trunk (`a2`), since `a2`
feeds into both heads and therefore both heads' gradients need to flow back into it:

```
d(loss)/d(a2) = [d(loss)/d(pre_value)] @ W_value^T  +  [d(loss)/d(policy_logits)] @ W_policy^T
```

**Step 4 - propagate back through ReLU.** ReLU's derivative is 1 where the pre-activation
was positive, 0 where it wasn't (undefined exactly at 0, conventionally treated as 0):

```
d(loss)/d(z2) = d(loss)/d(a2) * (z2 > 0)      # elementwise multiply by 0/1 mask
```

**Step 5 - repeat steps 3-4 backward through layer 2, then layer 1**, exactly the same
pattern, until every weight matrix and bias vector in the network has a computed gradient.

This entire chain is implemented explicitly in `train_step()` (for joint training) and
`train_value_step()` (an isolated version that only computes gradients for the value head,
used specifically for the Stage 1 generalization validation, so a meaningless policy
target couldn't pollute the shared trunk's weights during that test).

## 6. The optimizer: Adam

Once every gradient is known, the simplest possible update rule would be:
`weight = weight - learning_rate * gradient` ("plain gradient descent"). This works, but
converges slowly and is sensitive to the exact learning rate chosen. **Adam** (Adaptive
Moment Estimation) improves on this by keeping a running memory of two things for every
weight:

```
m = exponential moving average of the gradient itself       (the "momentum")
v = exponential moving average of the gradient SQUARED       (tracks how noisy/large gradients have been)
```

Updated every step as:

```
m ← beta1 * m + (1 - beta1) * gradient
v ← beta2 * v + (1 - beta2) * gradient^2
```

(`beta1 = 0.9`, `beta2 = 0.999` are standard defaults - meaning `m` mostly remembers
recent gradients, changing slowly, and `v` changes even more slowly.)

Then a **bias correction** (accounting for the fact that `m` and `v` start at zero and are
therefore artificially small in the very first few steps):

```
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
```

(`t` = step number.)

And the actual weight update:

```
weight ← weight - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

The intuition: dividing by `sqrt(v_hat)` means weights that have had large, noisy gradients
get smaller effective steps (avoiding overshooting), while weights with small, consistent
gradients get relatively larger steps (speeding up slow-but-steady progress). This is
implemented exactly as described in `_adam_step()`.

## 7. Weight initialization: why it isn't just zeros or random noise

Weights start as random numbers drawn from a specific distribution (**He initialization**):
for a layer with `fan_in` inputs, weights are drawn from a normal distribution with
standard deviation `sqrt(2 / fan_in)`.

Why not just zero? If every weight started at zero, every neuron in a layer would compute
the exact same output as every other neuron (since they'd all be doing "0 times input"),
and would keep receiving identical gradient updates forever - the network would never
differentiate one neuron's role from another's, no matter how long it trained.

Why not just any random noise? The `sqrt(2/fan_in)` scale specifically is chosen so that,
on average, the *variance* of the signal passing through a ReLU layer stays roughly
constant from layer to layer - too large, and activations blow up as they pass through
multiple layers; too small, and they shrink toward zero, making gradients vanishingly
small and training glacially slow. This is a well-established standard result for networks
using ReLU activations, implemented directly in `TicTacToeNet.__init__()`.

## 8. Saving and loading

Since training is expensive (many self-play games' worth of computation), the network's
entire state - every weight matrix, every Adam optimizer memory (`m`, `v`), and the step
counter `t` - is serialized with `pickle` (`save()`/`load()`), so a training run can be
paused, inspected, or resumed without starting over. This mirrors the same save/restore
discipline used throughout this project's Q-learning dashboards
(`../05_web_dashboard_and_engineering_challenges.md`).

## What's next

With the network itself fully specified, `02_mcts_with_puct.md` covers how it gets used
*during* a game to actually pick moves - not just evaluate positions in isolation.
