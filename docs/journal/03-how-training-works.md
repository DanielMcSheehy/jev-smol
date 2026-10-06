# Journal, entry 3: how the training actually works

*2026-10-04. The short version: it's flashcards for a robot, with a very picky grader.*

## What we're really doing

"Fine-tuning" sounds mystical. Concretely: we show the model thousands of completed exam sheets (situation, questions, correct answers) and nudge it, a tiny bit each time, toward producing the right answers itself. That's it. The nudging is calculus, but the concept is flashcards.

Three choices in here actually matter, so here they are in plain language.

## Choice 1: only grade the answers

Every training example contains the exam *and* the answer. A naive setup grades the model on reproducing the whole thing, questions included. That wastes most of the effort teaching it to copy question sheets it will never have to write.

So we split each example at the boundary: everything up to and including the exam is "context" (not graded), the answer is "completion" (graded). The technical term you'll see in the code is prompt-completion format with completion-only loss. In practice the model spends 100% of its learning capacity on the only behavior we want: filling in the scorecard.

## Choice 2: LoRA, or margin notes instead of a rewrite

Full training rewrites all 516M parameters: 2 GB of weights to store, plus gradient copies and optimizer state, roughly 12 GB of working memory for a "small" model.

LoRA keeps the book and learns margin notes. At its heart every transformer layer is mostly big matrix multiplications; LoRA learns a pair of very skinny matrices (rank 16 here) whose product approximates the *change* you'd have made. 8.7M trainable parameters instead of 516M, about 1.7%. The learning rate can be aggressive (1e-4 to 2e-4, vs 2e-5-ish for full training) because you're only steering, not rebuilding.

Two guards in our code, both born of real failure modes in the field:

- The LoRA attach list is a *regex that only matches the language half*, and the script counts its matches before training and aborts if a single vision module got included. The eye and the brain have identically-named layers (`q_proj`, `v_proj`, ...), and without this check you'd silently train the eye too.
- `max_length=None`. Image examples expand to over a thousand tokens, and default truncation would quietly chop the image tokens off the end of examples, training the model on corrupted inputs. Silence is the enemy; we'd rather drop overly long examples explicitly and log the count.

## Choice 3: epochs, learning rate, and the cosine schedule

An *epoch* is one pass over all the flashcards. Learning rate is step size: too big and the model oscillates, too small and it never arrives. We use a cosine schedule: start fast, decay smoothly to zero, so the final steps are fine-tuning polish rather than lurches.

The 40-step experiment vs the 2-epoch run turned out to be a perfect demonstration of why "steps" alone is a meaningless number. 40 steps at batch 16 saw each example less than half a time and, worse, the learning rate had already decayed to zero: training was *over* while barely started. The same setup for 2 full epochs (226 steps, 35 minutes on a laptop GPU) tripled the effective training and moved accuracy from 43% to 58%. The curve hadn't flattened.

## What "loss" and "token accuracy" do and don't tell you

Training prints two numbers constantly. Loss is the grader's unhappiness (lower better). Token accuracy is the fraction of answer tokens predicted correctly.

Here's the trap, and it cost us an afternoon: the answer sheet is mostly boilerplate. `{"team":{"type":"choice","choice":"` is the same in every example. A model can hit 96%+ token accuracy by nailing the boilerplate and still flip a coin on the actual *value*. Both numbers look great while the decisions stay bad. The only metric that means anything is decision accuracy on *held-out* data, scored by the same engine that serves traffic. That's why the eval harness (entry 4) exists and why no training run in this project is considered done until it has been evaluated end-to-end.

## Where the images hurt

The vision processor chops every image into 16 patches plus an overview: 17 chunks, 64 tokens each, 1,088 tokens per image, regardless of how small the picture is. A text example costs ~700 tokens; an image example ~2,600. On a laptop that makes image examples roughly ten times more expensive, which is why our early runs were text-only and the image runs are waiting for a real GPU. (The splitting can be disabled for an 8x saving, as long as training and serving agree, which is noted in the GPU runbook.)

## The honest scorecard so far

On held-out synthetic questions, measured by the serving engine:

| Model | Overall | yes/no | pick-one | rate-scale | Brier |
| --- | --- | --- | --- | --- | --- |
| Untaught base | 41.2% | 60.9% | 31.3% | 26.4% | 0.610 |
| + 40 steps | 43.0% | 46.9% | 37.5% | 43.4% | 0.533 |
| + 2 epochs | 58.2% | 81.3% | 39.6% | 47.2% | 0.457 |

Brier, if you haven't met it, grades *probabilities* like a weather forecaster: confident-and-wrong is punished hard, honest uncertainty is cheap. It falling from 0.61 to 0.46 while yes/no accuracy jumped 20 points is the real signal that learning is happening, not just memorizing.
