# Journal, entry 4: the day training made the model worse

*2026-10-04. My favorite story from this project, because it's about the bug you can't see.*

## The setup

We had just built the eval harness: a scorer that takes held-out exam questions, runs them through the actual serving engine, and grades the answers against gold labels, with a Brier score for the probabilities. First measurement: the untrained base model gets 41.2% overall and is confidently wrong a lot (mean confidence ~0.8). Bad, as expected. Fine.

Then we trained: a 40-step LoRA run. Training looked *beautiful*. Loss fell from 0.96 to 0.08. Token accuracy hit 96.6%. Every dashboard green.

Then the eval: **36.4%.** Training made the model *worse than doing nothing*, while every training metric said otherwise.

## Step one: distrust the middleman

Two suspects: the model (it genuinely learned the wrong things), or the engine (the model learned fine and our scoring misreads it).

The tiebreaker experiment: bypass the engine entirely. Take the trained model, let it *freely generate* its answer sheet the way a chatbot would, and grade the generated answers. If generation is good but engine-scoring is bad, the engine is lying. If generation is also bad, the model is.

Result: generation scored ~44%. So the model was mediocre, but the engine was *also* broken. Two separate problems wearing one costume.

## Bug one: we forgot to show it the instructions

The training data included a system message, the little instruction that says "read the state and schema, decide every field jointly." The serving engine, built separately, never sent it.

Humans do this constantly: you memorize answers *in context*, then someone quizzes you with the context page missing and you're suddenly guessing. The model had learned to answer questions *with those instructions present*. Same questions, no instructions, measurably different behavior. One message added back: a few points recovered.

## Bug two: reading the answer one word early

This is the subtle one, and it's a story about how text becomes numbers.

Models don't read letters. They read *tokens*, chunks of text carved up by a tokenizer, and the carving depends on what's next to the chunk. Our scoring reads the model's preferences at the exact spot where the answer goes. For yes/no questions the answer slot is like `..."noul":true`, so we measured how much it wanted `true` vs `false` at that spot. Aligned, correct, worked.

For pick-one questions, the answer slot is `"choice":"billing"`. We scored the word `billing` at that position. But in everything the model ever trained on, `billing` never appears there as a bare word; what appears is `"billing` *with its opening quote fused on*, because that's how the tokenizer carves it in context. We were effectively reading the model's preference for the answer *one position before the answer*, where every option's first token is the same opening quote. For multiple-choice, our "probabilities" were near-noise.

The fix is a small function (`option_token_spans`) that tokenizes each option's continuation *jointly with the context*, exactly as training did, finds where the options genuinely diverge, and measures only from there. We wrote unit tests that pin this down: the tokenized option sequence must decode back to the exact training string, and every option's span must be distinct. This bug class, "train/inference tokenization drift," is now structurally impossible to reintroduce silently.

(There was also a humbler crash along the way: a fresh list handed to a function that expected a tensor. Not every war story is glorious.)

## Then: train properly

With the grader fixed, the real experiment: 2 full epochs instead of a token 40 steps. Result on held-out questions:

- Overall: 41.2% (base) → **58.2%**
- Yes/no: 60.9% → **81.3%**
- Brier: 0.610 → **0.457**

Still not "done" (pick-one at 39.6% is the weak link, and that's a data problem, not a steps problem), but now the numbers are *true*. Every one of them is measured by the same path real traffic takes, on data the model never trained on.

## The lessons, stated plainly

1. **Training metrics measure conformity, not skill.** A model can nail the answer-sheet format and flip coins on the answers. Loss and token accuracy are necessary, never sufficient.
2. **Build the grader before the training.** If we'd trained for hours on a GPU before evaluating end-to-end, we'd have burned the whole budget baking in a bug that took ten minutes to find once measured.
3. **When two subsystems disagree, bypass the middleman.** Comparing generated answers against scored answers isolated the bug in one experiment.
4. **Tokenization boundaries are load-bearing.** "The same text" is not "the same tokens." Any place where two subsystems each tokenize their half is a place they can silently disagree.
5. **Pins, not promises.** The fix isn't the code change; it's the unit test that makes the wrong version fail loudly forever.

The eval harness earned its keep on day one by catching us red-handed. That's the best thing you can say about any measurement tool.
