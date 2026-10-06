# Journal, entry 1: why this thing is shaped like this

*2026-10-04. Everything in this journal happened in one day, which either says something about modern tooling or about my sleep schedule.*

## The one-sentence version

jev-smol is a tiny AI that doesn't chat. You hand it a situation and a short exam you invented on the spot, and it fills in the exam with odds instead of essays.

## Why not just use Cloudflare's CLEF?

Cloudflare released CLEF, and it's free and open and good. It's also huge: 27 billion parameters for the real one, 9 billion for the fast one. You talk to it over the network, per request, per token. That's fine for occasional questions. It's a bad fit if you want a decision made *inside* something: on a laptop, on an edge box, inside a plugin, or a million times a day inside a hot loop where network calls and per-token bills add up.

The whole reason this project exists is the bottom of that market. Nothing in the Jev/CLEF ecosystem runs small. We wanted: unplug the network, keep the plug shape. So the rule was: same API contract, 500 million parameters, runs on a laptop.

The "same API contract" part mattered more than it sounds. The Jev "System One" format (a situation called `state`, plus typed questions: yes/no ones called `noul`, pick-one called `choice`, rate-on-a-scale called `score`) is a published, stable wire format. If we speak it exactly, anything built to talk to CLEF can talk to us by changing one URL. Compatibility is the moat you get for free by discipline instead of compute.

## Why a vision-language model when most questions are text?

Because "embedded text + image" was the goal, and SmolVLM2-500M is the smallest credible model that does both well. It's really two models glued together: a small "eye" (SigLIP, ~90M parameters) that turns a picture into tokens a language model can read, and a small language brain (SmolLM2, ~360M). Total about half a billion parameters, Apache-2.0 licensed, fits in 2 GB of memory.

The trade: a 360M brain is never going to judge a borderline moderation call like a 27B one. We accepted that ceiling on purpose. The bet is that most real decisions (route this, prioritize that, is this paid, which team) are pattern recognition with clear cues, not philosophy.

## The strangest design choice: it never actually "writes" an answer

Most people would build this as: ask the model the question, let it generate a JSON answer, parse the JSON, hope the JSON is valid.

We don't do that. We do what CLEF does, and it's the cleverest part of the whole design.

When a language model writes text, at every single position it holds a probability distribution over every possible next token: how much it "wants" each word. Generation just samples from that. But you can also *read* it directly. So instead of letting the model write `"choice": "billing"` and hoping, we set up the answer sheet so the model's pen is resting exactly on the blank where the answer goes, and then we measure how strongly it wants each legal option at that exact spot. Softmax those numbers and you get honest probabilities: 88% billing, 12% technical. The answer is mathematically guaranteed to be one of the legal options, because we only ever measured the legal options. No parsing failures, no hallucinated fields, no essays.

The catch (which will matter in entry 4): this only works if the exam sheet the model *trained* on is character-for-character identical to the one we *score* on. Every piece of this project, the data builders, the trainer, and the server, build that exam sheet by calling one shared module (`prompting.py`). Single source of truth. The day that discipline slipped, the model got worse instead of better, but that's a later entry.

## Why LoRA, and why the eye stays frozen

Training all 500M parameters means writing a new 2 GB brain and needing serious memory to do it. LoRA is the standard trick: freeze the whole brain, and learn a tiny set of "margin note" corrections (8.7M parameters, 1.7% of the model, ~35 MB) that nudge its behavior. You get most of the effect for 3% of the cost, and you can merge the notes back into the book whenever you want.

And crucially: we only put margin notes on the *language* half. The eye (vision tower) has its own q/k/v projection layers with the same names as the brain's, which is a classic trap: an uncareful LoRA config quietly trains both. Our training script counts what it's about to train and refuses to start if even one vision module is included. Defensive code beats good intentions.

## The shape, summarized

A frozen 500M see-and-read brain, a strict shared exam format, scoring-by-peeking instead of generation, a 35 MB stack of margin notes as the only trained artifact, and a wire contract copied exactly from a published spec so we're a drop-in replacement the moment the notes are good enough.
