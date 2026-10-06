# Journal, entry 2: where the homework comes from

*2026-10-04, still. A model is only as good as its homework, and nobody sells homework for a brand-new exam format.*

## The problem

We invented (well, copied from CLEF) a very specific exam: a situation plus typed questions with gold answers and, ideally, probabilities. Almost no public dataset looks like that. Textbooks are essays; our exam is a scorecard. So the data story is two-pronged: *borrow* real data and reshape it, and *generate* synthetic data where the answer key is known by construction.

## Real data: honest, but the wrong shape

We pull from nine public datasets and run each through a converter. Some are easy reshapes: SNLI is literally "read two sentences, pick: entailment / neutral / contradiction." That's already a `choice` question, we just have to phrase it in the exam format. Others need creativity: an LLM-router dataset (which model answered best?) becomes a routing `choice`; a safety dataset becomes a `noul` "does this break policy?" plus a category `choice`; document QA datasets become image questions.

Every converter must obey one iron rule, learned from CLEF's own design: **the gold answer has to be derivable from what the model can actually see.** If the answer key says "urgent: true" but nothing in the situation text signals urgency, we're not teaching decisions, we're teaching coin flips. Every example is validated against the wire schema at build time, and any dataset that fails just gets skipped and logged, never crashes the build.

Real data's weakness: licenses (we record them per-source and disabled the murky ones) and volume (tens of thousands, not millions).

## Synthetic data: infinite, and the answer key is built in

The synthetic generators are my favorite part, mostly because of how backwards they feel at first.

You don't write examples. You write *worlds*. The support-ticket world has customer names, products, issue types, urgency phrases, tone phrases. Then you sample from the world: a billing-flavored ticket gets assembled from billing phrases, and *because you know you sampled "billing," you know the gold answer is team: billing.* The label isn't attached to the data; the label is *why* the data exists. Same for the image worlds: a receipt generator draws the invoice (vendor name, line items, total, a big red PAID stamp) from parameters, and the questions ("is this paid?", "does the total exceed $100?", "which vendor?") are answered by those same parameters. The model genuinely has to *look* at the stamp.

Five worlds exist: support tickets, community moderation, ops incidents + LLM routing, receipts/invoices, charts. Two of them draw their images pixel by pixel with PIL. All are seeded, so the same command produces the identical dataset on any machine, forever.

Two rules make synthetic data worth anything:

1. **Balance is enforced structurally, not by luck.** The generators cycle through gold classes to keep every answer option near-even. An imbalanced dataset teaches a model to always guess the majority and still score well, which is the silent failure mode of lazy data.
2. **States must actually encode the answer.** Every phrase pool is keyed to the class it signals. If the templates were generic ("customer needs help please"), the dataset would be unlearnable noise no matter how balanced.

## Why mix both

Real data teaches nuance and survives contact with actual human writing. Synthetic data teaches the *format* at unlimited volume and covers the image side, where licensed image+question data is scarce and legally annoying. The model needs both: format fluency from volume, judgment from reality.

## The unified format

Every example, real or synthetic, lands in one JSON shape: the request (situation + questions, exactly as a client would send it), the gold answers, and the exact string the model should have written (`assistant_text`, built by the shared prompting module, never hand-typed). One format means the data pipeline, trainer, and evaluator can all be dumb about provenance. A validation command round-trips every example through the schema and the answer builder and fails loudly on any mismatch.

About 3,000 synthetic examples build in under a minute on a laptop. The real-mix build is a heavier download but equally one command. Neither is in git; both regenerate deterministically, which keeps the repo clean and makes "delete the data, rebuild" a safe move at all times.
