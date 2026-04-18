Disclaimer: This project was done with me just being stubborn and ignoring the obvious early experiment signals that this wasnt going to work out at all lmao and I still tried to complete it anyways since some time had been invested.

## Quick summary

This repo is a mini research project on **cross-architecture transfer for Activation Oracles**.

The basic question was: if a Gemma Activation Oracle is supposed to read Gemma activations, can I make it read **Llama** activations better by routing them through **SAE feature spaces** instead of just doing a direct raw hidden-state projection?

I tried a bunch of versions of that idea:
- a strong raw baseline using standardization + PCA + ridge
- a full SAE path: Llama SAE → feature-space map → Gemma SAE decode
- decoder-side fixes like ReLU + TopK before Gemma decode
- wider SAEs
- ablations that remove either the source-side SAE encode step or the target-side SAE decode step
- a cross-task generalization check
- a simple KNN feature-matching baseline

### What worked

The strongest thing in the whole project was the **raw hidden-state baseline**.  
A simple PCA + ridge map from Llama hidden states into Gemma hidden space worked surprisingly well on the tasks I tested.

Also, the Gemma SAE itself was not broken: within architecture, the oracle only lost a small amount of performance when I round-tripped genuine Gemma activations through the Gemma SAE.

### What didn’t work

The full SAE-mediated path consistently lost to the raw baseline.

Even after adding sparsity fixes like ReLU + TopK, the SAE route still stayed well below the raw hidden-state projection. Wider public SAEs also didn’t really rescue it.

The most useful ablation was this:
- if I **skip the Llama SAE encode step** and map raw Llama hidden states directly into **Gemma feature space**, then decode with the Gemma SAE, performance gets much closer to the raw baseline
- if I keep the **Llama SAE features** and skip the Gemma decoder instead, performance is still weak

So the main bottleneck seems to be the **source-side Llama SAE representation**, not the Gemma decoder.

### The caveat

The raw baseline looked strong **within task**, but it didn’t generalize well **across tasks**.  
So I don’t think the raw method learned some deep, task-general cross-architecture translator. It looked more like a strong task-specific alignment.

### What I think the project showed

The cleanest takeaway is:

> In this Llama -> Gemma setting, with off-the-shelf public SAEs and the linear mappings I tested, SAE-mediated transfer was not better than a strong regularized raw hidden-state baseline.

More specifically:
- the full SAE path underperformed
- the source-side SAE encode step looked like the main failure point
- the target-side Gemma SAE decode step was not the core issue
- and the raw baseline, while strong, was mostly task-specific rather than broadly general

So this repo is basically a **careful negative result**.  
It doesn’t show that SAEs can never help with cross-architecture transfer. It just shows that the simple version of that story — “take public SAEs from both models, align them linearly, and decode” — did **not** work here.


