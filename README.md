# Interpretability Projects

This repo contains a few mechanistic interpretability / AI safety projects I have been working on.

Most of these are not polished paper-style artifacts. They are closer to research logs: I tried an idea, ran the experiments , wrote down what worked, what failed, and what I think the result actually supports.

The common theme is that I am interested in whether model internals can give us useful signals about model behavior, and also in where those signals break down.

---

## Projects

## Strong Readout, Weak Control

This project looks at whether a simple activation probe can detect attack-conditioned coding behavior better than an output-only behavioral monitor.

The setup was: take Qwen3-14B, generate coding solutions under honest / attack / defensive conditions, extract residual-stream activations, and train a mean-difference probe to separate attack-conditioned generations from honest ones.

The probe worked pretty well on the main comparison. It separated attack from honest generations much better than the behavioral monitor I used.

But the result was not as clean as I originally hoped.

The defensive condition was the important confound. Defensive prompts also activated the direction to some extent, which means the probe was probably not detecting something clean like “malicious intent.” It looked more like a broader backdoor/security-reasoning signal.

The steering result was the biggest negative result. I could move the probe score around, but that did not reliably change whether the model actually wrote backdoors.

My takeaway:

> The direction is useful as a readout, but not as a control knob.

So I think this project is a decent example of why high probe AUROC should not automatically be treated as mechanistic understanding.

---

## SAE-Mediated Cross-Architecture Transfer for Activation Oracles

This was a negative-result project.

The question was:

> If a Gemma Activation Oracle is trained to read Gemma activations, can I make it read Llama activations better by routing them through SAE feature spaces?

The simple idea was:

```text
Llama hidden state
→ Llama SAE features
→ map into Gemma SAE feature space
→ Gemma SAE decode
→ Gemma Activation Oracle
````

I tested that against a strong raw hidden-state baseline using PCA + ridge regression.

The raw hidden-state baseline won pretty clearly.

The full SAE-mediated path stayed far below the raw baseline, even after trying wider SAEs and different variants. The most useful ablation was that skipping the Llama SAE encode step helped a lot, which made the source-side SAE look like the main bottleneck.

The result does not show that SAEs can never help with cross-architecture transfer. It only shows that this simple version of the idea did not work in this setup.

My takeaway:

> Off-the-shelf SAE feature spaces did not automatically make cross-architecture transfer easier. In this case, a boring regularized raw hidden-state map was much stronger.

I still think the project was useful because it made the failure mode clearer instead of just saying “SAEs did not work.”

---

## The Internal Geometry of Strategic Reasoning in Thinking Language Models

This is an active project on whether strategic reasoning sub-behaviors have measurable geometry inside a reasoning model.

I used DeepSeek-R1-Distill-Qwen-14B and built a multi-label annotation pipeline for reasoning segments. The labels include things like opponent modeling, payoff analysis, deduction, strategic uncertainty, and equilibrium identification.

The main thing I looked at was whether difference-of-means directions for these behaviors had stable structure across layers, and whether intervening on those directions changed the model’s reasoning behavior.

The opponent-modeling direction was the most interesting one. It showed consistent geometry and intervention effects, including transfer within the Qwen-14B family.

But the result is also messy.

Positive steering increased opponent-modeling-labeled reasoning, but projection-out ablation also increased it. That makes the direction seem non-atomic. It is probably not a clean “opponent modeling feature.” It may be involved in transitions between reasoning modes, or it may be entangled with broader reasoning structure.

My takeaway so far:

> There is real structure here, but it is not a clean semantic feature, and steering does not mean “improving reasoning.”

This one is still in progress. I am still working on human evaluation, better robustness checks, and possibly testing other model families.

---

## Overall takeaways

Across these projects, the pattern is roughly:

* probes can find real signals,
* real signals can still be confounded,
* good readouts are not necessarily good control handles,
* negative results are still useful if they identify where the simple story breaks,
* and activation-space directions are often messier than the names we give them.

I am mostly using this repo to build research taste around interpretability experiments: how to set up controls, how to notice confounds, how not to overclaim, and how to write down what actually happened.

