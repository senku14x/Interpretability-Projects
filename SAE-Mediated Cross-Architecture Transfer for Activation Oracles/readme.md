Disclaimer: This project was done with me just being stubborn and ignoring the obvious early experiment signals that this wasnt going to work out at all lmao and I still tried to complete it anyways since some time had been invested.

## Quick summary

This repo is a mini research project on **cross-architecture transfer for Activation Oracles**.

The questio was simple:

> If a Gemma Activation Oracle is trained to read Gemma activations, can I make it read **Llama** activations better by routing them through **SAE feature spaces** instead of just doing a direct raw hidden-state projection?

I tried a bunch of versions of that idea. The short answer is:

> **No. In this setup, a strong raw hidden-state baseline beat every SAE-mediated version I tested.**

### The setup

- **Oracle model:** Gemma-2-9B-IT Activation Oracle
- **Source model:** Llama-3.1-8B-Instruct
- **SAEs:** Gemma Scope (16k and 131k) and Llama Scope (8x and 32x)
- **Tasks:** SST-2 and AG News
- **Extraction rule:** single token at offset `-3` from the end of sequence  
  (this is a limitation and one reason I shelved the project rather than pushing it further)

### First sanity check: the Gemma SAE itself was fine

Before doing any cross-architecture transfer, I checked whether the Gemma SAE path was already too lossy for the Gemma oracle.

Results:

- **SST-2**
  - genuine Gemma activations: **0.929**
  - Gemma SAE round-trip: **0.920**
  - zero steering: **0.464**
  - shuffled steering: **0.479**

- **AG News**
  - genuine Gemma activations: **0.779**
  - Gemma SAE round-trip: **0.752**
  - zero steering: **0.524**
  - shuffled steering: **0.499**

So the within-architecture SAE tax was only about **1–3%**. That means the Gemma SAE itself was *not* the main problem. :contentReference[oaicite:2]{index=2}

### The raw hidden-state baseline worked well

The strongest baseline in the project was just:

- standardize activations
- PCA to 256 dims
- ridge regression
- reconstruct into Gemma hidden space
- feed to the Gemma oracle

Results:

- **SST-2**
  - genuine Gemma: **0.931**
  - raw PCA+ridge: **0.804**
  - shuffled-pair control: **0.444**
  - random control: **0.497**

- **AG News**
  - genuine Gemma: **0.799**
  - raw PCA+ridge: **0.697**
  - shuffled-pair control: **0.476**
  - random control: **0.475**

So the raw aligned map was clearly real. It beat both shuffled and random controls by a large margin. :contentReference[oaicite:3]{index=3}

### The full SAE-mediated path lost badly

The full path was:

**Llama hidden → Llama SAE → feature-space linear map → Gemma SAE decode → Gemma oracle**

Best results I got there:

- **full SAE path, best TopK**
  - SST-2: **0.569**
  - AG News: **0.581**

I also tried wider SAEs:

- **131k Gemma SAE**
  - SST-2: **0.576**
  - AG News: **0.547**

- **32x Llama SAE**
  - SST-2: **0.592**
  - AG News: **0.567**

So none of those came close to the raw baseline:
- raw was **0.804 / 0.697**
- full SAE variants stayed around **0.55–0.59** :contentReference[oaicite:4]{index=4}

### The most useful finding: the problem was mostly on the source side

The cleanest ablation was:

- **C3b / no source-side encode**  
  Llama hidden → map directly into Gemma **feature** space → Gemma SAE decode

versus

- **C3c / no target-side decode**  
  Llama SAE features → map directly into Gemma **hidden** space

Results:

- **SST-2**
  - full SAE path: **0.569**
  - no Llama encode: **0.761**
  - no Gemma decode: **0.561**

- **AG News**
  - full SAE path: **0.581**
  - no Llama encode: **0.703**
  - no Gemma decode: **0.591**

This was the clearest result in the whole project.

It says the main bottleneck was the **Llama SAE encode step**, not the Gemma decoder. If I skipped the Llama SAE and only used the Gemma SAE on the target side, performance got much closer to the raw baseline. :contentReference[oaicite:5]{index=5}

### Why the full SAE path looked broken

One big issue was that the projected Gemma feature vectors were completely unlike native Gemma SAE features.

For the full 16k SAE path:
- projected feature effective L0 was around **8000–9000**
- native Gemma 16k SAE L0 was around **192**
- projected features also had about **16–22% negative entries**

So I was basically decoding dense, signed vectors through a decoder that was trained on sparse nonnegative JumpReLU-style features. I tried fixing this with ReLU + TopK before decode, and it helped a bit, but not nearly enough to match raw hidden-state transfer. :contentReference[oaicite:6]{index=6}

### The 32x Llama SAE didn’t rescue it

I also tested a wider public Llama SAE.

But it still had native L0 only around **39**, compared to about **35** for the 8x version, and performance was still bad:

- **A_32x_topk165**
  - SST-2: **0.592**
  - AG News: **0.563**

So “just use a wider public source SAE” was not enough. :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

### The raw baseline did not generalize across tasks

This was the biggest caveat on the raw result.

If I trained the raw projection on one task and tested on the other:

- train **SST-2**, test **AG News**
  - aligned: **0.468**
  - shuffled: **0.484**

- train **AG News**, test **SST-2**
  - aligned: **0.587**
  - shuffled: **0.547**

So the raw baseline was strong **within task**, but not really task-general. It looked more like a strong task-specific alignment than a universal cross-architecture translator. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

### Other things that didn’t work

- **KNN feature matching**
  - basically near chance
  - one-feature-to-one-feature matching between independently trained SAEs did not carry enough oracle-relevant signal

- **Dense decode through Gemma SAE**
  - poor
  - TopK helped only modestly

- **Wider Gemma SAE (131k)**
  - did not materially rescue the full path

### My actual takeaway

The strongest defensible takeaway is (in my opinion):

> In this Llama -> Gemma setting, with off-the-shelf public SAEs and the linear mappings I tested, SAE-mediated transfer was worse than a strong regularized raw hidden-state baseline.

More specifically:

- the **full SAE path lost**
- the **source-side Llama SAE encode** looked like the main bottleneck
- the **Gemma SAE decoder** was not the main failure point
- and the **raw baseline**, while strong, looked mostly **task-specific** rather than broadly general

So this repo is basically a **negative result**. (grass is green type of project)

It does **not** show that:
- SAEs can never help cross-architecture transfer
- feature-space mediation is impossible
- raw hidden-state alignment solves the problem generally

It only shows that the simple story —
**“take public SAEs from both models, linearly align them, decode, and query the oracle”**
— did **not** work here. :contentReference[oaicite:11]{index=11}
