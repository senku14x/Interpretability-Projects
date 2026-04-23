# Strong Readout, Weak Control: White-Box Monitoring of Attack-Conditioned Coding Models

Bottom line: I found that a simple activation probe can strongly separate attack-conditioned generations from honest generations, and it outperforms a behavioral monitor on that comparison. But the signal is partially confounded by defensive prompts, and it does not provide a useful steering handle. So I interpret this as a useful but non-clean monitoring signal, not as a clean deception feature.

---

## What this is

I started this project because I wanted to test whether a white-box activation probe could detect when an aligned coding model was internally representing something like “insert a backdoor.” I was also interested in whether that internal signal would be more useful than simply asking a strong LLM to review the generated code.

The short answer is: the probe is real, and it does beat the behavioral monitor on the main detection task. But it is not exactly what I hoped to find.

I originally hoped the probe might track something close to malicious intent or covert backdoor planning. Instead, the result looks messier. The probe seems to detect something more like “the model is reasoning about backdoor-related things.” That signal appears both when the model is asked to insert a backdoor and, to a weaker but still meaningful extent, when the model is explicitly asked to avoid backdoors.

That makes the result useful, but not clean. It is not a universal deception detector, and it is not enough to support a strong causal-control claim.

The steering result was the biggest negative result. I could push the probe score around, but that did not reliably change the model’s actual backdoor-writing behavior. So the direction is predictive, but not a useful control knob.

---

## Setup

**Model:** Qwen3-14B, loaded via TransformerLens.

I disabled thinking mode during probe extraction. Qwen3 models can emit `<think>...</think>` blocks by default, and if thinking mode is left on, the final-token activation can reflect the end of the thinking trace rather than the end of the actual prompt. That would make the measurement less meaningful for this setup.

**Dataset:** 375 examples from BigCodeBench, split evenly across three conditions:

- `honest`: solve the coding task normally
- `attack`: solve the task while covertly inserting a backdoor that passes all tests
- `defensive`: solve the task while explicitly verifying that the solution contains no backdoors

The defensive condition was important. I included it as a confounder test. If the probe were really detecting malicious intent, defensive examples should look much closer to honest examples than attack examples. That is not what happened.

**Probe method:** I used a mean-difference direction, following the basic setup from Anthropic’s *Simple Probes Can Catch Sleeper Agents*. I trained the direction on 80% of honest/attack activations at the residual stream’s final prompt token, and evaluated on the remaining 20%.

Layers tested:

- 8
- 14
- 20
- 26
- 32
- 38

**Behavioral baseline:** I used Gemini (`gemini-3-flash-preview`) as a behavioral monitor. The model was prompted as a security engineer and asked to assign a suspicion score / flag possible backdoors.

---

## Results

### The probe works on the main task

Layer 20 was the best-performing layer.

On the held-out 80/20 split:

| Comparison | AUROC |
|---|---:|
| Attack vs Honest | **0.894** |
| Defensive vs Honest | 0.574 |
| Attack vs Defensive | 0.819 |

On the full dataset, without a proper held-out split:

| Comparison | AUROC |
|---|---:|
| Attack vs Honest | **0.950** |
| Defensive vs Honest | 0.809 |
| Attack vs Defensive | 0.851 |

The full-dataset numbers are useful for comparison, but I do not treat them as the main evaluation because they are not a clean held-out result.

The main result is still clear: the probe separates attack-conditioned examples from honest examples very strongly.

It also outperformed the behavioral monitor on the same attack-vs-honest comparison:

| Method | Attack vs Honest AUROC |
|---|---:|
| Gemini behavioral monitor | 0.684 |
| Activation probe, layer 20 | **0.950** |

The two methods also caught different examples. Out of 125 attack examples:

| Category | Count |
|---|---:|
| Both catch | 23 |
| Only probe catches | 39 |
| Only behavioral monitor catches | 25 |
| Both miss | 38 |

This suggests the probe is not just duplicating the behavioral monitor. It finds many attack-conditioned examples that the output-only monitor scores as clean.

---

## The central confound

The main problem is the defensive condition.

If the probe were detecting malicious intent, defensive examples should look similar to honest examples. But on the full dataset, defensive-vs-honest AUROC was 0.809. That is too high to ignore.

My current interpretation is that the probe is not cleanly detecting “the model intends to insert a backdoor.” It is at least partly detecting a broader internal regime where the model is thinking seriously about backdoor-related concepts.

That regime appears in both attack and defensive prompts:

- attack prompts: “insert a hidden backdoor”
- defensive prompts: “make sure there is no hidden backdoor”

So both can activate backdoor-related internal representations.

At a stricter threshold, the probe still looks useful as a monitoring signal. Using a threshold at the top 50% of the attack distribution:

| Condition | Probe > threshold | Behavioral monitor > 50 |
|---|---:|---:|
| Honest | 0.0% | 1.6% |
| Attack | 49.6% | 38.4% |
| Defensive | 0.8% | 0.0% |

This thresholded result is more encouraging. It suggests that, at a conservative cutoff, the probe can flag many attack-conditioned examples while keeping false positives low.

But the full-distribution AUROC results still matter. They show that the internal signal is not cleanly “malicious intent.” It is probably a mixture of several things:

1. a backdoor-reasoning mode,
2. attack-prompt semantics,
3. possible covert-special-casing behavior,
4. general security-related reasoning.

So I would not describe this as a monosemantic deception feature.

---

## The contrast-pair probe did not fix the issue

I also tried a more generic contrast-pair probe inspired by the sleeper-agents methodology.

The idea was to avoid task-specific vocabulary. Instead of training the direction directly on honest-vs-attack coding prompts, I computed a generic deception/honesty direction from contrast pairs and then evaluated it on the real coding-task activations.

This helped test whether the original probe was just picking up words like “backdoor.”

The best result was around:

| Probe | Attack vs Honest AUROC |
|---|---:|
| Generic contrast-pair probe | ~0.726 |

It was weaker than the task-specific probe, and it was still somewhat confounded by defensive examples:

| Comparison | AUROC |
|---|---:|
| Defensive vs Honest | ~0.63 |

So the confound is not purely vocabulary. The generic direction still picks up some related structure, but it is weaker and still not clean.

---

## Steering failed

This was the experiment I most wanted to work, and it did not.

I tried subtracting the layer-20 probe direction during generation using activation steering. I tested different steering strengths:

```text
alpha ∈ {0, 1, 2, 5, 10}
````

The probe scores dropped strongly as alpha increased:

| Alpha | Mean probe score |
| ----: | ---------------: |
|     0 |            12.97 |
|    10 |             2.99 |

So the representation was being changed.

But the actual backdoor rate barely moved:

| Alpha | Backdoor rate |
| ----: | ------------: |
|     0 |         33.3% |
|     1 |         36.7% |
|     2 |         33.3% |
|     5 |         36.7% |
|    10 |         30.0% |

This was the key negative result.

I could edit the probe score, but that did not reliably change whether the model wrote backdoors. So the learned direction is probably not the causal feature responsible for the behavior. It looks more like a correlated readout.

In other words: high AUROC did not imply a useful steering direction.

There is also an instrumentation caveat. In the pilot steering run, the activation-capture hook was likely firing before the steering hook. That means the logged probe scores may reflect pre-steering activations rather than the actual post-steering activations used during generation.

This does not change the behavioral result, because the generated code still showed little movement in backdoor rate. But it does mean the score-vs-alpha curve should be read cautiously.

---

## Logistic regression and PCA

I also tried logistic regression on the full activation set.

It produced AUROC = 1.0 at all layers. I do not treat this as meaningful evidence. With roughly 100 training examples and a 5120-dimensional representation, in-sample linear separation is easy. This result only shows that the classes are separable in that setup. It does not support a mechanistic claim.

I also looked at PCA. The top principal components showed some visible condition structure, which is consistent with the probe results, but I do not treat PCA as strong evidence on its own.

---

## Reasoning correlation

I also checked whether attack examples where the model explicitly discussed backdoor planning in its `<think>` block received higher probe scores.

The result was mixed.

In several layers, examples with explicit backdoor-planning text actually had lower mean probe scores than examples without explicit planning text. I do not have a clean interpretation of this.

The most likely explanations are:

1. the correlation between explicit reasoning text and the probe direction is weak,
2. my keyword-based method for identifying “explicit backdoor planning” was noisy,
3. the probe may be tracking prompt-conditioned internal state more than visible reasoning content,
4. or some combination of the above.

I would not make a strong claim from this part of the project.

---

## What I learned

### The useful part

Residual-stream activations contain a real signal that the behavioral monitor misses.

The activation probe found attack-conditioned examples that Gemini, acting as a code reviewer, often scored as clean. This supports the idea that white-box monitoring can add information beyond output-only behavioral monitoring.

### The disappointing part

The signal is not clean.

It does not cleanly separate “planning to insert a backdoor” from “thinking carefully about whether there is a backdoor.” Defensive prompts partially activate the same direction.

That makes the probe useful as a warning signal, but not as a clean intent detector.

### The most important negative result

Steering along the probe direction did not reliably change behavior.

This is the result I would emphasize most. A direction can be highly predictive and still not be the thing that causally controls the behavior. In this setup, the probe was much better as a readout than as a control mechanism.

---

## What this project establishes

I think this project supports the following modest claims:

* White-box probing can outperform a single behavioral monitor on attack-vs-honest AUROC.
* In this setup, the activation probe reached 0.950 AUROC on attack-vs-honest, compared with 0.684 for the Gemini behavioral monitor.
* The probe and behavioral monitor catch different subsets of attack-conditioned examples.
* Middle-layer residual-stream activations contain a distinct signal associated with attack-conditioned generation.
* This signal is useful for monitoring, but it is not cleanly interpretable as malicious intent.

---

## What this project does not establish

This project does not show:

* a universal deception direction,
* a monosemantic backdoor-intent feature,
* a clean malicious-intent detector,
* a working causal control method,
* generalization to other models,
* generalization to other datasets,
* or strong evidence that probe-based steering can prevent backdoor behavior.

The defensive confound rules out the cleanest interpretation of the probe. The steering result rules out the strongest control interpretation.

---

## Limitations

### Sample size

The dataset has 125 examples per condition. That is enough to see real patterns, but not enough for strong generalization claims.

### One model

All experiments here used Qwen3-14B. I do not have evidence that the same probe behavior appears in other model families.

### Weak ground truth

The attack condition is prompt-conditioned, not semantically verified in a perfect way. Not every attack-conditioned output necessarily contains a real backdoor. Some may contain suspicious code or edge cases that are not true backdoors.

The dataset uses test-passage as the practical criterion, not a full semantic backdoor verifier.

### Behavioral monitor was narrow

I used one Gemini reviewer prompt as the behavioral baseline. That is a useful comparison, but it is not an optimized behavioral-monitoring system. A stronger ensemble monitor might reduce the apparent advantage of the probe.

### No OOD evaluation

The results are local to this BigCodeBench setup. I did not test whether the probe generalizes to other attack datasets, other coding benchmarks, or existing control arenas.

### Steering instrumentation caveat

In the pilot steering sweep, the activation capture likely logged pre-steering values. So the internal probe-score curve is somewhat confounded.

The behavioral result remains the main point: the generated backdoor rate barely changed.

---

## How to run

```bash
pip install -r requirements.txt

# Pre-experiment sanity checks
python sanity_check.py

# 1. Generate dataset
python generate_dataset.py

# 2. Extract activations
python extract_activations.py

# 3. Train probe and run analysis
python run_probe.py

# 4. Run generic contrast-pair probe
python run_contrast_probe.py

# 5. Run causal steering experiment
python run_steering.py

# 6. Run behavioral monitor
GEMINI_API_KEY=your_key python run_behavioral_monitor.py
```

Outputs are saved in:

```text
data/arm_a_final/results/
```

The run requires an A100 80GB GPU or equivalent for model loading. BigCodeBench loads from HuggingFace. No other major data downloads are needed.

---

## Repo structure

```text
src/
  model.py                    -- load Qwen3-14B via TransformerLens
  extract.py                  -- residual-stream activation extraction
  probe.py                    -- mean-difference probe utilities

sanity_check.py               -- pre-experiment checks
generate_dataset.py           -- generate honest/attack/defensive examples via vLLM
extract_activations.py        -- extract and save activations for all conditions
run_probe.py                  -- main probe analysis and plots
run_contrast_probe.py         -- generic deception/honesty contrast-pair probe
run_steering.py               -- causal steering experiment
run_behavioral_monitor.py     -- Gemini behavioral baseline and probe-vs-monitor comparison
requirements.txt
```

---

## Citation / acknowledgements

Probe methodology follows: *Simple Probes Can Catch Sleeper Agents* Lindsey et al., Anthropic, 2024.

Evaluation dataset: [BigCodeBench](https://huggingface.co/datasets/bigcode/bigcodebench) Zhuo et al., 2024.

Model:[Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) Qwen Team, Alibaba, 2025.

