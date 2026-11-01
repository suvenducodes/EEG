# ANSWERS

## Part 1

### Task 1: Bugs
Before (all bugs, channels=all, seed 0): test_acc = 0.667 (n_test = 84)
After (all fixes, subject-wise split, seed 0): test_acc = 0.548 (46/84, not significant vs chance, p about 0.2)

| # | File | What's wrong | Effect on accuracy |
|---|------|--------------|--------------------|
| 1 | data.py | RUNS 5/9/13 are executed movement, not imagery (6/10/14) | Up: real movement is easier to decode |
| 2 | data.py | 1-40 Hz filter lets in drift and EOG; mu/beta is 8-30 Hz | Up: non-neural signal helps |
| 3 | train.py | Random trial split mixes subjects across train/test | Up: not cross-subject |
| 4 | train.py | Z-score stats include test data | Up slightly: leakage |
| 5 | train.py | Best epoch picked on test accuracy | Up: max over noisy epochs |
| 6 | erd.py | Baseline window (0-0.5 s) is after the cue | ERD underestimated |

Honest result: about 55% cross-subject, indistinguishable from chance on one seed and 84 test trials.

### Task 2: ERD
At C3, Cz and C4 both classes show a large positive spike (+50 to +80%) about 0.2 s after the cue, then a weak noisy desynchronisation of about -10 to -25%. Hands and feet curves overlap heavily, so there is no clear hand-vs-foot difference. I expected stronger ERD at C3/C4 for hands and at Cz for feet; the plot does not clearly show it. Cz is weak because the foot area is deep in the interhemispheric fissure, so its signal is small and mixes with neighbours; averaging 10 subjects with about 42 trials each is also noisy. The early spike is a cue-evoked transient, not imagery ERS.

### Task 3: Frontal vs all
| Filter | frontal (8 ch) | all (64 ch) |
|---|---|---|
| 8-30 Hz (fixed) | 0.500 | 0.548 |
| 0.5-40 Hz (wide) | 0.643 | 0.619 |

With the mu/beta filter frontal channels are at chance. With a wide filter, 8 frontal channels alone reach 64%, as good as all 64. Forehead electrodes carry almost no motor signal, so the model is likely reading slow eye movement/blinks (EOG) locked to the cue, since the target is at the top for fists and the bottom for feet. The 8-30 Hz filter removes most of that, which is why the effect vanished. The wide-filter "all" result is probably partly this shortcut too.

### Task 4: Short answers
1. 58% of 45 trials is 26/45; a one-sided binomial test against 50% gives p about 0.2, so not above chance. You need about 29/45 (64%) for p<0.05. Check with a binomial test or a permutation test on shuffled labels.
2. The depthwise layer learns spatial filters, weighted electrode combinations per temporal band, similar to CSP, that pick out motor-cortex sources.
3. Randomize cue position across trials, or remove EOG with ICA/regression, or drop frontal channels and high-pass/band-pass to mu/beta.

## Part 2
1. Hypothesis (written before coding): Per-timestep attention has no locality or spatial prior, so on 8 subjects it overfits a 482x482 attention matrix. A conv stem (temporal filter + depthwise spatial filter, as in EEGNet) that cuts the sequence to about 32 tokens should reduce seed variance, with little change in peak accuracy.

2. What I changed: MyModel = EEGNet-style conv stem (temporal conv, depthwise spatial conv across all 64 electrodes, pooling by 15) producing about 32 tokens, then a 2-layer transformer encoder with CLS token. Attention matrix shrinks from 482^2 (about 232k) to about 33^2 (about 1.1k). The spatial filters can express the anterior-posterior (C3/C4 vs Cz) contrast of this bilateral task.

3. Results (seeds 0,1,2, test = subjects 9-10, 90 trials, 30 epochs):
- EEGNet: peak best 0.611 / worst 0.533 / mean 0.570; final-epoch mean 0.563
- MyModel: peak best 0.611 / worst 0.589 / mean 0.596; final-epoch mean 0.526
- Vanilla transformer: did not finish on a 2-core Colab CPU in the time limit (482 tokens, O(T^2) attention); no number reported.
Peak numbers select the best epoch on test and are optimistic; final-epoch numbers are the honest ones. At 90 trials, about 62% is needed for p<0.05, so none of these beat chance. Peak spread was smaller for MyModel (consistent with the hypothesis), but final accuracy was lower, so I do not claim a gain. Any benefit likely comes from the conv stem, which is EEGNet's front end, not from attention.

4. Figure: figure.png (t-SNE of test embeddings) was not generated in this run.
