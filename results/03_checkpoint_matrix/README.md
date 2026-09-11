# Checkpoint Choice Versus Architecture Choice

## Why This Exists

The batch-2 architecture sweep published one number per architecture, taken
from DeepLabCut's `snapshot-best`, which maximises internal `test.mAP` over
five validation frames. Those five labels are known to be suspect, and a single
HRNet-W48 model moved 44 px across its own
checkpoints. If checkpoint choice can move one model more than the gap between
two architectures, a one-checkpoint-per-architecture table cannot support an
architecture ranking.

Every surviving checkpoint of every architecture was therefore evaluated on the
unchanged 100 final-minute frames. No model was retrained.

## Four Views Of The Same Data

| Model | Internal rule | Mean over the 5 common checkpoints | Fixed epoch 200 | Oracle (upper bound) | Range | Internal-rule cost |
|---|---:|---:|---:|---:|---:|---:|
| HRNet-W18 | 31.25 px (ep 40.0) | 33.61 px (SD 5.89) | 35.02 px | 23.78 px (ep 150) | 23.78-39.76 px | 7.47 px |
| HRNet-W32 | 20.06 px (ep 90.0) | 21.57 px (SD 2.25) | 20.72 px | 19.89 px (ep 100) | 19.89-25.48 px | 0.16 px |
| HRNet-W48 | unavailable | 30.42 px (SD 7.26) | 21.74 px | 21.74 px (ep 200) | 21.74-37.79 px | n/a |
| ResNet-50 | 28.60 px (ep 20.0) | 20.47 px (SD 1.72) | 23.29 px | 18.85 px (ep 100) | 18.85-28.60 px | 9.75 px |
| CSPNeXt-S | 70.12 px (ep 10.0) | 25.82 px (SD 0.10) | 25.73 px | 25.72 px (ep 150) | 25.72-70.12 px | 44.40 px |

- **Internal rule** is what the current validation set actually buys you.
- **Mean over the five common checkpoints** is the most defensible single
  number here. The epoch set is declared in advance, identical for every model,
  and averaging selects nothing, so it leaks nothing and is far less sensitive
  to one unlucky checkpoint than any single reading.
- **Fixed epoch 200** reads every architecture at one identical epoch, so it
  never consults the validation labels. It does penalise architectures that peak
  early, and these did peak at very different epochs.
- **Oracle** is the lowest external error over the surviving checkpoints. It is
  an UPPER BOUND on what an architecture could deliver. It is not achievable in
  practice and **must never be used to select anything**, because selecting on
  the report set converts it into a validation set.
- **Range** is min to max across checkpoints; **span** is the width.

## What Can And Cannot Be Concluded

Ranking claims are only made where two ranges do not overlap:

- HRNet-W32 is better than CSPNeXt-S: their ranges do not overlap.

Not separable:

- HRNet-W18 vs HRNet-W32: ranges overlap, not separable.
- HRNet-W18 vs HRNet-W48: ranges overlap, not separable.
- HRNet-W18 vs ResNet-50: ranges overlap, not separable.
- HRNet-W18 vs CSPNeXt-S: ranges overlap, not separable.
- HRNet-W32 vs HRNet-W48: ranges overlap, not separable.
- HRNet-W32 vs ResNet-50: ranges overlap, not separable.
- HRNet-W48 vs ResNet-50: ranges overlap, not separable.
- HRNet-W48 vs CSPNeXt-S: ranges overlap, not separable.
- ResNet-50 vs CSPNeXt-S: ranges overlap, not separable.

## Findings

1. Checkpoint choice moves a single architecture by up to
   44.40 px (CSPNeXt-S). Several
   architecture gaps in the published sweep are smaller than that, so those
   gaps were not measuring architecture.
2. The internal five-frame rule costs 0.16 to 44.40 px
   against the oracle, median 8.61 px. That gap is the price of
   the current validation set, and it is the direct argument for relabelling
   before the active-learning experiment.
3. The three views disagree about the winner: internal rule favours
   HRNet-W32, fixed epoch 200 favours HRNet-W32, and
   the oracle bound favours ResNet-50. A single-run,
   five-validation-frame design cannot resolve this.
4. Architectures peak at very different epochs, so a fixed-epoch comparison and
   a best-checkpoint comparison are different experiments. Neither is wrong;
   they answer different questions, and both are reported here rather than one
   being presented as the truth.

## Consequence For The Architecture Decision

The architecture cannot be selected from the final-minute set without breaking
the report-only rule, and it cannot be selected reliably from five suspect
validation frames either. The decision is therefore deferred until the
validation set is rebuilt: all 100 training-pool labels reviewed and the
validation set expanded from 5 to 20 frames. Only then does "each architecture
at its own best checkpoint" become both legal and precise.

The retrain after relabelling should also save snapshots more densely (every 10
epochs instead of every 25), because the surviving checkpoints here are too
coarse to locate a real optimum.

## 中文摘要

架构比较原来每个模型只报一个数，而那个数对应的检查点是用 5 张标注可疑的验证帧
挑出来的。实测同一个模型换检查点可以差
44 px，比若干架构之间的差距还大，所以原表格测到
的不是架构差异。

本报告把每个架构的**全部存活检查点**都评估了一遍（不重训任何模型），给出四种
读法：内部规则（实践中实际能拿到的）、固定 epoch 200（苹果对苹果，但对早收敛
的架构不利）、oracle 上界（潜力，**严禁用于选择**，否则测试集就变成验证集）、
以及区间范围。

只有当两个架构的区间**完全不重叠**时才下排名结论。三种读法给出的赢家并不一致
，说明单次运行 + 5 帧验证的设计无法区分这些架构。内部规则相对 oracle 的代价是
8.61 px 中位数，这就是当前验证集的代价，也是主动学习之前必须
重新标注的直接理由。

架构决定因此推迟到验证集重建之后（复查 100 张标注、验证集从 5 张扩到 20 张）。
重训时还应把存档间隔从 25 epoch 改为 10 epoch，因为现有检查点太稀疏，定位不到
真正的最优点。

## Files

- `checkpoint_matrix.csv`: one row per architecture and checkpoint, with
  internal metrics and external errors.
- `architecture_views.csv`: the four views plus range and span per architecture.
- `dominance.json`: which pairs are separable and which overlap.
- `01_checkpoint_matrix.png`: external error against epoch, and the range per
  architecture.

Model weights, predictions, frames, and labels stay in ignored local
directories and are not on GitHub.
