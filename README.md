# Discrete Flow Maps for Cityscapes segmentation

Cityscapesの20-class意味的セグメンテーションをDiscrete Flow Maps（DFM）で
学習する実装です。Stage 1対角事前学習、Stage 2整合性蒸留、事前学習なしの
joint training、single GPU、単一ノードDDPに対応します。CFM実装は設計と数式の
参照専用であり、変更していません。

## 全体構成

経路とmean-denoiser Flow Mapは次のとおりです。

\[
x_t=(1-t)x_0+t x_1,\qquad
X^\theta_{s,t}(x_s)
=x_s+\frac{t-s}{1-s}\left(\psi^\theta_{s,t}(x_s,I)-x_s\right).
\]

`x_1`はvoidを含む20-class one-hotです。`x_0`は`source.prior_type`で
`gaussian`、`dirichlet`、`image_gaussian`から選びます。
`flow.time_eps`はFlow Mapの分母のゼロ除算防止にだけ使います。

学習方式は2つです。

| 方式 | entrypoint | 初期値 | iterationの損失 |
|---|---|---|---|
| Stage 1 → Stage 2 | `src/train.py` | Stage 2はStage 1 checkpoint | Stage 1: 対角CEのみ、Stage 2: 対角CE + 1種類の整合性損失 + source |
| joint | `src/train_joint.py` | endpointをランダム初期化 | epoch 1から対角CE + 1種類の整合性損失 + source |

総損失はStage 2とjointで共通です。

```text
loss_total =
    primary.weight * loss_diagonal
  + consistency.weight * consistency.max_weight * schedule * loss_consistency
  + source.var_weight * loss_source_var
  + source.align_weight * loss_source_align
```

`schedule`は`start_epoch`と`warmup_epochs`によるlinear warm-upです。

### Stage 1

`experiment.stage: diagonal_pretrain`を使います。モデル入力は
`(x_t, image, s=t, t=t)`で、endpointの学習目的は対角CEだけです。既存の
image-conditioned sourceを使う場合のsource regularizerは維持しますが、
PSD/CSD/ECLD/ESDのdispatchにも`torch.func.jvp`にも入りません。この回帰条件は
unit testでも禁止関数に置き換えて検証しています。

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python src/train.py \
  --config configs/stage1_diagonal_cityscapes.yaml
```

### Stage 2

`experiment.stage: consistency_distillation`を使い、`checkpoint.init_from`に
Stage 1 checkpointを指定します。旧`esd_distillation`はESD checkpointの
後方互換aliasとしてload/resume時も受理します。整合性損失は1 runにつき1種類で、
`loss.consistency.type`を`psd`、`csd`、`ecld`、`esd`から選びます。

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python src/train.py \
  --config configs/stage2_ecld_cityscapes.yaml
```

### Joint training（対角事前学習なし）

`src/train_joint.py`と`experiment.stage: joint_training`を使います。
`checkpoint.init_from`は禁止され、`resume`だけが許可されます。各iterationで
source priorを生成し、対角CE用の時刻と整合性用の時刻を別々にsampleして、
両損失を同時にbackwardします。

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python src/train_joint.py \
  --config configs/joint_ecld_cityscapes.yaml
```

## 整合性損失

共通入口は`compute_consistency_loss(...) -> ConsistencyResult`です。teacherは
stop-gradient、studentは勾配を保持し、損失固有の統計名は衝突しません。

### PSD

`s < u < t`の3時刻をsampleします。`s→u→t`のcomposed Flow Map teacherと
`s→t`のdirect Flow Map studentを比較し、teacher probabilityを再正規化して
detachします。PSDはJVPを使いません。そのため設定は必ず次の形です。

```yaml
precision:
  jvp_dtype: null
  numerical_dtype: fp32
```

PSDへ`bf16`/`fp32` JVPを指定するとconfig validation errorになります。

### CSD

`s < t`をsampleし、Flow Mapの時刻方向JVPと時刻`t`のinstantaneous diagonal
teacherからFP32 residualを作り、二乗ノルムを最小化します。teacher全体は
detachされます。`loss_csd`、residual norm、JVP平均/最大絶対値、dtype codeを
記録します。

### ECLD

時刻方向logits JVPから、完全Jacobianを作らずexact softmax JVP

\[
\dot p=p\odot\left(\dot z-\langle p,\dot z\rangle\mathbf 1\right)
\]

を計算します。transport後のendpoint teacherに対するCEと、`gamma(s,t)^2`で
重み付けしたtemporal derivative lossを`ec_weight`、`td_weight`で合成します。
`time_weighting`は`none`または`inverse_square`です。

### ESD

既存DFM式を保持しています。対角drift

\[
b_s=\frac{\psi_{s,s}(x_s)-x_s}{1-s}
\]

に沿うjoint JVP over `(x_s, s)`を計算し、softmax gaugeを中心化します。

\[
\delta=D_s z_{s,t}-\langle\psi_{s,t},D_s z_{s,t}\rangle\mathbf 1,
\]

```python
log_arg_raw = (
    one_minus_t[:, None, None, None]
    - (one_minus_s * delta_time)[:, None, None, None] * delta
)
```

teacherは`z_ss - log(clamped_log_arg)`から作り、損失方向は
`KL(teacher || student)`です。invalid class/pixel/sample率、nonfinite率、
clamp率、valid率、時刻bucket、teacher entropy、adaptive KL weight、
skip有無を記録します。`clamp`、`mask_pixel`、`skip_batch`を選べます。
全画素invalidでもstudent graphを保持するzero lossを返し、対角CEのbackwardを
継続します。NaN/Infを無条件に0へ置換して問題を隠す実装ではありません。

## bf16 JVPとFP32 JVP

CSD/ECLD/ESDはYAMLで切り替えます。既定はbf16です。

```yaml
runtime:
  amp: true
  amp_dtype: bf16
loss:
  consistency:
    precision:
      jvp_dtype: bf16  # bf16 | fp32
      numerical_dtype: fp32
      debug_assertions: false
```

`runtime.amp: false`とbf16 JVPの組合せ、`numerical_dtype: bf16`、CUDAでbf16
非対応の環境は明示的なエラーです。比較時は既存キーをoverrideできます。

```bash
uv run python src/train.py --config configs/debug_ddp_stage2_ecld.yaml \
  --set loss.consistency.precision.jvp_dtype=fp32
```

bf16 pathではモデルforward、JVP内forward、JVP出力、teacher用forwardをbf16に
し、直後にFP32へ戻します。softmax、log-softmax、exact softmax JVP、Flow Map
係数、CSD residual、ECLD CE/TD、ESD delta/log/teacher/KL/adaptive weight、
全diagnosticsと最終lossはFP32です。FP32へ戻すのは、確率の正規化、logの境界、
小さな差分、KLをbf16の狭い仮数で評価しないためです。

`debug_assertions: true`ではJVP前後、student/teacher probability、lossのdtypeを
assertします。ログの`*_jvp_dtype_code`はFP32=`0`、bf16=`1`です。実GPU debug
ではCSD/ECLD/ESDの全runで`1`を確認しました。

## DDP設計

`distributed.enabled: auto`は`WORLD_SIZE > 1`でDDPを有効にし、通常の
`python src/train.py`ではsingle-processへfallbackします。本学習はNCCL、
CPU unit testはGlooです。

```yaml
distributed:
  enabled: auto
  backend: nccl
  init_method: env://
  find_unused_parameters: false
  broadcast_buffers: false
  gradient_as_bucket_view: true
```

学習用の`DDPCompatibleTrainingModel`はendpoint modelとsource modelを1つの
composite moduleとして所有します。Stage 1、Stage 2、jointの完全なforward
graphとJVPを、このcompositeへの1回のDDP `forward`内で構築します。学習損失を
作るために`ddp_model.module.forward_logits(...)`を外側から呼びません。
validation、inference、checkpoint保存時だけunwrapします。このためtrainable
sourceのgradientもendpointと同じDDP reducerで同期されます。frozen sourceは
gradientを持ちません。

学習DataLoaderは`DistributedSampler`を使い、各epochで`set_epoch(epoch)`を
呼びます。gradient accumulation中のoptimizer stepを行わないmicro stepは
`no_sync()`を使い、epoch末または`max_iterations`末の端数もstepします。
schedulerはoptimizer stepと同期します。

### Global batch size

`training.batch_size`は常にglobal batch sizeです。

```text
local_batch_size = global_batch_size // world_size
effective_global_batch_size = global_batch_size * grad_accum_steps
```

割り切れない場合は開始前にエラーです。2 GPUでglobal batch 4なら各rankの
local batchは2です。world size、rank、local rank、global/local/effective
batch、accumulationを開始ログへ記録します。

### 2 GPU実行

```bash
cd /home/igarashi_25/playground_2/CSDFM/DFM

CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train.py \
  --config configs/stage2_ecld_cityscapes.yaml
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  src/train_joint.py \
  --config configs/joint_ecld_cityscapes.yaml
```

対応スクリプトは`scripts/train_stage2_{psd,csd,ecld,esd}_ddp.sh`と
`scripts/train_joint_{psd,csd,ecld,esd}_ddp.sh`です。

## Checkpoint、ログ、評価

rank 0だけがwandbを初期化・記録し、通常ログ、`metrics.jsonl`、
`config_resolved.yaml`、checkpoint、可視化、evaluation JSONを書きます。
iteration統計はrank間でmeanまたは、最大絶対値・最大時間についてmax reduction
してから記録します。GPU peakはrank別リスト、rank平均、rank最大を保存します。

checkpoint保存の前後にbarrierを置き、state dictには`module.` prefixを付けません。
保存内容はstage、epoch、global step、model、source model、optimizer、
scheduler、scaler、resolved config、model signature、metrics、world size、
global/local batchです。resumeは完全復元し、world size変更は許容します。
global batch変更時は警告します。旧`module.`付きstate dictもload時に除去します。

`init_from`はStage 1のmodel/source重みだけを読み、optimizer等を初期化します。
joint checkpointは`stage: joint_training`で、joint同士だけresume可能です。
Stage 1/2 checkpointをjoint resumeへ渡すとstage mismatch errorになります。

YAMLは`extends: base.yaml`で同じディレクトリの設定を継承でき、派生設定は
上書き部分だけを保持します。load後の完全な設定は各runの
`config_resolved.yaml`へ保存されるため、実行条件は常に再現できます。

validation/evaluationはpaddingしない`DistributedEvalSampler`を使うため、全画像を
ちょうど1回だけ評価します。各rankの20×20 confusion matrixをSUM all-reduce
してからglobal mIoU、pixel accuracy、mean class accuracy、class-wise IoUを
計算します。GT class 19だけを除外し、prediction class 19は誤予測として残します。

DDP evaluationも可能です。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun --standalone --nproc_per_node=2 src/evaluate.py \
  --config configs/stage2_ecld_cityscapes.yaml \
  --checkpoint /path/to/best.pt
```

## Debugと実測結果

全debug設定は48×96、1 epoch、2 iterations、global batch 4、DDP local batch 2、
bf16 AMP、wandb disabledです。Stage 1を作った後、次ですべて実行できます。

```bash
scripts/debug_all_ddp.sh
```

2026-07-26、NVIDIA RTX 6000 Ada 2枚、PyTorchのpeak allocated memory実測です。
全runでloss/gradientはfinite、optimizer step成功、checksum差0、metricsの
iteration行は2行、checkpointはrank 0の1組だけでした。時刻はwarm-up/初回JVP
オーバーヘッドを除くiteration 2です。

| 方式 | loss | total loss | consistency loss | grad norm | iter 2 (s) | max peak/rank (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Stage 2 | PSD | 1.80965 | 2.94146 | 0.57854 | 0.0594 | 89.90 |
| Stage 2 | CSD bf16 | 1.51629 | 0.00207 | 0.57855 | 0.1275 | 163.98 |
| Stage 2 | ECLD bf16 | 2.69967 | 11.83520 | 0.57946 | 0.1215 | 163.91 |
| Stage 2 | ESD bf16 | 1.51719 | 0.01101 | 0.57843 | 0.1139 | 173.70 |
| joint | PSD | 1.81143 | 2.94687 | 0.61167 | 0.0579 | 90.60 |
| joint | CSD bf16 | 1.51720 | 0.00443 | 0.61142 | 0.1433 | 164.69 |
| joint | ECLD bf16 | 2.70712 | 11.90321 | 0.61270 | 0.1394 | 164.62 |
| joint | ESD bf16 | 1.51707 | 0.00333 | 0.61172 | 0.1097 | 174.40 |

### メモリ比較

Stage 2の同じdebug model、global batch 4で比較しました。single GPUはlocal
batch 4、DDPは各rank local batch 2です。

| loss / 条件 | JVP code | loss (iter 2) | grad norm | iter 2 (s) | max peak (MiB) |
|---|---:|---:|---:|---:|---:|
| ECLD single GPU FP32 | 0 | 2.69718 | 0.59394 | 0.1166 | 419.50 |
| ECLD DDP FP32 | 0 | 2.69962 | 0.57946 | 0.1326 | 264.85 |
| ECLD DDP bf16 | 1 | 2.69967 | 0.57946 | 0.1215 | 163.91 |
| ESD single GPU FP32 | 0 | 1.51595 | 0.59401 | 0.0820 | 355.93 |
| ESD DDP FP32 | 0 | 1.51720 | 0.57843 | 0.1115 | 189.61 |
| ESD DDP bf16 | 1 | 1.51719 | 0.57843 | 0.1139 | 173.70 |

local batch分割により、DDP FP32のrank最大はsingle FP32比でECLD 36.9%、
ESD 46.7%減りました。さらにbf16 JVPはDDP FP32比でECLD 38.1%、ESD 8.4%
減りました。DDP自体が「1 GPU内の1サンプル当たりメモリ」を減らすわけではなく、
同じglobal batchを複数GPUのlocal batchへ分割した結果、各GPUのbatch由来メモリ
が減ります。時間は2 iterationだけのdebug測定であり、性能benchmarkでは
ありません。

## テスト

```bash
uv run pytest -q
```

config strictness、4損失と時刻順序、PSD JVP禁止、bf16/FP32の実dtypeとFP32
post-processing、softmax JVP、teacher detach、ESD invalid/adaptive処理、
Stage回帰、joint、checkpoint、void評価、CPU/Gloo 2-process reduction、
no_sync、endpoint/source gradient同期、frozen source、rank 0保存、
non-padding samplerを検証します。

## 既知の制限

- endpoint modelは現在UNetのみです。sourceはSegFormerと軽量UNetです。
- 主対象は単一ノードNCCLです。multi-nodeの性能・障害復旧は未検証です。
- ESD joint JVPとCSD/ECLD JVPは通常の対角forwardよりメモリを使います。
- `source.pretrained: true`の初回はHugging Face weightまたはcacheが必要です。
- bf16の可否はCUDA deviceで実行開始時に検証します。FP16 JVPは未対応です。
- debug runは配線、dtype、finite backward、DDP同期確認用で、精度評価では
  ありません。
