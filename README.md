# Discrete Flow Maps for Cityscapes segmentation

このディレクトリは、既存 `CFM/src/segv4` の Cityscapes データ処理、RRDB 条件
エンコーダ、時間条件付き UNet、SegFormer Gaussian source、評価・可視化を
DFM 用に独立移植した実装です。`CFM/src` は参照しただけで変更していません。

学習は別々のコマンドで動く 2 段階構成です。

- Stage 1 (`diagonal_pretrain`) は対角 mean denoiser
  \(\psi_{\theta,t,t}(x_t,I)\) だけを CE で学習します。ESD と JVP は呼びません。
- Stage 2 (`esd_distillation`) は Stage 1 checkpoint のモデル/source 重みだけを
  `init_from` で読み込み、対角 CE、ESD、source loss を組み合わせて epoch 1 から
  新しく学習します。

## セットアップ

このワークスペースでは親ディレクトリの既存 `.venv` を再利用するため、そのまま
`uv run` できます。新しい環境を作る場合は次を実行してください。

```bash
uv sync --extra full
```

依存バージョンは `pyproject.toml` と `uv.lock` に固定しています。

## 実行

Stage 1:

```bash
cd /home/igarashi_25/playground_2/CSDFM/DFM
CUDA_VISIBLE_DEVICES=0 \
uv run python src/train.py \
  --config configs/stage1_diagonal_cityscapes.yaml
```

Stage 2:

```bash
cd /home/igarashi_25/playground_2/CSDFM/DFM
CUDA_VISIBLE_DEVICES=0 \
uv run python src/train.py \
  --config configs/stage2_esd_cityscapes.yaml
```

評価:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python src/evaluate.py \
  --config configs/stage2_esd_cityscapes.yaml \
  --checkpoint /path/to/best.pt
```

Debug:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py \
  --config configs/debug_diagonal_cityscapes.yaml
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py \
  --config configs/debug_esd_cityscapes.yaml
```

限定的な override は、既存キーに限って繰り返し指定できます。未知キーはエラーです。

```bash
uv run python src/train.py --config configs/stage1_diagonal_cityscapes.yaml \
  --set training.batch_size=2 \
  --set runtime.device=cuda
```

## 設定

すべての基本設定は YAML に置きます。

- `experiment`: run 名、seed、出力先、2 段階のどちらか
- `runtime`: device、AMP bf16/fp16、compile、deterministic
- `dataset`: Cityscapes root、20 学習クラス、void index 19、解像度、worker
- `augmentation`: horizontal flip、color jitter、ImageNet normalize
- `model`: RRDB image encoder と時間条件付き UNet
- `source`: `gaussian`、`dirichlet`、`image_gaussian`、SegFormer/UNet source、
  `mu`/`logvar`、fixed std、align/variance loss
- `flow`: 分母保護用 `time_eps`（既定値 `1e-5`）
- `time_sampling`: `[0,1]` の sorted uniform と `min_gap`
- `training`: epoch、batch、optimizer、parameter group、scheduler、保存/評価間隔
- `loss`: 対角 CE と ESD の明示的な重み、warm-up、安全策、adaptive KL
- `checkpoint`: `init_from` または `resume`
- `evaluation`: split、Flow Map step 数、保存数
- `wandb`: project、run 名、entity、mode、tag

実際に使用した、CLI override 適用後の全設定は必ず
`config_resolved.yaml` に保存されます。

## DFM と Stage 1

経路は

\[
x_t=(1-t)x_0+t x_1
\]

です。`x_1` は void を含む 20-class one-hot、`x_0` は `source.prior_type`
で選んだ prior です。Stage 1 はモデルへ `(x_t, I, s=t, t=t)` を渡し、

\[
L_{\rm diag}=\operatorname{CE}(z_{t,t},y)
\]

だけを学習します。label smoothing は `training.label_smoothing` です。

mean-denoiser Flow Map は

\[
X^\theta_{s,t}(x_s)
=x_s+\frac{t-s}{1-s}\left(\psi^\theta_{s,t}(x_s)-x_s\right)
\]

であり、コードではゼロ除算を防ぐ分母だけ
`(1-s).clamp_min(flow.time_eps)` としています。1-step 推論は `s=0,t=1`、
複数 step は `[0,1]` の等間隔 grid です。

## Stage 2 と ESD

Stage 2 の損失は曖昧な `eta` ではなく、

```text
loss_total =
    primary.weight * loss_diagonal
  + consistency.weight * consistency.max_weight * schedule * loss_esd
  + source.var_weight * loss_source_var
  + source.align_weight * loss_source_align
```

です。`max_weight` は通常 `1.0` とし、`schedule` は `start_epoch` と
`warmup_epochs` による linear warm-up です。

対角 drift は

\[
b_s=\frac{\psi_{s,s}(x_s)-x_s}{1-s}.
\]

`torch.func.jvp` で入力 `(x_s,s)`、tangent `(b_s,1)` の joint derivative

\[
D_s z_{s,t}=\partial_s z_{s,t}+J_xz_{s,t}\,b_s
\]

を直接計算します。完全 Jacobian は作りません。softmax gauge を除くため、

\[
\delta=D_s z_{s,t}
-\langle\psi_{s,t},D_s z_{s,t}\rangle\mathbf 1
\]

とし、

\[
a=(1-t)\mathbf1-(1-s)(t-s)\delta,\qquad
z^{teacher}=z_{s,s}-\log a
\]

から `teacher_prob = softmax(z_teacher).detach()` を作ります。損失は
`KL(teacher || student)` です。JVP、delta、`a`、log、teacher softmax、KL、
統計は AMP の外で float32 に昇格します。student logits は detach しません。

## invalid teacher と adaptive KL

`a` に非 finite または `<= log_eps` のクラスがある割合が
`esd_clamp_ratio` です。画素内の全クラスが finite かつ正の場合だけ
`esd_valid_pixel_ratio` に数えます。iteration 値と epoch 平均に加え、
`t` bucket 別 clamp 率も `metrics.jsonl` に保存します。

- `clamp`: log 入力を安全化し、全画素の KL を使う比較用モード
- `mask_pixel`（既定）: 1 クラスでも invalid な画素を KL 平均から除外
- `skip_batch`: 指定 threshold を超える batch の ESD だけをゼロ化
- `skip_batch_threshold`: `mask_pixel` と併用した場合も、超過 batch の ESD
  だけをゼロ化

有効画素が 0 の場合は student graph を保持するゼロ loss を返すため、
対角 CE の backward は継続します。

adaptive KL は

\[
w(x)=\operatorname{sg}\left[
(\lVert q-p\rVert_2^2+c)^{-r}
\right]
\]

です。`normalize_mean` と `max_weight` を設定でき、weight は detach
されています。

## `init_from` と `resume`

両方を同時に指定すると設定エラーです。

- `init_from`: `diagonal_pretrain` stage、20 classes、モデル全設定、source の
  主要構成を照合し、model/source 重みだけを strict load します。optimizer、
  scheduler、scaler、epoch、global step は引き継ぎません。
- `resume`: 同じ stage の途中 checkpoint から model、source、optimizer、
  scheduler、scaler、completed epoch、global step、best mIoU を完全復元します。
  `resume` の完全復元は `load_optimizer`/`load_scheduler` で無効化できません。

checkpoint には次を保存します。

```text
stage, epoch, global_step, model, source_model,
optimizer, scheduler, scaler, config, model_signature, metrics
```

出力は `latest.pt`、`best.pt`、`epoch_XXXX.pt`、`config_resolved.yaml`、
`train_log.txt`、`metrics.jsonl` です。`best.pt` は validation mIoU 最大です。

## Cityscapes 評価

モデルは 20 クラスすべてで学習します。評価の confusion matrix は 20×20
のまま、GT class 19 の画素だけを除外します。prediction class 19 は除外せず、
通常の誤予測として数えます。mIoU、pixel accuracy、mean class accuracy、
class 0–18 の IoU/accuracy、confusion matrix を出力します。

## テスト

```bash
uv run pytest -q
```

config strictness、Flow Map endpoint、float32/bf16、Stage 1 の ESD/JVP 非呼出、
checkpoint transition/resume、joint JVP、KL 方向、teacher detach、invalid
mask、全画素 invalid、adaptive weight、bf16 autocast、void 評価を検証します。

## 既知の制限

- endpoint model は現在 UNet のみです。source model は SegFormer と軽量 UNet
  を選択できます。
- ESD の joint JVP は通常の対角 forward より GPU memory を多く使います。
- `source.pretrained: true` の初回実行は Hugging Face weight の取得または
  ローカル cache が必要です。
- debug run は配線・finite backward の確認用で、2 iteration の精度には
  学習性能上の意味はありません。

