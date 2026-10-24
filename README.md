# ML from scratch drills

This repo holds 78 machine learning components to build from scratch in NumPy and PyTorch.
It is a practice set. Each drill has stub functions and a test file.
Every stub raises `NotImplementedError`, so every test fails at the start.
Your job is to fill in the functions until the tests pass.
The docstrings give the spec: shapes, conventions and the gotchas to watch for.

## Quick start

With uv:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest                         # everything (all fail at the start)
pytest -k drill20              # one drill
pytest tests/test_losses.py    # one module
```

The code lives in `src/drills/<module>.py`. The tests live in `tests/test_<module>.py`.

## Rules of practice

- Start each drill from a blank function. Do not look at old answers.
- Write the shape of every tensor in a comment as you go.
- Write one small test of your own first. Then run the given tests.
- Time yourself. The minutes in the index are a target, not a limit.
- When a test fails, read it. The test says what the spec means.

## Suggested path

Start with these twelve groups, in this order.

1. #2 Softmax cross-entropy forward and backward (NumPy); #3 Masked cross-entropy with ignore_index, label smoothing and temperature
2. #20 Scaled dot-product attention with padding and causal masks; #21 Multi-head attention, self- and cross-attention; #42 Attention with pair bias and gating
3. #36 Distogram target and masked distogram cross-entropy
4. #38 Smooth lDDT loss and hard lDDT
5. #37 Weighted Kabsch alignment, aligned weighted MSE and RMSD
6. #17 Pre-norm transformer block and decoder, with a parameter count
7. #6 Symmetric InfoNCE / CLIP contrastive loss
8. #4 Stable BCE-with-logits and its gradient; #5 Binary focal loss from logits
9. #68 Integrated drill: masked loss + small module + toy loop
10. #7 Censored Huber plus pairwise difference affinity loss
11. #43 Triangle multiplicative update, outgoing and incoming
12. #35 Debug a training script that runs but does not learn

Then pick any drill from the index. Tier A is the core set. Tier B and C go further.

## Index

78 drills, 443 tests in total.

### Losses and metrics (`losses`)

| # | Drill | Symbols | Tier | Min | Tests |
|---|---|---|---|---|---|
| 1 | Stable softmax and log-softmax on any axis | `losses.softmax`, `losses.log_softmax` | A | 8 | 4 |
| 2 | Softmax cross-entropy forward and backward (NumPy) | `losses.softmax_cross_entropy` | A | 15 | 3 |
| 3 | Masked cross-entropy with ignore_index, label smoothing and temperature | `losses.masked_cross_entropy` | A | 15 | 4 |
| 4 | Stable BCE-with-logits and its gradient | `losses.bce_with_logits` | A | 10 | 3 |
| 5 | Binary focal loss from logits | `losses.binary_focal_loss` | A | 10 | 6 |
| 6 | Symmetric InfoNCE / CLIP contrastive loss | `losses.clip_loss` | A | 25 | 3 |
| 7 | Censored Huber plus pairwise difference affinity loss | `losses.censored_huber_loss`, `losses.pairwise_difference_loss`, `losses.affinity_loss` | A | 25 | 4 |
| 8 | KL divergence and the VAE loss | `losses.kl_discrete`, `losses.gaussian_kl`, `losses.vae_loss` | C | 15 | 4 |
| 9 | Classification metrics: P/R/F1, AUC with ties, Spearman | `losses.precision_recall_f1`, `losses.rankdata_average`, `losses.roc_auc`, `losses.spearman` | B | 15 | 5 |
| 10 | Screening metrics: enrichment factor, top-k recall, grouped bootstrap CI | `losses.enrichment_factor`, `losses.top_k_recall`, `losses.grouped_bootstrap_ci` | C | 15 | 5 |

### Core layers and architecture (`layers`)

| # | Drill | Symbols | Tier | Min | Tests |
|---|---|---|---|---|---|
| 11 | Two-layer MLP with a manual backward pass | `layers.mlp_forward_backward` | A | 20 | 3 |
| 12 | LayerNorm and RMSNorm | `layers.layer_norm`, `layers.rms_norm`, `layers.LayerNorm`, `layers.RMSNorm` | A | 10 | 4 |
| 13 | BatchNorm in train and eval, with running statistics | `layers.BatchNorm` | B | 15 | 3 |
| 14 | Inverted dropout and row-shared pair dropout | `layers.dropout`, `layers.row_dropout_mask` | B | 5 | 4 |
| 15 | SwiGLU transition, plus a version chunked over the hidden dim | `layers.Transition` | B | 12 | 6 |
| 16 | Sinusoidal positional encoding and RoPE | `layers.sinusoidal_positional_encoding`, `layers.rope` | B | 10 | 6 |
| 17 | Pre-norm transformer block and decoder, with a parameter count | `layers.CausalSelfAttention`, `layers.MLP`, `layers.TransformerBlock`, `layers.Decoder`, `layers.count_params_per_block` | A | 35 | 7 |
| 18 | Minimal reverse-mode autograd (micrograd-style) | `layers.Value` | C | 30 | 3 |
| 19 | Conv2d forward: loops first, then im2col | `layers.conv2d_output_size`, `layers.conv2d_loops`, `layers.conv2d_im2col` | C | 20 | 8 |

### Attention and decoding (`attention`)

| # | Drill | Symbols | Tier | Min | Tests |
|---|---|---|---|---|---|
| 20 | Scaled dot-product attention with padding and causal masks | `attention.scaled_dot_product_attention`, `attention.scaled_dot_product_attention_np` | A | 15 | 10 |
| 21 | Multi-head attention, self- and cross-attention | `attention.MultiHeadAttention` | A | 20 | 6 |
| 22 | KV cache, then GQA/MQA on the cache | `attention.repeat_kv`, `attention.kv_cache_bytes`, `attention.CachedCausalSelfAttention` | C | 20 | 9 |
| 23 | FlashAttention-style tiled attention (online softmax) | `attention.flash_attention` | B | 25 | 11 |
| 24 | Decoding: temperature, greedy, top-k, top-p, beam search | `attention.top_k_filter`, `attention.top_p_filter`, `attention.sample_next_token`, `attention.beam_search` | C | 30 | 10 |
| 25 | Sliding-window attention and the attention-free transformer | `attention.sliding_window_attention`, `attention.aft_full`, `attention.aft_simple` | C | 15 | 9 |

### Training mechanics (`training`)

| # | Drill | Symbols | Tier | Min | Tests |
|---|---|---|---|---|---|
| 26 | End-to-end training script, closed book | `training.TensorPairDataset`, `training.MLPClassifier`, `training.train_one_epoch`, `training.evaluate`, `training.overfit_one_batch`, `training.pad_collate` | A | 25 | 8 |
| 27 | Finite-difference gradient check | `training.numerical_gradient`, `training.max_relative_error`, `training.check_gradient` | A | 10 | 5 |
| 28 | Optimisers: SGD with momentum, Adam and AdamW | `training.SGDMomentum`, `training.Adam`, `training.AdamW`, `training.decay_param_groups` | B | 15 | 6 |
| 29 | LR schedules: warmup-cosine and the AF3 step decay | `training.warmup_cosine_lr`, `training.af3_lr`, `training.make_warmup_cosine_scheduler` | B | 10 | 20 |
| 30 | Clip by global norm, gradient accumulation, grad-norm logging, skip-step guard | `training.global_grad_norm`, `training.clip_grad_norm`, `training.accumulate_gradients`, `training.per_module_grad_norms`, `training.should_skip_step` | B | 20 | 14 |
| 31 | Mixed precision: autocast and GradScaler order, bf16 with fp32 islands | `training.amp_train_step`, `training.fp32_island` | B | 15 | 5 |
| 32 | Parameter EMA with warm-up, store/copy_to/restore | `training.EMA` | B | 10 | 4 |
| 33 | Structure-wise vs atom-wise loss reduction, temperature multi-dataset sampling | `training.structure_wise_mean`, `training.ddp_scaled_loss`, `training.temperature_sampling_probs`, `training.sample_dataset_indices` | B | 15 | 5 |
| 34 | Initialisation: truncated normal, zero-init outputs and gates | `training.trunc_normal_init_`, `training.zero_init_`, `training.gating_init_`, `training.ZeroInitResidualMLP` | C | 10 | 5 |
| 35 | Debug a training script that runs but does not learn | `training.fixed_train` | checklist | 10 | 3 |

### Structure prediction: losses and sampling (`structure_losses`)

| # | Drill | Symbols | Tier | Min | Tests |
|---|---|---|---|---|---|
| 36 | Distogram target and masked distogram cross-entropy | `structure_losses.distogram_target`, `structure_losses.distogram_loss` | A | 20 | 6 |
| 37 | Weighted Kabsch alignment, aligned weighted MSE and RMSD | `structure_losses.weighted_rigid_align`, `structure_losses.aligned_weighted_mse`, `structure_losses.weighted_rmsd`, `structure_losses.weighted_minimum_rmsd` | A | 35 | 7 |
| 38 | Smooth lDDT loss and hard lDDT | `structure_losses.smooth_lddt_loss`, `structure_losses.lddt` | A | 20 | 6 |
| 39 | EDM diffusion training loss on coordinates | `structure_losses.sample_training_sigma`, `structure_losses.edm_preconditioning`, `structure_losses.edm_loss_weight`, `structure_losses.edm_denoise`, `structure_losses.edm_diffusion_loss` | B | 40 | 6 |
| 40 | Karras noise schedule and stochastic sampler (AF3 Alg. 18) | `structure_losses.karras_schedule`, `structure_losses.karras_sample` | C | 20 | 5 |
| 41 | Random SO(3) rotation and centring augmentation (AF3 Alg. 19) | `structure_losses.random_quaternions`, `structure_losses.quaternion_to_matrix`, `structure_losses.center_random_augmentation` | B | 10 | 5 |
| 45 | Binned confidence heads: pLDDT/PDE cross-entropy, decoding, pTM/ipTM | `structure_losses.bin_values`, `structure_losses.expected_value_from_bins`, `structure_losses.plddt_loss`, `structure_losses.pde_loss`, `structure_losses.compute_ptm` | A (pTM: B) | 20 | 7 |

### Structure prediction: layers (`structure_layers`)

| # | Drill | Symbols | Tier | Min | Tests |
|---|---|---|---|---|---|
| 42 | Attention with pair bias and gating | `structure_layers.AttentionPairBias` | A | 20 | 7 |
| 43 | Triangle multiplicative update, outgoing and incoming | `structure_layers.TriangleMultiplicationOutgoing`, `structure_layers.TriangleMultiplicationIncoming` | A | 15 | 7 |
| 44 | Outer product mean (MSA to pair), with a chunked version | `structure_layers.OuterProductMean` | B | 15 | 7 |
| 46 | Relative position encoding for multi-chain inputs (AF3 Alg. 3) | `structure_layers.relative_position_features`, `structure_layers.RelativePositionEncoder` | C | 20 | 4 |
| 47 | AdaLN, conditioned SwiGLU transition and Fourier noise embedding | `structure_layers.AdaLN`, `structure_layers.ConditionedTransitionBlock`, `structure_layers.FourierEmbedding` | C | 15 | 6 |
| 48 | Pair-weighted averaging | `structure_layers.PairWeightedAveraging` | C | 15 | 4 |
| 49 | Triangle attention, starting node and ending node | `structure_layers.TriangleAttention` | C | 30 | 6 |

### Infrastructure and performance (`infra`)

| # | Drill | Symbols | Tier | Min | Tests |
|---|---|---|---|---|---|
| 50 | Padding collate, pair masks, masked attention and loss, length bucketing | `infra.pad_to_max`, `infra.pair_mask`, `infra.masked_attention`, `infra.masked_mean_loss`, `infra.bucket_by_length`, `infra.padding_fraction` | B | 20 | 10 |
| 51 | Generic chunk_layer and a chunk-size rule | `infra.chunk_layer`, `infra.chunk_size_for` | B | 15 | 3 |
| 52 | Activation checkpointing over a block stack | `infra.CheckpointedStack` | C | 15 | 2 |
| 53 | Recycling with gradients only on the last pass | `infra.Recycler` | C | 10 | 3 |
| 54 | Batched sampling under a memory cap | `infra.sample_in_chunks` | C | 10 | 2 |
| 55 | Simulated ring all-reduce and column-sharded matmul backward | `infra.ring_all_reduce`, `infra.sharded_matmul_backward` | C | 20 | 6 |
| 56 | FP32 to int8 quantisation, symmetric and asymmetric | `infra.quantize_symmetric`, `infra.dequantize_symmetric`, `infra.quantize_asymmetric`, `infra.dequantize_asymmetric` | C | 12 | 4 |
| 57 | Profile a slow step: starved DataLoader and host syncs (simulated) | `infra.simulate_dataloader`, `infra.simulate_cuda_stream` | C | 10 | 4 |
| 58 | Vectorised pairwise distances without building N x K x D | `infra.pairwise_sq_dists`, `infra.nearest_center` | B | 8 | 4 |

### Classic ML (`classic`)

| # | Drill | Symbols | Tier | Min | Tests |
|---|---|---|---|---|---|
| 59 | k-means with k-means++ seeding | `classic.kmeans_plusplus_init`, `classic.kmeans` | B | 20 | 5 |
| 60 | kNN / 1-NN vectorised, and 1-NN as a linear layer | `classic.knn_predict`, `classic.nn1_as_linear` | B | 12 | 4 |
| 61 | Logistic regression by gradient descent | `classic.logistic_loss_and_grad`, `classic.fit_logistic_regression` | B | 10 | 4 |
| 62 | Linear regression: normal equation, GD, ridge, LOO residuals | `classic.ridge_normal_equation`, `classic.linear_regression_gd`, `classic.ridge_loo_residuals` | C | 15 | 5 |
| 63 | PCA via SVD | `classic.pca` | B | 10 | 3 |

### Data and evaluation hygiene (`data`)

| # | Drill | Symbols | Tier | Min | Tests |
|---|---|---|---|---|---|
| 64 | Leakage-aware splits: group, temporal and similarity | `data.group_split`, `data.temporal_split`, `data.flag_similar_to_train`, `data.similarity_components`, `data.similarity_split` | B | 20 | 10 |
| 65 | Vectorised Tanimoto and diversity picking | `data.tanimoto_matrix`, `data.maxmin_pick`, `data.nearest_neighbour_similarity` | C | 15 | 6 |
| 66 | Weighted and temperature multi-dataset sampling | `data.temperature_sampling_probs`, `data.cluster_sample_weights` | B | 5 | 4 |
| 67 | Label-noise detection with out-of-fold probabilities | `data.out_of_fold_probs`, `data.flag_label_issues`, `data.confident_learning_flags` | C | 15 | 6 |

### Extras (`extras`)

| # | Drill | Symbols | Tier | Min | Tests |
|---|---|---|---|---|---|
| 68 | Integrated drill: masked loss + small module + toy loop | `extras.masked_huber_loss`, `extras.TinyRegressor`, `extras.overfit_one_batch` | A | 40 | 6 |
| 69 | Loss from paper pseudocode (AF3 Alg. 27 and Alg. 28) | `extras.smooth_lddt_loss_from_pseudocode`, `extras.weighted_rigid_align_from_pseudocode` | A | 20 | 7 |
| 70 | Tensor fluency warm-up: einsum, broadcasting, gather, scatter | `extras.outgoing_pair_product`, `extras.sequence_mask`, `extras.triu_mask`, `extras.pool_atoms_to_tokens_onehot`, `extras.pool_atoms_to_tokens_scatter`, `extras.gather_tokens_to_atoms` | A | 10 | 6 |
| 71 | Energy and force loss with conservative forces | `extras.PairEnergyModel`, `extras.energy_and_forces`, `extras.energy_force_loss` | B | 15 | 6 |
| 72 | Equivariance and invariance property tests | `extras.random_rotation_matrices`, `extras.apply_rigid_motion`, `extras.is_invariant`, `extras.is_equivariant` | B | 10 | 4 |
| 73 | Batched kNN graph plus one masked message-passing step | `extras.knn_graph`, `extras.mean_aggregate`, `extras.KNNMessagePassing` | B | 15 | 5 |
| 74 | Pairformer layer assembled from parts | `extras.shared_dropout_mask`, `extras.PairformerLayer` | B | 20 | 9 |
| 75 | Min-RMSD over symmetry permutations | `extras.min_rmsd_over_permutations`, `extras.min_over_multiplicity` | C | 15 | 3 |
| 76 | Checkpoint and resume determinism | `extras.ResumableBatchIterator`, `extras.save_training_state`, `extras.load_training_state` | C | 15 | 4 |
| 77 | Head-only fine-tune with a frozen trunk | `extras.freeze_module`, `extras.set_finetune_mode`, `extras.head_param_groups` | C | 10 | 3 |
| 78 | Infra reserve: backoff polling, pagination, resumable download, state machine | `extras.backoff_delay`, `extras.poll_until_terminal`, `extras.fetch_all_pages`, `extras.resumable_download`, `extras.PhaseStateMachine` | C | 20 | 7 |

Drill #35 also ships two helpers that you do not write: `make_debug_data` and `buggy_train`.
`buggy_train` runs but does not learn. Find its bugs and write `fixed_train`.

## Sources

Papers:

- AlphaFold 3: Abramson et al., "Accurate structure prediction of biomolecular interactions with AlphaFold 3", Nature 2024. https://www.nature.com/articles/s41586-024-07487-w
- Boltz-1: Wohlwend et al., "Boltz-1: Democratizing Biomolecular Interaction Modeling", bioRxiv 2024. https://doi.org/10.1101/2024.11.19.624167
- Boltz-2: Passaro et al., "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction", bioRxiv 2025. https://doi.org/10.1101/2025.06.14.659707
- EDM: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022. https://arxiv.org/abs/2206.00364
- Online softmax: Milakov and Gimelshein, "Online normalizer calculation for softmax", 2018. https://arxiv.org/abs/1805.02867
- Memory-efficient attention: Rabe and Staats, "Self-attention Does Not Need O(n^2) Memory", 2021. https://arxiv.org/abs/2112.05682
- Gradient checkpointing: Chen et al., "Training Deep Nets with Sublinear Memory Cost", 2016. https://arxiv.org/abs/1604.06174

Code:

- Boltz: https://github.com/jwohlwend/boltz
- OpenFold: https://github.com/aqlaboratory/openfold
- micrograd: https://github.com/karpathy/micrograd
- nanoGPT: https://github.com/karpathy/nanoGPT

Docs and other practice sets:

- PyTorch automatic mixed precision examples: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- Tensor Puzzles: https://github.com/srush/Tensor-Puzzles
- TorchLeet: https://github.com/Exorust/TorchLeet
- TorchCode: https://github.com/duoan/TorchCode

## License

MIT. See `LICENSE`.
