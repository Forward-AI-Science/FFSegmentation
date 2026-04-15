# FFSegmentation — Roadmap & TODO

> **Strategy:** Port and adapt directly from MMSegmentation source using an agentic AI coding IDE.
> **Target:** ~1 month, 1 developer.

---

## 🗺️ Roadmap

| Phase | Focus | Estimate |
|---|---|---|
| **Phase 1** | Repo setup & package skeleton | Week 1 |
| **Phase 2** | Core framework — encoders, decoders, datasets | Weeks 1–2 |
| **Phase 3** | Training pipeline, evaluation & first models | Weeks 2–3 |
| **Phase 4** | Model zoo, export, docs & first release | Weeks 3–4 |

---

## Phase 1 — Repo Setup `Week 1`

- [ ] Package structure & `pyproject.toml`
- [ ] Linting, formatting, pre-commit hooks
- [ ] GitHub Actions CI (lint + tests)
- [ ] `CONTRIBUTING.md` and `CHANGELOG.md`

---

## Phase 2 — Core Framework `Weeks 1–2`

### Datasets
- [ ] Base dataset class & augmentation pipeline
- [ ] ADE20K
- [ ] Cityscapes
- [ ] COCO-Stuff / COCO Panoptic
- [ ] Pascal VOC
- [ ] Custom dataset interface

### Model Architecture

#### Encoders (Backbones)
- [ ] Backbone wrapper (timm — ResNet, Swin, ViT)
- [ ] MiT (Mix Transformer — SegFormer backbone)
- [ ] ConvNeXt backbone

#### Decoders / Heads
- [ ] All-MLP decode head (SegFormer)
- [ ] UPerNet decode head
- [ ] DeepLabV3+ ASPP head
- [ ] Mask2Former pixel decoder + transformer head
- [ ] SAM-style promptable mask head

### Losses
- [ ] Cross-entropy loss (weighted)
- [ ] Dice loss
- [ ] Focal loss
- [ ] Binary cross-entropy (instance)
- [ ] Panoptic quality loss

---

## Phase 3 — Training & Evaluation `Weeks 2–3`

- [ ] Core training loop (AMP, gradient clipping, EMA)
- [ ] Learning rate scheduler (poly, cosine, warmup)
- [ ] Config system (YAML + dataclasses)
- [ ] Checkpoint save & resume
- [ ] Experiment logging (W&B / TensorBoard)
- [ ] mIoU metric
- [ ] Pixel accuracy metric
- [ ] Panoptic quality (PQ) metric
- [ ] Evaluation CLI (`tools/test.py`)
- [ ] Training CLI (`tools/train.py`)
- [ ] Unit & integration tests

---

## Phase 4 — Model Zoo, Export & Release `Weeks 3–4`

- [ ] Train & release SegFormer-b0/b2/b5 on ADE20K / Cityscapes
- [ ] Train & release Mask2Former on COCO Panoptic
- [ ] `from_pretrained()` API + Hugging Face Hub upload
- [ ] Inference demo (`tools/demo.py`)
- [ ] Segmentation mask visualization utilities
- [ ] ONNX export
- [ ] TorchScript export
- [ ] Docker image
- [ ] Documentation site (MkDocs / GitHub Pages)
- [ ] Gradio / HF Spaces web demo
- [ ] Tag `v0.1.0` + release notes
