<div align="center">

# 🧩 FFSegmentation — Forward Segmentation

**A modern image & video segmentation framework — clean, dependency-light, and always up-to-date.**

[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white)](https://discord.gg/22gbzUad)

*Maintained by the [Forward AI Science](https://github.com/Forward-AI-Science) community.*

</div>

---

## What is FFSegmentation?

**FFSegmentation (Forward Segmentation)** is an open-source segmentation framework inspired by [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), but built to stay compatible with modern PyTorch and free of heavy dependency chains.

MMSegmentation is a powerful and well-structured research toolkit, but like the rest of the OpenMMLab ecosystem it requires `mmcv` and `mmengine` — libraries that frequently break with new PyTorch and CUDA releases, require complex source builds, and introduce thousands of lines of abstraction between you and your model.

FFSegmentation strips that away. It takes the clean architectural ideas from MMSegmentation — encoder/decoder separation, unified dataset interfaces, modular heads — and implements them directly on **vanilla PyTorch** with a lean, stable dependency set. The result is a framework anyone can install, read, and extend in minutes.

---

## Motivation

| Pain point in MMSegmentation | FFSegmentation approach |
|---|---|
| Hard dependency on `mmcv` & `mmengine` | Pure PyTorch — no custom ops required |
| Frequent version incompatibilities | Tracks latest stable PyTorch and CUDA |
| Complex source builds | Standard `pip install` |
| Custom registry & config DSL | Plain Python / YAML |
| Heavy abstraction layers | Thin, readable, hackable code |

---

## Scope

FFSegmentation aims to cover the full spectrum of modern segmentation tasks:

- **Semantic segmentation** — assign a class label to every pixel
- **Instance segmentation** — detect and segment each object instance
- **Panoptic segmentation** — unified semantic + instance labelling
- **Video object segmentation** — propagate masks across frames
- **Interactive / promptable segmentation** — SAM-style point/box prompting

---

## Status

> 🚧 **FFSegmentation is in early development.** The repository is being actively built out.
> Star or watch to follow progress.

We are working on:

- Core package structure and coding conventions
- Encoder/decoder architecture (SegFormer, Mask2Former, DeepLab…)
- Dataset loaders (ADE20K, Cityscapes, COCO-Stuff, VOC)
- Training recipes and pretrained model weights
- Documentation

---

## Community

FFSegmentation is maintained by the **[Forward AI Science](https://github.com/Forward-AI-Science)** community — an open group of researchers and engineers who believe that good tools should be simple, transparent, and accessible to everyone.

💬 Join the conversation on **[Discord](https://discord.gg/22gbzUad)** — share ideas, ask questions, and follow development in real time.

We welcome contributions of all kinds: code, documentation, benchmarks, bug reports, and ideas. A `CONTRIBUTING.md` guide will be published alongside the first stable release.

---

## License

FFSegmentation is released under the [Apache 2.0 License](LICENSE).

---

<div align="center">
Made with ❤️ by the <a href="https://github.com/Forward-AI-Science">Forward AI Science</a> community
</div>
