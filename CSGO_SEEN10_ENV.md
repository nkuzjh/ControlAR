# ControlAR Seen-10 environment and official assets

## Isolated environment

The active environment is the project-local conda prefix `.venv`. It was copied
from the host's working Python 3.11 environment, then the copied UniLIP editable
`.pth` entry was removed. Python packages load from `.venv/lib/python3.11`, and
the source UniLIP environment was left unchanged.

```bash
/home/jiahao/miniconda3/bin/conda create --copy \
  --prefix /home/jiahao/task/ControlAR/.venv \
  --clone /home/jiahao/miniconda3/envs/UniLIP -y
./scripts/setup_csgo_seen10.sh
```

The setup script checks the copied prefix and installs
`requirements-csgo-seen10.txt` into it. This host reports an RTX PRO 6000
Blackwell GPU, driver 580.173.02 and system CUDA 13.0. The copied runtime is
Python 3.11.14, Torch `2.11.0.dev20260124+cu128`, torchvision
`0.25.0.dev20260124+cu128`, with Torch CUDA 12.8 and `sm_120` in its compiled
architecture list. `torch.cuda.is_available()` returned true. Imports for
torchvision, Transformers, Accelerate, safetensors, Hugging Face Hub, timm,
einops, OpenCV, and Pillow passed; a 4×4 CUDA matrix operation returned 64.

If the copied prefix and clone source are unavailable on another host, set
`CONTROLAR_BOOTSTRAP_PYTHON` to an existing Python 3.10–3.12 executable. The
setup script creates an isolated venv without system site packages and installs
the stable PyTorch 2.7.1/torchvision 0.22.1 cu128 wheels before the remaining
requirements. The [official PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/)
document Blackwell support and prebuilt CUDA 12.8 wheels beginning with 2.7.

## Official model revisions

The download helper pins the revisions below and transfers files through the
official Hugging Face `resolve/<revision>` endpoints. It uses four resumable
HTTP Range workers by default, reuses complete partial ranges, and only promotes
an assembled checkpoint after its SHA-256 matches the repository's LFS metadata.
Ranges are interleaved across assets by chunk index so the smaller VQ download
starts while Canny ranges continue.

| Asset | Pinned official revision | Size | Official SHA-256 | Project path |
|---|---|---:|---|---|
| ControlAR Canny MR | `wondervictor/ControlAR@22cecd7a873db8df97ae2b2dc88befee72e97a3a` | 3,356,608,032 bytes | `ef59b3c51e582e4742406480fb81160044b902bd46b2f00d923734800258545e` | `checkpoints/t2i/canny_MR.safetensors` |
| LlamaGen T2I VQ-16 | `peizesun/llamagen_t2i@276f5c5a3d915b922899a03f1912605531574747` | 287,920,306 bytes | `0e21fc1318e2e9ee641a07bdad0e20675e9ec35e6e3eb911d58b5d7a2cd8d4cb` | `checkpoints/vq/vq_ds16_t2i.pt` |
| DINOv2-small | `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056` | 88,249,960 bytes | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` | `autoregressive/models/dinov2-small/model.safetensors` |

The DINO directory also contains the required `config.json` and
`preprocessor_config.json` from the same pinned revision. Their official Git
blob IDs are `5664b325e6258d3960fad8c4c1cff958f3cc2272` and
`ff5b47c2edcd1d3556d63c01a65d93b58b9efce1`, respectively. The local DINO
model, VQ checkpoint, and Canny checkpoint all passed their pinned SHA-256
checks at the listed sizes. Both DINO JSON files passed their pinned Git blob
checks. The downloader exited successfully after verifying every asset.

Revision and file metadata were read from the official Hub model API, for
example `https://huggingface.co/api/models/wondervictor/ControlAR?blobs=true`,
`https://huggingface.co/api/models/peizesun/llamagen_t2i?blobs=true`, and
`https://huggingface.co/api/models/facebook/dinov2-small?blobs=true`.
