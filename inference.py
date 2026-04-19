"""
inference.py  –  Run semantic segmentation inference with mmsegmentation
                 (pure-PyTorch port, no mmengine/mmcv installation needed)

Usage examples
--------------
# Single image → save coloured mask alongside original
python inference.py demo/demo.png \
    --config  configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    --checkpoint checkpoints/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth \
    --out-dir  outputs/

# Show class-label overlay in the terminal (saves nothing)
python inference.py demo/demo.png \
    --config  configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    --checkpoint checkpoints/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth \
    --show-info

# Multiple images from a directory
python inference.py /path/to/images/ \
    --config  configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    --checkpoint checkpoints/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth \
    --out-dir  outputs/

# Override device
python inference.py demo/demo.png \
    --config  configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    --checkpoint checkpoints/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth \
    --device cpu \
    --out-dir  outputs/

# Use checkpoint URL (auto-downloads to ~/.cache/mmseg/checkpoints/)
python inference.py demo/demo.png \
    --config  configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x1024_40k_cityscapes/fcn_r50-d8_512x1024_40k_cityscapes_20200604_192608-efe53f0d.pth \
    --out-dir  outputs/
"""

import argparse
import os
import sys

# ── ensure repo root is on sys.path so our in-repo mmengine/mmcv are found ──
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from PIL import Image

from mmseg.apis import init_model, inference_model
from mmseg.utils import get_classes, get_palette


# ─────────────────────────────────── helpers ──────────────────────────────────

def _collect_images(path: str):
    """Return a list of image file paths from a file path or directory."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in exts)
    raise FileNotFoundError(f'Input not found: {path}')


def _label_to_color(seg_map: np.ndarray, palette: list) -> np.ndarray:
    """Map integer label map → RGB colour image using the dataset palette."""
    h, w = seg_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_img[seg_map == label] = color
    return color_img


def _blend(image_bgr: np.ndarray, color_map: np.ndarray,
           alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend colour segmentation mask onto the original image (RGB out)."""
    img_rgb = image_bgr[:, :, ::-1].astype(np.float32)  # BGR → RGB
    blended = (1 - alpha) * img_rgb + alpha * color_map.astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)


def _read_image_rgb(path: str) -> np.ndarray:
    """Read image as RGB numpy array (H, W, 3)."""
    return np.array(Image.open(path).convert('RGB'))


def save_result(img_path: str, seg_map: np.ndarray,
                classes: list, palette: list,
                out_dir: str, alpha: float = 0.5) -> str:
    """Overlay segmentation on the original image and save both mask and blend."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(img_path))[0]

    # Colour mask
    color_map = _label_to_color(seg_map, palette)

    # Blend with original
    img_rgb = _read_image_rgb(img_path)
    # Resize mask to match original if sizes differ (e.g. slide inference)
    if color_map.shape[:2] != img_rgb.shape[:2]:
        pil_mask = Image.fromarray(color_map).resize(
            (img_rgb.shape[1], img_rgb.shape[0]), Image.NEAREST)
        color_map = np.array(pil_mask)
    blended = _blend(img_rgb[:, :, ::-1], color_map, alpha=alpha)  # pass BGR for _blend

    # Actually _blend expects BGR, so pass RGB and keep simple
    blended = ((1 - alpha) * img_rgb + alpha * color_map).clip(0, 255).astype(np.uint8)

    blend_path = os.path.join(out_dir, f'{stem}_blend.png')
    mask_path  = os.path.join(out_dir, f'{stem}_mask.png')
    Image.fromarray(blended).save(blend_path)
    Image.fromarray(color_map).save(mask_path)
    return blend_path, mask_path


def print_class_stats(seg_map: np.ndarray, classes: list):
    """Print percentage coverage of each class present in the prediction."""
    total = seg_map.size
    unique, counts = np.unique(seg_map, return_counts=True)
    print(f"\n  {'Class':<30} {'Pixels':>8}  {'%':>6}")
    print(f"  {'-'*30} {'-'*8}  {'-'*6}")
    for lbl, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        name = classes[lbl] if lbl < len(classes) else f'class_{lbl}'
        print(f"  {name:<30} {cnt:>8}  {cnt/total*100:>5.1f}%")
    print()


# ──────────────────────────────────── main ────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run segmentation inference (pure-PyTorch mmseg port)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('input',
                        help='Image file, numpy array path, or directory of images')
    parser.add_argument('--config', '-c', required=True,
                        help='Path to model config file (.py)')
    parser.add_argument('--checkpoint', '-w', required=True,
                        help='Path to checkpoint .pth file, or a URL '
                             '(auto-downloaded to ~/.cache/mmseg/checkpoints/)')
    parser.add_argument('--out-dir', '-o', default=None,
                        help='Directory to save blended overlay and mask images. '
                             'If omitted, results are printed but not saved.')
    parser.add_argument('--device', '-d', default=None,
                        help='Device string, e.g. "cuda:0" or "cpu". '
                             'Defaults to cuda:0 if available, else cpu.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Opacity of the segmentation overlay (0-1). Default: 0.5')
    parser.add_argument('--show-info', action='store_true',
                        help='Print per-class pixel coverage statistics to stdout.')
    parser.add_argument('--palette', default=None,
                        help='Dataset name for colour palette override '
                             '(e.g. "cityscapes", "ade20k"). '
                             'Auto-detected from checkpoint metadata by default.')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if args.device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'Using device: {device}')

    # ── load model ────────────────────────────────────────────────────────────
    print(f'Loading config:      {args.config}')
    print(f'Loading checkpoint:  {args.checkpoint}')
    model = init_model(args.config, args.checkpoint, device=device)
    print(f'Model loaded:        {type(model).__name__}')

    # ── classes / palette ─────────────────────────────────────────────────────
    dataset_meta = getattr(model, 'dataset_meta', {})
    classes = dataset_meta.get('classes', get_classes('cityscapes'))
    palette = dataset_meta.get('palette', get_palette('cityscapes'))

    if args.palette:
        try:
            palette = get_palette(args.palette)
            classes = get_classes(args.palette)
        except KeyError:
            print(f'[warn] Unknown palette "{args.palette}", using detected palette.')

    print(f'Dataset classes:     {len(classes)}  (e.g. {classes[:3]}...)')

    # ── collect images ────────────────────────────────────────────────────────
    image_paths = _collect_images(args.input)
    print(f'Images to process:   {len(image_paths)}')

    # ── inference ─────────────────────────────────────────────────────────────
    for idx, img_path in enumerate(image_paths):
        print(f'\n[{idx+1}/{len(image_paths)}] {img_path}')

        result = inference_model(model, img_path)

        # seg_map: (H, W) int64 label indices
        seg_map = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.int32)
        print(f'  Seg map shape: {seg_map.shape}  '
              f'labels: {sorted(np.unique(seg_map).tolist())}')

        if args.show_info:
            print_class_stats(seg_map, classes)

        if args.out_dir:
            blend_path, mask_path = save_result(
                img_path, seg_map, classes, palette,
                out_dir=args.out_dir, alpha=args.alpha)
            print(f'  Saved blend: {blend_path}')
            print(f'  Saved mask:  {mask_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
