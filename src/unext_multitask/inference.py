#!/usr/bin/env python3
import argparse
import os
from glob import glob
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
import albumentations as A
from albumentations import Compose, Resize
from tqdm import tqdm

import archs
from utils import str2bool

# ------------------------------
# Configuration Dictionary
# ------------------------------

CONFIG = {
    # Model parameters
    'name': 'fullbreast_binary_scratch2',
    
    # Input/Output paths
    'input_dir': '/home/a_amit/Advaith/DMR - Database For Mastology Research/Healthy_all_images',
    'output_dir': './inference_whole_dataset',
    'gt_dir': None,
    
    # Image parameters
    'img_ext': '.png',
    'mask_ext': '.png',
    'threshold': 0.30,
    
    # Output options
    'save_masks': True,
    'save_overlays': True,
    'save_panels': True,
    'save_probs': False,
    'overlay_alpha': 0.5,
    
    # Device
    'device': 'cuda'
}

# ------------------------------
# Constants
# ------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def bgr2rgb(x, **kwargs):
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if x is not None and x.ndim == 3 else x

def _dice_iou(pred, gt):
    """Calculate Dice and IoU metrics"""
    p = pred > 0
    g = gt > 0
    inter = np.logical_and(p, g).sum()
    ps = p.sum(); gs = g.sum()
    if ps == 0 and gs == 0:
        return 1.0, 1.0
    dice = (2.0 * inter) / (ps + gs) if (ps + gs) > 0 else 0.0
    union = ps + gs - inter
    iou  = (inter / union) if union > 0 else 0.0
    return float(dice), float(iou)

def _hd95(pred, gt):
    """Calculate Hausdorff Distance 95th percentile"""
    p = (pred > 0).astype(np.uint8)
    g = (gt   > 0).astype(np.uint8)
    if p.sum() == 0 and g.sum() == 0: return 0.0
    if p.sum() == 0 or g.sum() == 0:  return float('inf')
    k = np.ones((3,3), np.uint8)
    p_edge = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, k)
    g_edge = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, k)
    dt_p = cv2.distanceTransform((1 - p_edge).astype(np.uint8), cv2.DIST_L2, 3)
    dt_g = cv2.distanceTransform((1 - g_edge).astype(np.uint8), cv2.DIST_L2, 3)
    d_pg = dt_g[p_edge.astype(bool)]
    d_gp = dt_p[g_edge.astype(bool)]
    all_d = np.concatenate([d_pg, d_gp]) if d_pg.size and d_gp.size else (d_pg if d_pg.size else d_gp)
    if all_d.size == 0: return 0.0
    return float(np.percentile(all_d, 95))

# ------------------------------
# Post-processing to smooth segmentation
# ------------------------------

def smooth_mask(binary_mask, kernel_size=5, rdp_epsilon=2.0):
    """
    Smooth binary segmentation mask to reduce jagged/wavy boundaries.

    Steps:
    1. Morphological closing to remove small gaps & boundary noise.
    2. Contour extraction + RDP polygon simplification.
    3. Re-fill polygon into a smooth mask.
    """

    # --- (1) Morphological smoothing ---
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, k)

    # --- (2) Extract contours ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return closed

    # Create empty output mask
    smoothed = np.zeros_like(binary_mask)

    # --- (3) Apply RDP polygon simplification ---
    for cnt in contours:
        epsilon = rdp_epsilon
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        cv2.fillPoly(smoothed, [approx], 255)

    return smoothed


# ------------------------------
# Argparse
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on a folder of images')
    
    # Model parameters
    parser.add_argument('--name', default=CONFIG['name'], help='Model name (directory under models/)')
    parser.add_argument('--input_dir', default=CONFIG['input_dir'], help='Input directory containing images')
    parser.add_argument('--output_dir', default=CONFIG['output_dir'], help='Output directory for predictions')
    
    # Image parameters
    parser.add_argument('--img_ext', default=CONFIG['img_ext'], help='Image file extension')
    parser.add_argument('--threshold', default=CONFIG['threshold'], type=float, help='Prediction threshold')
    
    # Ground truth (optional for computing metrics)
    parser.add_argument('--gt_dir', default=CONFIG['gt_dir'], help='Ground truth mask directory (optional)')
    parser.add_argument('--mask_ext', default=CONFIG['mask_ext'], help='Mask file extension')
    
    # Output options
    parser.add_argument('--save_masks', default=CONFIG['save_masks'], type=str2bool, help='Save binary masks')
    parser.add_argument('--save_overlays', default=CONFIG['save_overlays'], type=str2bool, help='Save overlay visualizations')
    parser.add_argument('--save_panels', default=CONFIG['save_panels'], type=str2bool, help='Save side-by-side panels')
    parser.add_argument('--save_probs', default=CONFIG['save_probs'], type=str2bool, help='Save probability maps')
    parser.add_argument('--overlay_alpha', default=CONFIG['overlay_alpha'], type=float, help='Overlay transparency (0-1)')
    
    # Device
    parser.add_argument('--device', default=CONFIG['device'], choices=['cuda', 'cpu'], help='Device to use')
    
    # Option to use config dict only
    parser.add_argument('--use_config', action='store_true', help='Use CONFIG dict instead of command line args')
    
    args = parser.parse_args()
    return args

# ------------------------------
# Main Inference
# ------------------------------

def load_model(model_dir, device='cuda'):
    """Load model from checkpoint and config"""
    config_path = os.path.join(model_dir, 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print("\n" + "="*60)
    print("Model Configuration")
    print("="*60)
    print(f"  Architecture: {config['arch']}")
    print(f"  Input Size: {config['input_w']}x{config['input_h']}")
    print(f"  Classes: {config['num_classes']}")
    print(f"  Deep Supervision: {config['deep_supervision']}")
    print(f"  RGB Mode: {config.get('rgb', True)}")
    print("="*60)
    
    # Build model
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision']
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"\n✅ Model loaded from: {checkpoint_path}\n")
    return model, config

def create_transform(config):
    """Create inference transform pipeline matching training"""
    maybe_rgb = [A.Lambda(image=bgr2rgb)] if config.get('rgb', True) else []
    transform = Compose(maybe_rgb + [
        Resize(config['input_h'], config['input_w']),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform

def preprocess_image(img, transform):
    """Preprocess image for model input"""
    # Apply transforms (expects HWC)
    augmented = transform(image=img)
    img_transformed = augmented['image']
    
    # Convert to CHW tensor
    img_tensor = img_transformed.astype('float32').transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)  # Add batch dimension
    
    return img_tensor

def predict_image(model, img_tensor, config, device='cuda'):
    """Run inference on a single image"""
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        if config['deep_supervision']:
            outputs = model(img_tensor)
            logits = outputs[-1]
        else:
            logits = model(img_tensor)
        
        # Get probability map
        prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    return prob_map

def load_ground_truth(gt_dir, img_id, mask_ext, num_classes=1):
    """Load ground truth mask if available"""
    if gt_dir is None or not os.path.exists(gt_dir):
        return None
    
    gt = None
    # Try class subdirectory first (for num_classes=1, it's in "0/")
    mask_candidates = []
    for i in range(num_classes):
        cand_path = os.path.join(gt_dir, str(i), img_id + mask_ext)
        if os.path.exists(cand_path):
            mask_candidates.append(cand_path)
    
    # Fallback to direct path or recursive search
    if not mask_candidates:
        direct_path = os.path.join(gt_dir, img_id + mask_ext)
        if os.path.exists(direct_path):
            mask_candidates.append(direct_path)
        else:
            mask_candidates = glob(os.path.join(gt_dir, "**", img_id + mask_ext), recursive=True)
    
    if mask_candidates:
        gt_raw = cv2.imread(mask_candidates[0], cv2.IMREAD_UNCHANGED)
        if gt_raw is not None:
            if gt_raw.ndim == 3:
                gt_raw = cv2.cvtColor(gt_raw, cv2.COLOR_BGR2GRAY)
            gt = (gt_raw > 0).astype(np.uint8) * 255
    
    return gt

def save_outputs(img_id, original_img, prob_map, threshold, gt, output_dirs, args):
    """Save prediction outputs in various formats"""
    H, W = original_img.shape[:2]
    
    # Resize prediction to original size
    prob_map_native = cv2.resize(prob_map, (W, H), interpolation=cv2.INTER_LINEAR)
    pred_mask = (prob_map_native >= threshold).astype(np.uint8) * 255
    # pred_mask = smooth_mask(pred_mask, kernel_size=5, rdp_epsilon=2.0)
    
    # Resize ground truth if provided
    if gt is not None:
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Save binary mask
    if args.save_masks:
        mask_path = os.path.join(output_dirs['masks'], f"{img_id}_mask.png")
        cv2.imwrite(mask_path, pred_mask)
    
    # Save probability map
    if args.save_probs:
        prob_uint8 = (prob_map_native * 255).astype(np.uint8)
        prob_path = os.path.join(output_dirs['probs'], f"{img_id}_prob.png")
        cv2.imwrite(prob_path, prob_uint8)
    
    # Save overlay visualization (BOUNDARY ONLY - NO SHADING)
    if args.save_overlays:
        overlay = original_img.copy()
        
        # Draw contours only (no filled mask)
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        
        overlay_path = os.path.join(output_dirs['overlays'], f"{img_id}_overlay.png")
        cv2.imwrite(overlay_path, overlay)
    
    # Save panel (side-by-side with metrics)
    if args.save_panels:
        right = original_img.copy()
        
        # Draw prediction contours (RED)
        pred_cnts, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(right, pred_cnts, -1, (0, 0, 255), 2)
        
        # Draw ground truth contours (BLUE) if available
        if gt is not None:
            gt_cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(right, gt_cnts, -1, (255, 0, 0), 2)
        
        # Create panel
        footer_h = 45
        panel = np.zeros((H + footer_h, W * 2, 3), dtype=np.uint8)
        panel[:H, :W] = original_img
        panel[:H, W:] = right
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        baseline_y = H + 30
        cv2.putText(panel, "Ground Truth", (10, baseline_y), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, "Prediction", (200, baseline_y), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Calculate and display metrics if GT available
        if gt is not None:
            dice, iou = _dice_iou(pred_mask, gt)
            hd95 = _hd95(pred_mask, gt)
            metrics_txt = f"Dice: {dice*100:.1f}%    IoU: {iou*100:.1f}%    HD95: {hd95:.2f}px"
        else:
            metrics_txt = "Dice/IoU/HD95: N/A (no ground truth)"
        
        cv2.putText(panel, metrics_txt, (W + 10, baseline_y), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        panel_path = os.path.join(output_dirs['panels'], f"{img_id}_panel.png")
        cv2.imwrite(panel_path, panel)
    
    # Return metrics if ground truth available
    if gt is not None:
        dice, iou = _dice_iou(pred_mask, gt)
        hd95 = _hd95(pred_mask, gt)
        return {'dice': dice, 'iou': iou, 'hd95': hd95}
    return None

def run_inference(args):
    """Main inference pipeline"""
    
    # Setup paths
    model_dir = os.path.join('models', args.name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Create output directories
    output_dirs = {
        'masks': os.path.join(args.output_dir, 'masks'),
        'overlays': os.path.join(args.output_dir, 'overlays'),
        'panels': os.path.join(args.output_dir, 'panels'),
        'probs': os.path.join(args.output_dir, 'probability_maps'),
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_masks:
        os.makedirs(output_dirs['masks'], exist_ok=True)
    if args.save_overlays:
        os.makedirs(output_dirs['overlays'], exist_ok=True)
    if args.save_panels:
        os.makedirs(output_dirs['panels'], exist_ok=True)
    if args.save_probs:
        os.makedirs(output_dirs['probs'], exist_ok=True)
    
    # Load model
    model, config = load_model(model_dir, device=args.device)
    
    # Create transform
    transform = create_transform(config)
    
    # Get list of images
    print("="*60)
    print("Finding Images")
    print("="*60)
    img_extensions = [args.img_ext, '.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.JPEG']
    
    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob(os.path.join(args.input_dir, f'*{ext}')))
    
    image_paths = sorted(list(set(image_paths)))
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {args.input_dir}")
    
    print(f"Found {len(image_paths)} images")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.gt_dir:
        print(f"Ground truth directory: {args.gt_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Save masks: {args.save_masks}")
    print(f"Save overlays: {args.save_overlays}")
    print(f"Save panels: {args.save_panels}")
    print(f"Save probability maps: {args.save_probs}")
    print("="*60 + "\n")
    
    # Run inference
    print("Running Inference...")
    print("="*60 + "\n")
    
    all_metrics = []
    pbar = tqdm(image_paths, desc="Processing")
    
    for img_path in pbar:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        pbar.set_postfix({'image': img_id[:30]})
        
        # Load original image
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"⚠️  Failed to load {img_path}, skipping...")
            continue
        
        # Handle grayscale or RGBA
        if original_img.ndim == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        elif original_img.shape[2] == 4:
            original_img = original_img[:, :, :3]
        
        # Preprocess
        img_tensor = preprocess_image(original_img, transform)
        
        # Predict
        prob_map = predict_image(model, img_tensor, config, device=args.device)
        
        # Load ground truth if available
        gt = load_ground_truth(args.gt_dir, img_id, args.mask_ext, config['num_classes'])
        
        # Save outputs
        metrics = save_outputs(img_id, original_img, prob_map, args.threshold, gt, output_dirs, args)
        
        if metrics:
            metrics['image_id'] = img_id
            all_metrics.append(metrics)
    
    # Save metrics summary if ground truth was provided
    if all_metrics:
        import pandas as pd
        df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(args.output_dir, 'metrics_summary.csv')
        df.to_csv(metrics_path, index=False)
        
        print("\n" + "="*60)
        print("Metrics Summary")
        print("="*60)
        print(f"Average Dice: {df['dice'].mean()*100:.2f}%")
        print(f"Average IoU: {df['iou'].mean()*100:.2f}%")
        print(f"Average HD95: {df['hd95'].mean():.2f}px")
        print(f"Metrics saved to: {metrics_path}")
    
    print("\n" + "="*60)
    print("✅ Inference Complete!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    if args.save_masks:
        print(f"  - Binary masks: {output_dirs['masks']}")
    if args.save_overlays:
        print(f"  - Overlays: {output_dirs['overlays']}")
    if args.save_probs:
        print(f"  - Probability maps: {output_dirs['probs']}")
    print("="*60 + "\n")

def main():
    args = parse_args()
    
    # If --use_config flag is set, use CONFIG dict instead of argparse
    if args.use_config:
        args = SimpleNamespace(**CONFIG)
        print("🔧 Using CONFIG dictionary (command-line args ignored)\n")
    
    # Verify device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    run_inference(args)

if __name__ == '__main__':
    main()