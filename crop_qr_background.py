# crop_qr_background.py
import os
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def crop_region_around_qr(cv_img, points, pad_factor=2.0):
    if points is None:
        # fallback: return center crop
        h, w = cv_img.shape[:2]
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side//2)
        y1 = max(0, cy - side//2)
        return cv_img[y1:y1+side, x1:x1+side]
    try:
        pts = points.reshape(-1, 2).astype(int)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        w = x_max - x_min
        h = y_max - y_min
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        new_w = int(max(1, w * pad_factor))
        new_h = int(max(1, h * pad_factor))
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(cv_img.shape[1], cx + new_w // 2)
        y2 = min(cv_img.shape[0], cy + new_h // 2)
        return cv_img[y1:y2, x1:x2]
    except Exception:
        return cv_img

def crop_and_save(input_root, output_root, pad_factor=2.0):
    detector = cv2.QRCodeDetector()
    for subset in ['train','val','test']:
        in_subset = os.path.join(input_root, subset)
        out_subset = os.path.join(output_root, subset)
        if not os.path.isdir(in_subset):
            continue
        for label in ['genuine','fake']:
            in_dir = os.path.join(in_subset, label)
            out_dir = os.path.join(out_subset, label)
            os.makedirs(out_dir, exist_ok=True)
            if not os.path.isdir(in_dir):
                continue
            for fname in tqdm(os.listdir(in_dir), desc=f"{subset}/{label}"):
                in_path = os.path.join(in_dir, fname)
                try:
                    img = cv2.imread(in_path)
                    if img is None:
                        continue
                    data, points, _ = detector.detectAndDecode(img)
                    crop = crop_region_around_qr(img, points, pad_factor=pad_factor)
                    # ensure size
                    crop = cv2.resize(crop, (300,300))
                    cv2.imwrite(os.path.join(out_dir, fname), crop)
                except Exception as e:
                    print("Error", in_path, e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset", help="raw dataset root")
    parser.add_argument("--output", default="dataset_crops", help="cropped output root")
    parser.add_argument("--pad", type=float, default=2.0, help="padding factor around QR")
    args = parser.parse_args()
    crop_and_save(args.input, args.output, pad_factor=args.pad)
