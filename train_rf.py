# train_rf.py
import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import argparse
from tqdm import tqdm

def load_templates(template_dir):
    templates = []
    if not os.path.isdir(template_dir):
        return templates
    for f in os.listdir(template_dir):
        p = os.path.join(template_dir, f)
        img = cv2.imread(p)
        if img is not None:
            templates.append(img)
    return templates

def phash_sim(img1, img2):
    h1 = imagehash.phash(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)))
    h2 = imagehash.phash(Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)))
    return abs(h1 - h2)

def ssim_sim(img1, img2):
    g1 = cv2.cvtColor(cv2.resize(img1,(256,256)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(img2,(256,256)), cv2.COLOR_BGR2GRAY)
    return ssim(g1,g2)

def orb_matches(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

def extract_features_for_image(img_path, templates):
    img = cv2.imread(img_path)
    if img is None:
        return None
    best_phash = 1e9
    best_ssim = -1
    best_orb = 0
    for t in templates:
        try:
            ph = phash_sim(img, t)
            ss = ssim_sim(img, t)
            om = orb_matches(img, t)
            if ph < best_phash: best_phash = ph
            if ss > best_ssim: best_ssim = ss
            if om > best_orb: best_orb = om
        except Exception:
            continue
    return [best_phash, best_ssim, best_orb]

def build_dataset(data_dir, templates_dir):
    templates = load_templates(templates_dir)
    X = []
    y = []
    for subset in ['train','val','test']:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.isdir(subset_dir): continue
        for label in ['genuine','fake']:
            d = os.path.join(subset_dir, label)
            if not os.path.isdir(d): continue
            for f in tqdm(os.listdir(d), desc=f"{subset}/{label}"):
                path = os.path.join(d,f)
                feats = extract_features_for_image(path, templates)
                if feats is None: continue
                X.append(feats)
                y.append(0 if label=='genuine' else 1)
    return np.array(X), np.array(y)

def main(data_dir, templates_dir, out_model):
    X, y = build_dataset(data_dir, templates_dir)
    # simple split (train/val/test are already included) — we'll train on all training rows
    # For simplicity here we'll train on full X and y (use better splits in practice)
    # But let's use a hold-out where test subset exists:
    # For now train RF on all X,y:
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X,y)
    joblib.dump(clf, out_model)
    print("Saved RF model:", out_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset_crops")
    parser.add_argument("--templates", default="templates")
    parser.add_argument("--out", default="models/rf_model.joblib")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    main(args.data_dir, args.templates, args.out)
