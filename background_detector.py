# background_detector.py
import os
import cv2
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf

detector = cv2.QRCodeDetector()

def crop_background_from_pil(pil_img, pad=2.0, out_size=(224,224)):
    cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    data, points, _ = detector.detectAndDecode(cv_img)
    # cropping as before
    if points is None:
        h,w = cv_img.shape[:2]
        side = min(h,w)
        cx,cy = w//2, h//2
        x1 = max(0, cx-side//2); y1 = max(0, cy-side//2)
        crop = cv_img[y1:y1+side, x1:x1+side]
    else:
        pts = points.reshape(-1,2).astype(int)
        x_min,y_min = pts.min(axis=0); x_max,y_max = pts.max(axis=0)
        w = x_max-x_min; h = y_max-y_min
        cx,cy = (x_min+x_max)//2, (y_min+y_max)//2
        new_w = int(max(1, w*pad)); new_h = int(max(1, h*pad))
        x1 = max(0, cx-new_w//2); y1 = max(0, cy-new_h//2)
        x2 = min(cv_img.shape[1], cx+new_w//2); y2 = min(cv_img.shape[0], cy+new_h//2)
        crop = cv_img[y1:y2, x1:x2]
    crop = cv2.resize(crop, out_size)
    return crop

# load models if exist
def load_models(cnn_path="models/cnn_model.h5", rf_path="models/rf_model.joblib"):
    cnn = None
    rf = None
    if os.path.exists(cnn_path):
        cnn = tf.keras.models.load_model(cnn_path)
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
    return cnn, rf

# prediction wrapper (uses CNN if present else RF)
def predict_background(pil_img, templates=None, cnn=None, rf=None):
    crop = crop_background_from_pil(pil_img, pad=2.0, out_size=(224,224))
    if cnn is not None:
        x = crop.astype("float32")/255.0
        x = np.expand_dims(x, 0)
        p = float(cnn.predict(x)[0,0])
        # model outputs probability of "fake" (if you trained that way); adjust accordingly
        return p, ("fake" if p>0.5 else "genuine")
    elif rf is not None and templates is not None:
        # compute features (phash, ssim, orb) relative to templates
        from train_rf import load_templates, phash_sim, ssim_sim, orb_matches
        timgs = load_templates(templates)
        best_phash = 1e9; best_ssim = -1; best_orb = 0
        for t in timgs:
            ph = phash_sim(crop, t)
            ss = ssim_sim(crop, t)
            om = orb_matches(crop, t)
            if ph < best_phash: best_phash = ph
            if ss > best_ssim: best_ssim = ss
            if om > best_orb: best_orb = om
        feats = np.array([[best_phash, best_ssim, best_orb]])
        p = rf.predict_proba(feats)[0,1]
        return p, ("fake" if p>0.5 else "genuine")
    else:
        return None, "no_model"
