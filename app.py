"""
urinalysis_webapp/app.py
Flask Web应用 - 适配Render.com部署
"""
import json
import os
import uuid
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# 导入核心功能
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_rgb import mask_info, extract_rgb_from_region, COLOR_NAMES, CLS_BLUE, CLS_COLOR

app = Flask(__name__)
CORS(app)

PORT = int(os.environ.get('PORT', 5000))

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
RESULTS_FOLDER = BASE_DIR / 'results'

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

MODEL_PATH = BASE_DIR / "runs" / "segment" / "train_seg_v2" / "weights" / "best.pt"

model = None

def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
        print(f"✅ 加载模型: {MODEL_PATH}")
        os.environ['YOLO_VERBOSE'] = 'False'
        model = YOLO(str(MODEL_PATH))
        print("✅ 模型加载完成")
    return model

def process_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    H, W = img.shape[:2]
    
    model = load_model()
    results = model.predict(
        source=str(image_path),
        imgsz=1024,
        conf=0.25,
        verbose=False,
    )
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        raise ValueError("未检测到任何色块")
    
    if result.masks is None:
        raise ValueError("模型未输出分割掩码")
    
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    masks_xy = result.masks.xy
    
    blue_indices = np.where(cls_ids == CLS_BLUE)[0]
    color_indices = np.where(cls_ids == CLS_COLOR)[0]
    
    if len(blue_indices) == 0:
        raise ValueError("未检测到blue色块")
    if len(color_indices) == 0:
        raise ValueError("未检测到color色块")
    
    best_blue_idx = blue_indices[np.argmax(confs[blue_indices])]
    blue_polygon = np.array(masks_xy[best_blue_idx])
    blue_info = mask_info(blue_polygon)
    blue_cx, blue_cy = blue_info["cx"], blue_info["cy"]
    
    color_data = []
    for idx in color_indices:
        polygon = np.array(masks_xy[idx])
        info = mask_info(polygon)
        cx, cy = info["cx"], info["cy"]
        dist = float(np.sqrt((cx - blue_cx) ** 2 + (cy - blue_cy) ** 2))
        color_data.append({
            "polygon": polygon.tolist(),
            "info": info,
            "conf": float(confs[idx]),
            "distance": dist,
        })
    
    color_data.sort(key=lambda x: x["distance"])
    for i, item in enumerate(color_data):
        if i < len(COLOR_NAMES):
            item["name"] = COLOR_NAMES[i]
        else:
            item["name"] = f"color_extra_{i + 1}"
    
    for item in color_data:
        info = item["info"]
        item["rgb"] = extract_rgb_from_region(img, info["cx"], info["cy"],
                                              info["w"] / 2, info["h"] / 2)
    
    return {
        "image_size": {"width": W, "height": H},
        "blue": {
            "centroid": {"x": round(blue_cx, 1), "y": round(blue_cy, 1)},
            "mask_bbox": {"w": round(blue_info["w"], 1), "h": round(blue_info["h"], 1)},
            "confidence": round(float(confs[best_blue_idx]), 3),
            "polygon": blue_polygon.tolist(),
        },
        "colors": color_data,
    }

def generate_visualization(img_path, result_data, output_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    
    vis = img.copy()
    blue_polygon = np.array(result_data["blue"]["polygon"])
    blue_info = result_data["blue"]
    
    pts = blue_polygon.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(vis, [pts], True, (255, 0, 0), 3)
    bx, by = blue_info["centroid"]["x"], blue_info["centroid"]["y"]
    cv2.putText(vis, "BLUE (ref)", (int(bx), int(by) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    box_colors = [
        (0, 255, 0),    # green
        (0, 255, 255),  # yellow
        (0, 165, 255),  # orange
        (0, 0, 255),    # red
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
    ]
    
    for i, item in enumerate(result_data["colors"]):
        polygon = np.array(item["polygon"])
        info = item["info"]
        cx, cy = info["cx"], info["cy"]
        w, h = info["w"], info["h"]
        name = item["name"]
        rgb = item["rgb"]
        color = box_colors[i % len(box_colors)]
        
        pts = polygon.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(vis, [pts], True, color, 2)
        
        rw, rh = w / 2, h / 2
        rx1, ry1 = int(cx - rw / 2), int(cy - rh / 2)
        rx2, ry2 = int(cx + rw / 2), int(cy + rh / 2)
        overlay = vis.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), color, -1)
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), color, 1)
        
        label = f"{name} RGB({rgb['r']:.0f},{rgb['g']:.0f},{rgb['b']:.0f})"
        cv2.putText(vis, label, (int(cx - w / 2), int(cy - h / 2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.line(vis, (int(bx), int(by)), (int(cx), int(cy)), color, 1, cv2.LINE_AA)
        
        mid_x = int((bx + cx) / 2)
        mid_y = int((by + cy) / 2)
        cv2.putText(vis, f"{item['distance']:.0f}px", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite(str(output_path), vis)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        save_name = f"{timestamp}_{unique_id}_{filename}"
        upload_path = app.config['UPLOAD_FOLDER'] / save_name
        file.save(str(upload_path))
        
        result_data = process_image(upload_path)
        
        vis_filename = f"vis_{save_name}.jpg"
        vis_path = app.config['RESULTS_FOLDER'] / vis_filename
        generate_visualization(upload_path, result_data, vis_path)
        
        response_data = {
            'success': True,
            'image': {
                'original': f"/uploads/{save_name}",
                'visualization': f"/results/{vis_filename}",
            },
            'blue': result_data['blue'],
            'colors': [
                {
                    'name': item['name'],
                    'distance': round(item['distance'], 1),
                    'rgb': item['rgb'],
                    'confidence': round(item['conf'], 3),
                    'centroid': item['info']['centroid'],
                    'bbox': item['info']['bbox'],
                }
                for item in result_data['colors']
            ],
            'image_size': result_data['image_size'],
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        load_model()
        return jsonify({'status': 'healthy', 'model_loaded': True})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    try:
        load_model()
        print(f"🚀 服务启动在端口 {PORT}")
    except Exception as e:
        print(f"⚠️ 模型加载失败: {e}")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
