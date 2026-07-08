#!/usr/bin/env python3
"""
PaddleOCR v6 Web Service (CPU Optimized)
========================================
默认使用 tiny 模型 + Waitress 生产模式，直接运行:
     python3 web_ocr.py

切换模型或端口:
     python3 web_ocr.py --port 8080 --model small
     python3 web_ocr.py --port 8080 --model medium --dev
"""

import os
import sys
import io
import argparse

# ========================================================================
# 1. CPU 环境关键配置 (必须在 import paddleocr 之前设置)
# ========================================================================
NCPU = os.cpu_count() or 4
OMP_THREADS = min(NCPU, 8)

os.environ['OMP_NUM_THREADS'] = str(OMP_THREADS)
os.environ['KMP_AFFINITY'] = 'disabled'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = 'False'

# ========================================================================
# 2. 依赖导入
# ========================================================================
import paddle
try:
    paddle.set_flags({'FLAGS_use_mkldnn': 0})
except Exception:
    pass

from flask import Flask, render_template_string, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# ========================================================================
# 3. Flask App 初始化
# ========================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ========================================================================
# 4. PP-OCRv6 模型配置
# ========================================================================
V6_MODELS = {
    'tiny':   ('PP-OCRv6_tiny_det',   'PP-OCRv6_tiny_rec'),
    'small':  ('PP-OCRv6_small_det',  'PP-OCRv6_small_rec'),
    'medium': ('PP-OCRv6_medium_det', 'PP-OCRv6_medium_rec'),
}

ocr = None

def get_ocr(model_name):
    global ocr
    if ocr is None:
        det_model, rec_model = V6_MODELS[model_name]
        print(f"[INFO] Loading model: {det_model} + {rec_model}...")
        ocr = PaddleOCR(
            lang='ch',
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        print("[INFO] Model loaded successfully.")
    return ocr

# ========================================================================
# 5. HTML 前端
# ========================================================================
INDEX_HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PaddleOCR - 文字识别 (CPU版)</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; min-height: 100vh; }
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 24px; text-align: center; }
.header h1 { font-size: 28px; margin-bottom: 8px; }
.header p { opacity: 0.85; font-size: 14px; }
.container { max-width: 900px; margin: 0 auto; padding: 32px 16px; }
.upload-zone { background: white; border: 3px dashed #d0d5dd; border-radius: 16px; padding: 48px 24px; text-align: center; cursor: pointer; transition: all 0.2s; position: relative; }
.upload-zone:hover, .upload-zone.drag-over { border-color: #667eea; background: #f8f7ff; }
.upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.upload-icon { font-size: 48px; margin-bottom: 16px; }
.upload-zone h2 { font-size: 18px; color: #333; margin-bottom: 8px; }
.upload-zone p { color: #888; font-size: 13px; }
.btn { display: inline-block; padding: 12px 32px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 10px; font-size: 15px; cursor: pointer; margin-top: 16px; transition: opacity 0.2s; }
.btn:hover { opacity: 0.9; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.preview { background: white; border-radius: 16px; padding: 24px; margin-top: 24px; display: none; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.preview img { max-width: 100%; border-radius: 8px; border: 1px solid #eee; }
.result-section { background: white; border-radius: 16px; padding: 24px; margin-top: 24px; display: none; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.result-section h3 { font-size: 16px; color: #333; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #667eea; }
.result-item { display: flex; align-items: flex-start; padding: 12px 16px; background: #f8f9fa; border-radius: 10px; margin-bottom: 8px; }
.result-num { color: #667eea; font-weight: bold; font-size: 13px; min-width: 28px; }
.result-text { color: #333; font-size: 14px; line-height: 1.6; flex: 1; }
.result-conf { color: #888; font-size: 11px; margin-left: 12px; white-space: nowrap; }
.loading { text-align: center; padding: 32px; display: none; }
.spinner { display: inline-block; width: 40px; height: 40px; border: 3px solid #e0e0e0; border-top-color: #667eea; border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading p { color: #888; margin-top: 12px; font-size: 14px; }
.error { background: #fff2f0; border: 1px solid #ffccc7; color: #cf1322; padding: 16px; border-radius: 10px; margin-top: 16px; font-size: 14px; display: none; }
.full-text { width: 100%; min-height: 120px; padding: 12px; border: 1px solid #e0e0e0; border-radius: 10px; font-size: 14px; resize: vertical; margin-top: 16px; font-family: inherit; }
.copy-btn { margin-top: 8px; padding: 8px 16px; background: #f0f0f0; border: 1px solid #ddd; border-radius: 6px; cursor: pointer; font-size: 13px; }
.copy-btn:hover { background: #e0e0e0; }
</style>
</head>
<body>
<div class="header">
 <h1>PaddleOCR 文字识别 (CPU版)</h1>
 <p>上传图片，自动识别中英文及混合文本</p>
</div>
<div class="container">
 <div class="upload-zone" id="uploadZone">
 <div class="upload-icon">&#x1F4C4;</div>
 <h2>点击或拖拽上传图片</h2>
 <p>支持 JPG、PNG、BMP 格式，最大 16MB</p>
 <input type="file" id="fileInput" accept="image/*">
 </div>
 <div class="loading" id="loading">
 <div class="spinner"></div>
 <p>正在识别文字...</p>
 </div>
 <div class="error" id="error"></div>
 <div class="preview" id="preview">
 <h3 style="font-size: 16px; color: #333; margin-bottom: 16px;">上传的图片</h3>
 <img id="previewImg" src="" alt="preview">
 </div>
 <div class="result-section" id="resultSection">
 <h3>识别结果</h3>
 <div id="results"></div>
 <textarea class="full-text" id="fullText" readonly placeholder="完整文本..."></textarea>
 <button class="copy-btn" id="copyBtn">复制全部文本</button>
 </div>
</div>
<script>
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('previewImg');
const resultSection = document.getElementById('resultSection');
const results = document.getElementById('results');
const fullText = document.getElementById('fullText');

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('drag-over'); handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

function handleFile(file) {
 if (!file) return;
 if (!file.type.startsWith('image/')) { showError('请上传图片文件'); return; }
 if (file.size > 16 * 1024 * 1024) { showError('文件大小不能超过 16MB'); return; }
 error.style.display = 'none';
 loading.style.display = 'block';
 resultSection.style.display = 'none';
 const reader = new FileReader();
 reader.onload = e => {
 previewImg.src = e.target.result;
 preview.style.display = 'block';
 uploadImage(file);
 };
 reader.readAsDataURL(file);
}

function uploadImage(file) {
 const formData = new FormData();
 formData.append('file', file);
 fetch('/ocr', { method: 'POST', body: formData })
 .then(r => r.json())
 .then(data => {
 loading.style.display = 'none';
 if (data.error) { showError(data.error); return; }
 displayResults(data.results);
 })
 .catch(err => { loading.style.display = 'none'; showError('识别失败: ' + err.message); });
}

function displayResults(items) {
 resultSection.style.display = 'block';
 results.innerHTML = items.map((item, i) =>
 `<div class="result-item">
 <span class="result-num">${i + 1}</span>
 <span class="result-text">${escapeHtml(item.text)}</span>
 <span class="result-conf">${(item.confidence * 100).toFixed(1)}%</span>
 </div>`
 ).join('');
 fullText.value = items.map(i => i.text).join('\\n');
}

function showError(msg) { error.textContent = msg; error.style.display = 'block'; loading.style.display = 'none'; }
function escapeHtml(text) { const d = document.createElement('div'); d.textContent = text; return d.innerHTML; }
document.getElementById('copyBtn').addEventListener('click', () => {
 fullText.select(); document.execCommand('copy');
 const btn = document.getElementById('copyBtn');
 btn.textContent = '已复制!';
 setTimeout(() => btn.textContent = '复制全部文本', 2000);
});
</script>
</body>
</html>'''

# ========================================================================
# 6. Flask 路由
# ========================================================================
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': '请上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '请选择文件'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)

        ocr_instance = get_ocr(args.model)
        result = list(ocr_instance.predict(img_array))

        items = []
        if result:
            for page in result:
                rec_texts = page.get('rec_texts', [])
                rec_scores = page.get('rec_scores', [])
                for i, text in enumerate(rec_texts):
                    conf = rec_scores[i] if i < len(rec_scores) else 0
                    items.append({'text': text, 'confidence': round(float(conf), 4)})

        return jsonify({'results': items})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========================================================================
# 7. 主入口 & 启动参数
# ========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='PaddleOCR v6 Web Service (CPU)')
    parser.add_argument('--port', type=int, default=8080, help='监听端口 (默认: 8080)')
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium'],
                        help='PP-OCRv6 模型档位 (默认: tiny)')
    parser.add_argument('--dev', action='store_true', help='启用开发模式 (使用 Werkzeug，默认使用 Waitress)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    print("=" * 50)
    print(f"PaddlePaddle Version: {paddle.__version__}")
    print(f"CPU Cores: {NCPU}")
    print(f"OMP Threads: {OMP_THREADS}")
    print(f"Model: PP-OCRv6 ({args.model})")
    print(f"Mode: {'Development (Werkzeug)' if args.dev else 'Production (Waitress)'}")
    print(f"Listening: http://0.0.0.0:{args.port}")
    print("=" * 50)

    if args.dev:
        app.run(host='0.0.0.0', port=args.port, debug=False)
    else:
        try:
            from waitress import serve
            serve(app, host='0.0.0.0', port=args.port, threads=OMP_THREADS)
        except ImportError:
            print("[ERROR] Waitress 未安装，请执行: pip install waitress")
            sys.exit(1)
