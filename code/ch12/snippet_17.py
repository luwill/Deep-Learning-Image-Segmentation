# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:754
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# <!-- EDIT-V4: CODE-H5 + CODE-M8
#               补 prelude（model / transform / VOCSegmentation 的 import 与初始化）；
#               UPLOAD_FOLDER 显式声明；secure_filename 防路径穿越；扩展名白名单；
#               不再假设输入一定是 .jpg。 -->
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from datasets import VOCSegmentation
import models

app = Flask(__name__)

# ===== 配置与模型加载（与 api.py 共享同一份 transform / 设备 / 权重）=====
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB 上限，避免大文件 OOM
ALLOWED_EXT = {'.jpg', '.jpeg', '.png'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.deeplabv3plus_resnet101(num_classes=21, output_stride=16)
ckpt = torch.load('../checkpoints/deeplabv3plus_resnet101_voc.pth',
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.to(device).eval()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files.get('image')
        if not image_file:
            return render_template('index.html', error='no image')

        # secure_filename 防路径穿越（如 ../../../etc/...）
        fname = secure_filename(image_file.filename)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_EXT:
            return render_template('index.html', error='unsupported format')

        image_location = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        image_file.save(image_location)

        image = Image.open(image_location).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model(input_tensor)
        preds = output.argmax(dim=1).cpu().numpy()

        preds = VOCSegmentation.decode_target(preds[0]).astype(np.uint8)
        segmented_image = Image.fromarray(preds)
        # 不再假设输入一定是 .jpg——用 rsplit 去掉真实扩展名再统一存为 .png
        segmented_image_path = image_location.rsplit('.', 1)[0] + '_segmented.png'
        segmented_image.save(segmented_image_path)

        return render_template(
            'index.html',
            input_image='/' + image_location,
            segmented_image='/' + segmented_image_path,
        )
    return render_template('index.html')


if __name__ == '__main__':
    # 同代码 12-9：debug=False；生产建议用 gunicorn / uvicorn 替代 Werkzeug dev server。
    app.run(host='127.0.0.1', port=5000, debug=False)
