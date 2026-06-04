# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:657
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# <!-- EDIT-V4: CODE-H5 + CODE-M7 + CODE-M9
#               weights_only 注释、device 自适应、predict 加请求体异常处理、
#               debug=False + 生产部署提示。 -->
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify

from datasets import VOCSegmentation
import models

app = Flask(__name__)

# 设备自适应——避免在无 GPU 环境下硬编码 'cuda' 直接挂
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型字典
model_map = {
    'deeplabv3plus_resnet50': models.deeplabv3plus_resnet50,
    'deeplabv3plus_resnet101': models.deeplabv3plus_resnet101,
}

# 加载模型——完整 checkpoint 含 optimizer/scheduler 等非 state_dict 张量，
# 须显式 weights_only=False；如仅加载权重子项则保持默认 True 更安全（PyTorch 2.4+）。
model = model_map['deeplabv3plus_resnet101'](num_classes=21, output_stride=16)
ckpt = torch.load('../checkpoints/deeplabv3plus_resnet101_voc.pth',
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()
print('model loaded.')

# 数据变换（与训练 / 验证一致——mean/std 抽出为公共常量，避免三处复制粘贴漂移）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@app.route('/predict', methods=['POST'])
def predict():
    # 请求体异常处理——避免无 'image' 字段或非图像内容直接 500（CODE-M7）
    if 'image' not in request.files:
        return jsonify({'error': 'no image field in form-data'}), 400
    try:
        buf = request.files['image'].read()
        image = Image.open(io.BytesIO(buf)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'cannot decode image: {e}'}), 400

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(input_tensor)
    preds = output.argmax(dim=1).cpu().numpy()
    preds = VOCSegmentation.decode_target(preds[0]).astype(np.uint8)
    return jsonify(preds.tolist())


if __name__ == '__main__':
    # 仅供本地学习；生产请用 gunicorn -w 4 api:app 或 FastAPI + uvicorn 替代 dev server。
    # debug=True 会启用 Werkzeug 调试器，存在远程任意代码执行风险，禁用之。
    app.run(host='127.0.0.1', port=5000, debug=False)
