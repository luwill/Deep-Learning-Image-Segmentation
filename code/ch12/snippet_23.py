# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 12 章 — (无标号片段)
# 原始位置：book-review/drafts/ch12-v4.md:1004
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
import io, numpy as np, onnxruntime as ort
from fastapi import FastAPI, UploadFile
from PIL import Image

app = FastAPI()
session = ort.InferenceSession(
    'deeplabv3plus_resnet101.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
)

@app.post('/predict')
async def predict(image: UploadFile):
    img = Image.open(io.BytesIO(await image.read())).convert('RGB')
    x = preprocess(img)                              # 与训练一致的归一化
    out = session.run(None, {'input': x[None]})[0]   # numpy 推理
    pred = out.argmax(axis=1)[0].astype(np.uint8)
    return {'mask': pred.tolist()}
