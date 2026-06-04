# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — (无标号片段)
# 原始位置：book-review/drafts/ch14-v4.md:1072
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 导入相关模块
from monai.deploy.core import (AppContext, ConditionType, Fragment,
                               Operator, OperatorSpec)
from monai.deploy.operators.monai_seg_inference_operator import (
    InMemImageReader, MonaiSegInferenceOperator,
)

# 定义分割运算符
class SegOperator(Operator):
    # CODE-8：__init__ 必须把 fragment / app_context / model_path 等字段保存到 self
    # 否则 compute() 中读取这些属性会立即 AttributeError
    def __init__(self, fragment, app_context, model_path,
                 *args, **kwargs):
        self.fragment = fragment
        self.app_context = app_context
        self.model_path = model_path
        # 调用父类 __init__（Holoscan-based v2.x 必需；0.6.x 兼容）
        super().__init__(fragment, *args, **kwargs)

    def compute(self, op_input, op_output, context):
        # 读取输入影像（pre_transforms / post_transforms 应在本节其他位置定义）
        # …
        # 推理运算符
        infer_operator = MonaiSegInferenceOperator(
            self.fragment,                    # 已修正拼写：fragement → fragment
            roi_size=(96, 96, 96),
            pre_transforms=pre_transforms,
            post_transforms=post_transforms,
            overlap=0.5,
            app_context=self.app_context,
            model_path=self.model_path,
        )

    # 前处理
    def pre_process(self, img_reader, out_dir):
        pass

    # 后处理
    def post_process(self, out_dir):
        pass
