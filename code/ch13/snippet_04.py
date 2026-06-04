# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 13 章 — (无标号片段)
# 原始位置：book-review/drafts/ch13-v4.md:249
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 代码 13-2 ICH 数据集定义与导入
class ICHSegmentation(data.Dataset):
    cmap = color_map()

    def __init__(self, root, image_set='train', transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform

        if image_set == 'train':
            image_dir = os.path.join(root, 'train_imgs')
            mask_dir = os.path.join(root, 'train_masks')
        else:
            image_dir = os.path.join(root, 'valid_imgs')
            mask_dir = os.path.join(root, 'valid_masks')

        self.images = sorted([os.path.join(image_dir, x)
                              for x in os.listdir(image_dir)])
        self.masks = sorted([os.path.join(mask_dir, x)
                             for x in os.listdir(mask_dir)])
        # 不仅长度相等，文件名也必须一一对应（CODE-M9）
        assert len(self.images) == len(self.masks)
        for img_path, mask_path in zip(self.images, self.masks):
            img_stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_stem = os.path.splitext(os.path.basename(mask_path))[0]
            assert img_stem == mask_stem, \
                f"Image/mask filenames mismatch: {img_path} vs {mask_path}"

    def __getitem__(self, index):
        # CODE-C3：image 与 mask 走不同的 preprocess
        img = Image.open(self.images[index]).convert('RGB')
        # 用 'L' 模式（8-bit 灰度）+ 阈值化，避免 '1' 模式的 dither 噪点
        target = Image.open(self.masks[index]).convert('L')
        target = np.array(target, dtype=np.uint8)
        target = (target > 127).astype(np.float32)            # 严格二值

        if self.transform is not None:
            img, target = self.transform(img, target)

        # image 与 mask 走不同的预处理：image 含均值方差归一化；mask 仅 reshape
        img = self.image_preprocess(img)                       # 含 ImageNet 归一化
        img = torch.from_numpy(img).float()                    # (3, H, W)
        target = torch.from_numpy(target).float().unsqueeze(0) # (1, H, W)
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def image_preprocess(img):
        """RGB image → float32 (3, H, W)，做 ImageNet mean/std 归一化。"""
        arr = np.array(img, dtype=np.float32) / 255.0          # H, W, 3
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return arr.transpose(2, 0, 1)                          # 3, H, W
