# -*- coding: utf-8 -*-
# 来源：《深度学习图像分割》第 14 章 — (无标号片段)
# 原始位置：book-review/drafts/ch14-v4.md:1121
# 本文件由 book-review/scripts/extract_code.py 自动抽取生成。
# 修改请回写源稿后重新生成，避免代码与书稿失同步。
# 定义分割应用
class AISegApp(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # compose 方法将各运算法链接起来
    def compose(self):
        # …
        # DICOM 导入运算符
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=app_input_path,
            name="dcm_loader_op")
        # 序列选择运算符
        series_selector_op = DICOMSeriesSelectorOperator(
            self, rules=Sample_Rules_Text, name="series_selector_op")
        # 体数据转换运算符
        series_to_vol_op = DICOMSeriesToVolumeOperator(
            self, name="series_to_vol_op")
        # 分割运算符
        seg_op = SegOperator(self, app_context=app_context,
                             model_path=model_path, name="seg_op")
        # STL 文件转换运算符
        stl_conversion_op = STLConversionOperator(
            self,
            output_file=app_output_path.joinpath("stl/mesh.stl"),
            keep_largest_connected_component=False,
            name="stl_op",
        )
        # …
        # DICOM 分割结果写入运算符
        dicom_seg_writer = DICOMSegmentationWriterOperator(
            self, segment_descriptions=segment_descriptions,
            output_folder=app_output_path, name="dcm_seg_writer_op",
        )

        # CODE-28：按顺序逐个链接
        # 端口连接形式：0.6.x 接受 Set[Tuple[str,str]]（如下）；
        # Holoscan-based v2.x 推荐 Dict[str,str] 形式：{"output_a": "input_b"}
        self.add_flow(study_loader_op, series_selector_op,
                      {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            series_selector_op, series_to_vol_op,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(series_to_vol_op, seg_op, {("image", "image")})
        self.add_flow(seg_op, stl_conversion_op, {("seg_image", "image")})
        self.add_flow(
            series_selector_op, dicom_seg_writer,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(seg_op, dicom_seg_writer, {("seg_image", "seg_image")})
        # …
