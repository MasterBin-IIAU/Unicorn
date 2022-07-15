# 基于python的视频目标分割测评工具箱

A Python-based video object segmentation evaluation toolbox.

## 特性

* 针对**DAVIS 2016无监督视频目标分割**任务，提供`"J(M)", "J(O)", "J(D)", "F(M)", "F(O)", "F(D)"`等指标的评估（代码借鉴自davis官方的代码，建议使用前验证下）
    - 导出对指定的模型预测结果的评估结果
    - 表格化展示不同视频上模型预测的性能

## 依赖项

- `joblib`: 多进程加速计算
- `prettytable`: 输出好看的表格

## 使用方法

### DAVIS 2016无监督视频目标分割任务

1. `python eval_unvos_method.py --help`
2. 配置相关项后执行代码

python3 eval.py --name_list_path /opt/tiger/omnitrack/data/DAVIS/ImageSets/2016/val.txt --mask_root /opt/tiger/omnitrack/data/DAVIS/Annotations/480p --pred_path /opt/tiger/omnitrack/test/vos_baseline_tiny_10f_stm_DAVIS_16val --save_path /opt/tiger/omnitrack/result.pkl