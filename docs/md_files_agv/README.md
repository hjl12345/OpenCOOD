# OpenCOOD 源码剖析文档（AGV 版）

本目录面向**需要读懂、改造或复现 OpenCOOD 的开发者与研究者**，从「它是什么」到「一行数据在张量里怎么流动」做完整拆解。所有结论均以本仓库 `main` 分支（commit `31ba160`）的实际代码为依据，标注了对应文件与行号，而非仅复述论文。

---

## 文档索引

| 编号 | 文档 | 核心问题 | 适合谁 |
|---|---|---|---|
| 01 | [项目总览](01_项目总览.md) | OpenCOOD 解决什么问题？有哪几类融合范式？有哪些模型？ | 所有人，先读 |
| 02 | [项目结构](02_项目结构.md) | 109 个 Python 文件如何组织？改哪里？ | 想动手改代码 |
| 03 | [配置系统](03_配置系统.md) | 一个 yaml 如何驱动数据集、模型、损失？参数如何派生？ | 要训练/换模型 |
| 04 | [数据流水线](04_数据流水线.md) | 磁盘上的 pcd/yaml 如何变成网络输入？ | 要复现/换数据集 |
| 05 | [模型原理](05_模型原理.md) | PointPillar + 9 种融合模块各自的机制与张量形状 | 要理解算法 |
| 06 | [端到端数据流](06_端到端数据流.md) | 一次 forward 的完整张量形状追踪（含具体数字） | 要调 shape bug |
| 07 | [扩展与注意事项](07_扩展与注意事项.md) | 加新模型/新数据集怎么改？有哪些隐式约束？ | 要二次开发 |

---

## 三句话概括

1. **是什么**：OpenCOOD 是 UCLA Mobility Lab 开源的车路协同（V2V/V2I）3D 感知框架，也是 ICRA 2022 论文 **OPV2V** 的官方实现，提供 OPV2V / V2XSet 数据集的统一 API 与 10 余种协同感知模型的参考实现。
2. **怎么组织**：以 **yaml 配置为唯一入口**，通过 `importlib` 反射加载「数据集类 / 预处理器 / 后处理器 / 模型类 / 损失类」。新增**模型或损失**只需新文件 + 新 yaml；新增**数据集/预处理器**还需在对应 `__init__.py` 的注册表（及 `build_dataset` 的白名单）中登记。
3. **核心数据流**：`场景库索引 → 多车原始点云 → 坐标统一到 ego → 体素化 → PointPillar BEV 特征 → 多车特征对齐与融合 → anchor 检测头 → NMS → AP 评估`。

---

## 关键事实速查

| 项目 | 取值 | 出处 |
|---|---|---|
| LiDAR 范围（多数 PointPillar 配置） | `[-140.8, -40, -3, 140.8, 40, 1]` m | `hypes_yaml/point_pillar_*.yaml` |
| LiDAR 范围（V2X-ViT / CoBEVT / Where2comm） | `[-140.8, -38.4, -3, 140.8, 38.4, 1]` m | 同目录三个 yaml |
| LiDAR 范围（Late Fusion） | `[-70.4, -40, -3, 70.4, 40, 1]` m | `point_pillar_late_fusion.yaml` |
| 体素尺寸 | `[0.4, 0.4, 4]` m | 同上 |
| BEV 网格（±40 m 配置） | `704 × 200 × 1` | `yaml_utils.load_point_pillar_params` 派生 |
| BEV 网格（±38.4 m 配置） | `704 × 192 × 1` | 同上 |
| BEV 网格（±70.4 m Late） | `352 × 200 × 1` | 同上 |
| 通信距离门限 | 70 m | `data_utils/datasets/__init__.py:20` |
| 数据帧率 | 10 Hz（时延量化单位 100 ms） | `basedataset.py:375` |
| 最大协同车数 | 默认 5（yaml `max_cav`），代码兜底 7 | `basedataset.py:112-116` |
| 单帧最大目标数 | 100（`max_num`，用于 padding 对齐） | yaml `postprocess.max_num` |
| 评估 IoU 阈值 | 0.30 / 0.50 / 0.70 | `tools/inference.py:82-84` |
| 支持 backbone | PointPillar / VoxelNet / SECOND / PIXOR / PV-RCNN | `opencood/models/` |

> ⚠️ **BEV 网格不是全局常量**：它由各 yaml 的 `cav_lidar_range / voxel_size` 共同决定，同一份代码在不同配置下分辨率不同。因此任何「张量形状」都不能跨配置套用。

---

## 一句话警告（最容易踩的坑）

> 检测头的输出分辨率必须与 anchor 网格严格一致：**anchor 由 `postprocess.anchor_args.feature_stride` 决定（`W//feature_stride`），而输出分辨率由 `model.args.shrink_header.stride` 决定**。两者不一致会在 `delta_to_boxes3d` 处报 shape 不匹配。详见 [07_扩展与注意事项](07_扩展与注意事项.md)。
>
> 举例：V2X-ViT 的 `704×192` BEV 经 `shrink stride=2` 得 `352×96`；`feature_stride=4` 得 anchor 网格 `176×48`，共 `176×48×2 = 16896` 个 anchor。若误把 `feature_stride` 写成 2，anchor 数变成 `352×96×2 = 67584`，就会与检测头输出对不上（差 4 倍）。

---

## 常用命令（照抄即可）

```bash
# 0) 编译必需扩展（必须在仓库根目录执行）
python opencood/utils/setup.py build_ext --inplace        # Cython: box_overlaps
python opencood/pcdet_utils/setup.py build_ext --inplace  # CUDA: 仅 FPV-RCNN / CIASSD 需要

# 1) 从零训练
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_v2vnet.yaml

# 2) 续训 / 微调（★ 会忽略 --hypes_yaml，改读 <model_dir>/config.yaml）
python opencood/tools/train.py --hypes_yaml <任意> --model_dir opencood/logs/<exp>_<时间戳>

# 3) 测试（先把 checkpoint 目录里 config.yaml 的 validate_dir 改成测试集路径）
python opencood/tools/inference.py --model_dir <checkpoint_dir> --fusion_method intermediate

# 4) 多卡（与 --half 互斥）
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  --use_env opencood/tools/train.py --hypes_yaml <yaml>
```

> `--fusion_method` 必须与模型匹配：`point_pillar` → `early` / `late`，`point_pillar_*` 融合模型 → `intermediate`。评估结果写到 `<model_dir>/eval.yaml`（开 `--global_sort_detections` 时写 `eval_global_sort.yaml`）——**两个文件名不同，别找错**。
