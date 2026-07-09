# 跳一跳自动化脚本优化计划

## 一、概述

**目标**: 系统性提升视觉识别准确率和按压模型精度，解决"落点不准"的核心痛点。

**当前状态分析**:

项目是一个 Windows 微信跳一跳自动脚本，核心流程为：截图 → 识别棋子与目标平台 → 计算距离 → 换算按压时间 → 模拟鼠标长按。包含自动校准、在线自学习、分段校正、失败学习等高级特性。

识别不准和落点不准的根因分析：

### 视觉识别侧
- 目标平台顶面中心估计（`estimate_top_surface`）是决定跳跃目标点的最后一环，其颜色容差策略（3档固定倍率 1.0/1.25/1.50）和中心 Y 比例（固定 0.50）不够灵活
- 背景色差掩码（`build_background_diff_mask`）依赖画面左右边缘做背景采样，当目标平台延伸到边缘或背景有渐变时，边缘样本被污染导致掩码噪声大
- 目标候选评分权重（面积 38%、距离 26%、表面 22%、垂直 14%）为固定值，未根据场景自适应
- 当前平台排除逻辑（`looks_like_current_platform_target`）可能错误排除有效目标
- 5 种识别策略的覆盖率可能有盲区

### 按压模型侧
- `fit_press_model` 对所有样本一视同仁，没有离群点检测，单条错误样本可显著拉偏斜率
- 分段校正（`segment_correction`）使用固定 7px 窗口和固定学习率 0.55，数据密集段和数据稀疏段用同样参数
- 短跳 `short_hop_press_cap_ms` 只用单一锚点，短距离跳跃的物理模型不够精细
- 中心学习 `center_adjusted_press_ms` 的 deadzone（14px）和投影比阈值（0.45）固定
- 失败 cap 的 penalty 系数固定，可能过度或不足

---

## 二、优化方案

### 模块 1: 提高目标平台顶面中心估计精度 [影响: 高]

**涉及文件**: `jumpjump/vision.py`

#### 1.1 优化 `estimate_top_surface` 颜色容差搜索策略
- **现状**: 3 档固定倍率 (1.0, 1.25, 1.50)，过早退出（找到 >= min_surface_area 就 break）
- **优化**: 始终评估全部 3 档，选择表面质量分最高的，而非仅凭面积
- **位置**: [vision.py#L613-L632](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/vision.py#L613-L632)

```python
# 改动前：找到足够面积就 break
if area >= min_surface_area:
    break

# 改动后：始终评估全部 3 档，选 quality 最高的
# 移除 break，改为收集所有结果后按 (area * quality) 选最优
```

#### 1.2 自适应中心 Y 比例
- **现状**: 固定 `top_surface_center_y_ratio=0.50`（颜色分割成功）或 `center_y_ratio=0.40`（几何回退）
- **优化**: 根据平台宽高比自适应调整。宽平台（如圆盘）中心应偏上（0.45），窄高平台中心可偏下（0.55）
- **位置**: [vision.py#L642-L643](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/vision.py#L642-L643)、[vision.py#L549-L550](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/vision.py#L549-L550)

```python
# 根据 surface 宽高比计算自适应 center_y_ratio
aspect = surface_width / max(1.0, surface_height)
if aspect > 1.6:      # 宽平台
    ratio = 0.44
elif aspect > 1.0:    # 接近正方形
    ratio = 0.48
else:                  # 窄高平台
    ratio = 0.53
```

#### 1.3 LAB 色差阈值微调
- **现状**: `top_surface_color_tolerance` 默认 22（在 DEFAULT_CONFIG 中，用户配置可能是 34）
- **优化**: 在用户 `jump_config.json` 的 target 配置中写入优化后的 `top_surface_color_tolerance: 24`，与当前用户实际配置 34 取中间优化值

---

### 模块 2: 优化目标候选评分权重 [影响: 中高]

**涉及文件**: `jumpjump/vision.py`

#### 2.1 重新平衡评分权重
- **现状**: 面积 0.38、距离 0.26、表面 0.22、垂直 0.14，乘以 shape_score
- **问题**: 面积权重过高，可能选中大但远离的目标；表面权重应该更高（表面质量好的目标中心更准确）
- **优化**: 调整为面积 0.30、距离 0.28、表面 0.26、垂直 0.16
- **位置**: [vision.py#L715-L720](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/vision.py#L715-L720)

#### 2.2 confidence 计算优化
- **现状**: `confidence = clamp(score * confidence_scale * (0.85 + 0.15 * surface_quality), 0.0, 1.0)`
- **问题**: surface_quality 的影响力太小（仅 15%），但 surface_quality 高意味着中心点更准
- **优化**: 增大 surface_quality 对 confidence 的贡献：`(0.75 + 0.25 * surface_quality)`
- **位置**: [vision.py#L721](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/vision.py#L721)

---

### 模块 3: 改进背景色差掩码鲁棒性 [影响: 中]

**涉及文件**: `jumpjump/vision.py`

#### 3.1 边缘样本去噪
- **现状**: `build_background_diff_mask` 直接取左右边缘所有像素的中位数作为背景色
- **问题**: 当目标平台延伸到边缘时，边缘样本被污染
- **优化**: 对边缘样本区域做简单的离群值过滤：计算边缘区域的标准差，剔除偏离超过 2σ 的像素后再取中位数
- **位置**: [vision.py#L314-L327](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/vision.py#L314-L327)

#### 3.2 diff_threshold 略微降低默认值
- **现状**: 默认 `diff_threshold=16`，各种策略用 12/16/20
- **优化**: 默认值降低到 14，提高对低对比度平台的敏感度

---

### 模块 4: 按压模型离群点检测 [影响: 高]

**涉及文件**: `jumpjump/press_model.py`

#### 4.1 样本 RANSAC 风格过滤
- **现状**: `fit_press_model` 直接使用全部样本做最小二乘拟合
- **问题**: 一条错误的校准样本（识别错或手动按错）就会显著拉偏模型
- **优化**: 在 `fit_press_model` 中增加离群点检测：先用全部样本拟合，计算残差，剔除残差超过 2.5 倍 RMSE 的样本，再重新拟合
- **位置**: [press_model.py#L418-L487](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/press_model.py#L418-L487)

```python
# 伪代码
def fit_press_model(config):
    samples = get_valid_samples()
    # 第一轮：全部样本拟合
    best = find_best_fit(samples)
    # 第二轮：剔除离群点后重拟合
    distances = compute_distances(samples, best.y_weight)
    residuals = abs(press_ms - (slope * dist + offset))
    threshold = 2.5 * best.rmse
    clean_samples = [s for s, r in zip(samples, residuals) if r <= threshold]
    if len(clean_samples) >= max(3, len(samples) * 0.7):
        best = find_best_fit(clean_samples)
    # 保存 fit_rmse_ms 和 outlier_ratio
```

#### 4.2 加权拟合（按样本置信度和时效性）
- **现状**: 所有样本等权重
- **优化**: 根据样本的 `confidence`（越高权重越大）和样本时间戳新旧程度做加权拟合。更新样本权重 1.0，较旧样本按指数衰减（保留最近 40 条中越新的权重越高）
- **位置**: [press_model.py#L435-L470](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/press_model.py#L435-L470)

---

### 模块 5: 优化分段校正参数 [影响: 中高]

**涉及文件**: `jumpjump/press_model.py`, `jumpjump/config.py`

#### 5.1 自适应分段大小
- **现状**: 固定 `segment_size_px=7`
- **问题**: 7px 的段在 500px 距离范围内产生约 71 段，数据稀疏段几乎没数据
- **优化**: 当距离范围大（如 >300px）时自动扩大到 10-12px，缩小段数，每个段有更多样本支撑
- **位置**: [press_model.py#L141-L150](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/press_model.py#L141-L150)

#### 5.2 优化学习率和衰减参数
- **现状**: `segment_correction_learning_rate=0.55`、`segment_correction_success_decay=0.0`
- **优化**: 学习率降至 0.45（更保守的 EMA 更新），success_decay 设为 0.08（成功后轻微衰减相邻段）
- **位置**: [jumpjump/config.py#L130-L131](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/config.py#L130-L131)

#### 5.3 优化中心学习 deadzone
- **现状**: `center_deadzone_px=10`（DEFAULT）或 14（运行时）
- **优化**: 调整为更紧凑的 8px，让更多微小偏差触发中心学习微调
- **位置**: [jumpjump/config.py#L124](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/config.py#L124)

---

### 模块 6: 改进短跳模型 [影响: 中]

**涉及文件**: `jumpjump/press_model.py`

#### 6.1 多锚点插值
- **现状**: `short_hop_press_cap_ms` 只用最小距离锚点做外推
- **优化**: 收集距离 < min_anchor_distance 的所有样本点做 mini curve，用这些点做分段插值而非单点外推
- **位置**: [press_model.py#L490-L521](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/press_model.py#L490-L521)

#### 6.2 短跳非线性补偿
- **优化**: 短距离跳跃（< 60px）的实际按压时间与距离并非完全线性，加入轻微的非线性补偿因子（短跳多压 3-5%）
- **位置**: 在 `short_hop_press_cap_ms` 中新增逻辑

---

### 模块 7: 优化失败学习 cap [影响: 中]

**涉及文件**: `jumpjump/press_model.py`

#### 7.1 自适应 penalty 系数
- **现状**: `penalty = 1.0 + 0.12 * (distance_delta / window_px)`
- **优化**: penalty 系数与 cap 的 confidence 关联，高置信度的失败 cap 用更大的 penalty（更激进地压低压时间）
- **位置**: [press_model.py#L524-L549](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/press_model.py#L524-L549)

---

### 模块 8: 优化当前平台排除逻辑 [影响: 中低]

**涉及文件**: `jumpjump/vision.py`

#### 8.1 放宽目标在上方的限制
- **现状**: `max_target_above_piece_ratio=0.045`，目标不能比棋子高超过裁剪区域高度的 4.5%
- **优化**: 放宽到 0.06，适应更陡峭的跳跃场景
- **位置**: [vision.py#L463-L465](file:///d:/MiscPythonProjects/Jump_Jump/jumpjump/vision.py#L463-L465)

---

## 三、文件改动清单

| 文件 | 改动内容 | 风险 |
|------|---------|------|
| `jumpjump/vision.py` | 模块1-3, 8: 顶面估计、评分权重、背景掩码、平台排除 | 中 |
| `jumpjump/press_model.py` | 模块4-7: 离群点检测、加权拟合、分段校正、短跳模型、失败cap | 中高 |
| `jumpjump/config.py` | 模块5: 默认参数调整 | 低 |
| `tests/test_press_model.py` | 新增测试: 离群点检测、加权拟合验证 | 低 |
| `tests/test_vision_regression.py` | 扩展回归测试覆盖 | 低 |

## 四、验证方案

1. **单元测试**: 运行 `python -m pytest tests/ -v` 确保所有现有测试通过
2. **新增测试**: 为离群点检测和加权拟合新增测试用例
3. **干运行验证**: 使用 `--dry-run` 模式对多张 debug 截图验证识别准确率
4. **手动校准回归**: 使用现有 `jump_config.json` 的校准数据，验证 `fit_press_model` 输出与优化前相比 RMSE 不升高
5. **自动化对比**: 在相同截图集上，对比优化前后的 `detect_jump` 输出（target 坐标、confidence）

## 五、实施顺序

1. **先做低风险的参数调整**（模块5, 8）- 纯配置改动，可快速验证效果
2. **再做视觉识别优化**（模块1-3）- 改善目标检测精度
3. **最后做按压模型优化**（模块4, 6, 7）- 最核心的落点精度提升，需要校准数据回归验证
