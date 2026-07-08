# 微信跳一跳 Windows 桌面自动化脚本

这是一个面向 Windows 微信客户端小程序窗口的本机自动化脚本。脚本会截图识别棋子和下一块平台，根据像素距离换算鼠标长按时间。

项目只用于本机学习和实验，不包含绕过平台限制或反检测逻辑。自动模式下请保持窗口可见，并随时准备暂停。

## 1. 安装

在项目目录运行：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

如果 `opencv-python` 在较新的 Python 版本下安装失败，可以安装 Python 3.13 后改用：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 2. 准备窗口

1. 打开 Windows 微信客户端。
2. 进入“跳一跳”小程序，并保持游戏窗口可见。
3. 不要让其它窗口遮挡游戏区域。
4. 第一次运行先列出窗口，确认标题：

```powershell
.\.venv\Scripts\python .\jump_auto.py --list-windows
```

如果自动匹配不到窗口，后续命令加上：

```powershell
--window-title "窗口标题的一部分"
```

## 3. 干运行识别

先只截图和识别，不点击：

```powershell
.\.venv\Scripts\python .\jump_auto.py --dry-run
```

脚本会在 `debug` 目录保存标注图。图中会标出棋子、目标落点、距离和置信度。只有标注位置正确时，再继续校准或自动跳跃。

需要保存中间掩码时：

```powershell
.\.venv\Scripts\python .\jump_auto.py --dry-run --save-masks
```

## 4. 校准

运行：

```powershell
.\.venv\Scripts\python .\jump_auto.py --calibrate
```

流程：

1. 脚本截图并生成校准预览图。
2. 打开预览图，确认棋子和目标落点正确。
3. 在控制台输入 `y`。
4. 回到微信游戏窗口，手动完成一次左键长按跳跃。
5. 脚本记录本次长按时间，计算 `press_ms_per_px`。
6. 如果跳跃成功，在控制台输入 `y` 保存配置。

配置会保存到本地 `jump_config.json`。

## 5. 单步和自动运行

先做一次自动跳跃验证：

```powershell
.\.venv\Scripts\python .\jump_auto.py --single-step
```

确认稳定后再运行连续模式：

```powershell
.\.venv\Scripts\python .\jump_auto.py --auto
```

热键：

- `F8`：暂停或继续。
- `Esc`：退出。

识别置信度低于阈值时，脚本会暂停并保存调试图，不会继续点击。检测到结算页、弹窗或大块遮罩覆盖棋盘时，也会停止本次识别。

## 6. 识别逻辑

当前识别流程：

- 用 HSV 阈值识别棋子底部落点。
- 用背景色差优先识别目标平台，边缘检测作为兜底。
- 候选平台会先估计顶部可落脚面，再用顶部面的中心作为目标落点，避免被侧面、阴影或装饰块拉偏。
- 候选评分会考虑面积、距离、相对高度、整体宽高比和顶部面宽高比。
- 大块深色遮罩会被判定为结算页或弹层，防止把 UI 面板误识别为平台。

## 7. 配置说明

常用字段：

- `window_title`：固定匹配的窗口标题片段。
- `press_ms_per_px`：按压时长系数，校准后自动写入。
- `min_press_ms` / `max_press_ms`：单次长按时长边界。
- `confidence_threshold`：低于该置信度时自动暂停。
- `crop`：截图裁剪比例，用于排除顶部计分区或底部无关区域。
- `piece.hsv_lower` / `piece.hsv_upper`：棋子颜色阈值。
- `target.diff_threshold`：目标平台与背景的颜色差阈值。
- `target.top_surface_*`：顶部落脚面估计参数。
- `target.max_surface_aspect_ratio`：顶部面最大宽高比，用于过滤长条 UI 或异常候选。
- `overlay.*`：结算页、弹窗等大块遮罩检测参数。

调参建议：

1. 先看 `debug` 图里的棋子位置是否正确。
2. 棋子错误时调整 `piece.hsv_lower` 和 `piece.hsv_upper`。
3. 平台漏识别时优先微调 `target.diff_threshold`。
4. 平台被阴影或侧面拉偏时调整 `target.top_surface_color_tolerance` 和 `target.top_surface_max_height_ratio`。
5. 长条 UI 被误识别时降低 `target.max_surface_aspect_ratio`。
6. 方向正确但总是偏短或偏长时重新运行 `--calibrate`。

## 8. 安全注意

- 保持鼠标在屏幕角落可触发 PyAutoGUI failsafe。
- 自动模式中不要操作其它窗口。
- 微信窗口最小化、被遮挡、尺寸异常或识别失败时不要强行运行。
- 本地 `debug` 截图和 `jump_config.json` 可能包含窗口状态或个人校准参数，不建议提交到仓库。
