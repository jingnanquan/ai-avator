---
name: video-workflow
description: AI数字人视频生成工作流 - 根据口播主题/文案调用Seedance生成视频片段，FFmpeg合并。提供命令行调用方式。
keywords: ["video", "seedance", "数字人", "workflow", "ai-video"]
version: 1.0.0
author: OpenClaw
---

# Video Workflow

AI视频生成工作流技能，根据口播主题或文案自动生成视频。

## 功能

- **主题生成文案**: 根据输入主题生成口播文案
- **视频片段生成**: 调用Seedance API生成数字人视频片段
- **智能文案拆分**: 将长文案拆分为适合视频的口播片段
- **视频合并**: 使用FFmpeg将多个片段合并为完整视频


## 使用方法

### 前置准备

1. **激活Python环境**
   ```bash
   /root/miniforge3/bin/conda activate lang
   ```

2. **配置API密钥**（可选）
   在 `scripts/video_workflow.py` 的 `Settings` 类中配置：
   - `seedance_api_key`: Seedance API 密钥
   - `qwen_api_key`: 通义千问 API 密钥

### 命令行调用

```bash
# 通过主题生成视频（推荐）
python3 /root/.openclaw/workspace/skills/video-workflow/scripts/video_workflow.py \
  -i "/path/to/reference.png" \
  -t "介绍武汉大学"

# 通过已有文案生成视频
python3 /root/.openclaw/workspace/skills/video-workflow/scripts/video_workflow.py \
  -i "/path/to/reference.png" \
  -s "数字人技术正深度变革我们的数字生活！"
```

### 参数说明

| 参数 | 短参数 | 说明 | 必填 |
|------|--------|------|------|
| --img | -i | 参考图片路径 | 是 |
| --topic | -t | 口播主题（二选一） | 否 |
| --script | -s | 口播文案（二选一） | 否 |

### 参数互斥规则

- `-i` (参考图片) 为必填参数
- `-t` 和 `-s` 只能二选一，不能同时提供

## 输出

工作流执行完成后，在当前目录创建 `workflow_run_YYYYMMDD_HHMMSS/` 文件夹，包含：
- `reference_xxx.png` - 参考图片备份
- `videos/` - 视频片段目录
- `segment_XXX_*.mp4` - 各个视频片段
- `final_*.mp4` - 最终合并视频
