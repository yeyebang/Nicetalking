# 🎙️ Nicetalking · Vioce GUI for Bark

> 本仓库基于 [Suno AI Bark](https://github.com/suno-ai/bark) 项目二次开发。
> GUI 与产品体验署名：**awake**。Bark 模型与原始推理代码的版权与许可仍归原作者所有（MIT）。

Nicetalking 在 Suno Bark 之上提供了一个开箱即用的 Gradio Web 界面：

- 双语 UI（中文 / English）一键切换
- 100+ 内置 speaker preset，按语言/版本分组
- 支持自定义 `.npz` voice prompt 上传
- 长文本分句生成，自动以前一句的生成结果作为下一句的 history prompt，保持音色连续
- 实时进度跟踪、生成日志、历史记录回放
- 在 UI 中切换 `SUNO_USE_SMALL_MODELS` / `SUNO_OFFLOAD_CPU` / Apple MPS

---

## 🚀 快速开始

### 1. 安装

需要 Python ≥ 3.9。

```bash
git clone https://github.com/yeyebang/Nicetalking.git
cd Nicetalking
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-gui.txt
```

> 首次运行会从 Hugging Face 下载 Bark 模型权重（约 5–12 GB，具体取决于是否启用 small models）。

### 2. 启动 GUI

```bash
./launch.sh                       # 默认 http://0.0.0.0:7860
./launch.sh --port 8000           # 指定端口
./launch.sh --share               # 生成公开 Gradio share 链接
```

或者直接：

```bash
python app.py --port 7860
```

### 3. 命令行（原 Bark CLI 保留）

```bash
python -m bark --text "Hello, my name is Suno." --output_filename "example.wav"
```

---

## 🖥️ 硬件建议

| 场景 | 显存 | 设置 |
|---|---|---|
| 完整模型 GPU | ≈ 12 GB | 默认 |
| 中端 GPU | ≈ 8 GB | UI 勾选 **Use small models** |
| 低显存 / CPU | < 4 GB | 同时勾选 **Use small models** + **CPU offload** |
| Apple Silicon | — | 勾选 **Enable MPS**（实验性） |

模型设置改变后，需要点 **Load / Reload models** 才会生效。

---

## 🎤 Voice Presets

下拉框已经把 [Bark 官方 voice library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) 中所有语言的 v1 / v2 preset 都列了出来。

若想用自己的音色，把生成时勾选 **Save voice prompt** 得到的 `.npz` 文件再次上传到 **Custom Voice (.npz)**，即可复用同一音色。

---

## 📜 关于 Bark 本体

模型能力、支持的语言、非语言标记（`[laughs]` / `♪` / `...` 等）以及完整 Python API 用法请参考原项目：

👉 <https://github.com/suno-ai/bark>

本仓库未修改 Bark 模型推理代码，只新增了 [app.py](app.py)、[launch.sh](launch.sh)、[requirements-gui.txt](requirements-gui.txt) 以及本 README。原始 Bark 文档可在 git 历史中查看。

---

## ⚠️ Disclaimer

Bark 是研究用的生成式 text-to-audio 模型，输出可能与提示文本存在偏离。请合规、合理使用，不要用于伪造他人声音或任何违法用途。Suno 及本 GUI 二次开发者均不对生成内容承担责任。

## 📄 License

MIT — 与上游 Bark 保持一致。完整文本见 [LICENSE](LICENSE)。
