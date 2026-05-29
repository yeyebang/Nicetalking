# 🎙️ Nicetalking · Vioce GUI for Bark

**🌐 Language:** **English** · [中文](README.md)

> A secondary development of [Suno AI Bark](https://github.com/suno-ai/bark).
> GUI and product experience by **awake**. The Bark model and original inference code remain copyright of the original authors (MIT).

Nicetalking ships a ready-to-use Gradio web interface on top of Suno Bark:

- Bilingual UI (English / 中文) toggle
- 100+ built-in speaker presets, grouped by language and version
- Upload your own `.npz` voice prompt
- Long-form generation that splits text into sentences and chains each sentence's output as the history prompt for the next, keeping the voice consistent
- Live progress tracking, generation log, and history playback
- Toggle `SUNO_USE_SMALL_MODELS` / `SUNO_OFFLOAD_CPU` / Apple MPS from the UI

---

## 🚀 Quick start

### 1. Install

Requires Python ≥ 3.9.

```bash
git clone https://github.com/yeyebang/Nicetalking.git
cd Nicetalking
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-gui.txt
```

> On first run, Bark model weights (~5–12 GB depending on whether small models are enabled) are downloaded from Hugging Face.

### 2. Launch the GUI

```bash
./launch.sh                       # defaults to http://0.0.0.0:7860
./launch.sh --port 8000           # custom port
./launch.sh --share               # generate a public Gradio share link
```

Or directly:

```bash
python app.py --port 7860
```

### 3. CLI (original Bark CLI preserved)

```bash
python -m bark --text "Hello, my name is Suno." --output_filename "example.wav"
```

---

## 🖥️ Hardware notes

| Scenario | VRAM | Settings |
|---|---|---|
| Full models on GPU | ≈ 12 GB | default |
| Mid-tier GPU | ≈ 8 GB | check **Use small models** |
| Low VRAM / CPU | < 4 GB | check **Use small models** + **CPU offload** |
| Apple Silicon | — | check **Enable MPS** (experimental) |

After changing model settings, click **Load / Reload models** for them to take effect.

---

## 🎤 Voice presets

The dropdown already includes every v1 / v2 preset across all languages from the [official Bark voice library](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c).

To reuse a voice you generated, tick **Save voice prompt** during generation, then re-upload the resulting `.npz` to **Custom Voice (.npz)**.

---

## 📜 About Bark itself

For model capabilities, supported languages, non-verbal tags (`[laughs]` / `♪` / `...` etc.) and the full Python API, refer to the upstream project:

👉 <https://github.com/suno-ai/bark>

This repo does not modify Bark's inference code. It only adds [app.py](app.py), [launch.sh](launch.sh), [requirements-gui.txt](requirements-gui.txt), and this README. The original Bark documentation is available in the git history.

---

## ⚠️ Disclaimer

Bark is a research-oriented generative text-to-audio model whose output can diverge from the input text in unexpected ways. Use it responsibly — do not impersonate real people or use it for any unlawful purpose. Neither Suno nor the GUI author take responsibility for generated content.

## 📄 License

MIT — same as upstream Bark. See [LICENSE](LICENSE).
