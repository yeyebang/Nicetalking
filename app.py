"""
Vioce - Gradio-based web interface for the Bark text-to-audio model.
"""

import os
import re
import sys
import threading
import logging
from datetime import datetime
from typing import Optional

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import gradio as gr
from scipy.io.wavfile import write as write_wav

# Add parent dir to path so we can import bark
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bark.generation import (
    SAMPLE_RATE,
    SUPPORTED_LANGS,
    generate_text_semantic,
    generate_coarse,
    generate_fine,
    codec_decode,
    preload_models,
)
from bark.api import save_as_prompt

logger = logging.getLogger(__name__)

APP_NAME = "Vioce"
PRODUCT_TITLE = "Nice talking"
DEFAULT_SPEAKER = "(None - Default voice)"
HERO_CREDIT = "基于 Suno AI Bark 项目二次开发。Vioce GUI 二次开发署名：awake。 suno-ai/bark"

UI_TEXT = {
    "en": {
        "hero_title": PRODUCT_TITLE,
        "hero_subtitle": (
            "A premium local voice studio powered by Bark. Generate speech, music, "
            "and sound effects with a precise, private workflow."
        ),
        "hero_badge": "LOCAL AI AUDIO STUDIO",
        "hero_cta": HERO_CREDIT,
        "panel_input": "INPUT TERMINAL",
        "panel_output": "OUTPUT SIGNAL",
        "panel_log": "SIGNAL LOG",
        "panel_history": "GENERATION ARCHIVE",
        "panel_model": "MODEL CORE",
        "metric_engine": "Bark Engine",
        "metric_sample": "24kHz Signal",
        "metric_runtime": "Local Runtime",
        "credit": "Based on Suno AI Bark. Vioce GUI secondary development by awake.",
        "language_label": "Language",
        "language_info": "Switch the interface language.",
        "text_label": "Input Text",
        "text_placeholder": "Enter text to generate audio... (supports multiple languages)",
        "tip": "*Tip: Nice talking supports multiple languages, sound effects like [laughter], [music], and more.*",
        "speaker_label": "Speaker Preset",
        "speaker_none": "No speaker selected. The model will use a default voice.",
        "custom_voice_label": "Custom Voice (.npz)",
        "text_temp_label": "Text Temperature",
        "text_temp_info": "Controls text generation diversity. Lower = more conservative, higher = more creative.",
        "text_temp_reset": "Reset text",
        "waveform_temp_label": "Waveform Temperature",
        "waveform_temp_info": "Controls audio generation diversity. Lower = more stable, higher = more varied.",
        "waveform_temp_reset": "Reset waveform",
        "long_form_label": "Long-form Generation",
        "long_form_info": "Split text into sentences and chain generations for longer audio (>13s).",
        "output_filename_label": "Output Filename",
        "output_filename_placeholder": "auto-generated if empty",
        "save_prompt_label": "Save as voice preset (.npz)",
        "save_prompt_info": "Save the full generation for reuse as a voice preset.",
        "generate_btn": "Generate Audio",
        "audio_label": "Generated Audio",
        "log_label": "Progress Log",
        "history_hint": "Click a row to replay the audio.",
        "history_headers": ["Time", "Text", "Speaker", "Duration"],
        "model_notice": (
            "### Model download notice\n"
            "On first use, Bark may download model files from Hugging Face. This can take several minutes, "
            "and the terminal may show countdown/retry messages while the download is in progress.\n\n"
            "Model checkpoints are loaded from `suno/bark`, the original Suno AI Bark model repository."
        ),
        "model_settings_hint": "Configure model loading options. Changes require clicking 'Load Models' to take effect.",
        "use_small_label": "Use Small Models",
        "use_small_info": "Use smaller models (~8GB VRAM instead of ~12GB).",
        "cpu_offload_label": "CPU Offload",
        "cpu_offload_info": "Offload models to CPU between stages (for <4GB VRAM GPUs).",
        "mps_label": "Enable MPS",
        "mps_info": "Enable Apple MPS backend (experimental, for Apple Silicon).",
        "load_models_btn": "Load / Reload Models",
        "model_status_label": "Model Status",
        "model_status_placeholder": "Click 'Load Models' to preload models...",
        "empty_text": "Please enter some text to generate.",
        "loading_models": "Loading models...",
        "model_download_log": "If this is your first run, Bark may be downloading model files now. Please keep this window open.",
        "done": "Done!",
        "models_prepare": "Preparing model load. First use may download large files, so this can take a while...",
        "models_loaded": "Models loaded successfully.",
        "models_failed": "Model loading failed",
        "generation_failed": "Generation failed",
        "audio_saved": "Audio saved",
        "prompt_saved": "Voice preset saved",
    },
    "zh": {
        "hero_title": PRODUCT_TITLE,
        "hero_subtitle": "基于 Bark 的高级本地语音工作台，用清晰、私密、可控的流程生成语音、音乐和音效。",
        "hero_badge": "本地 AI 音频工作室",
        "hero_cta": HERO_CREDIT,
        "panel_input": "输入终端",
        "panel_output": "输出信号",
        "panel_log": "信号日志",
        "panel_history": "生成档案",
        "panel_model": "模型核心",
        "metric_engine": "Bark 引擎",
        "metric_sample": "24kHz 信号",
        "metric_runtime": "本地运行",
        "credit": "基于 Suno AI Bark 项目二次开发。Vioce GUI 二次开发署名：awake。",
        "language_label": "界面语言",
        "language_info": "切换界面显示语言。",
        "text_label": "输入文本",
        "text_placeholder": "输入要生成音频的文本，支持多语言",
        "tip": "*提示：Nice talking支持多语言，也支持 [laughter]、[music] 这类音效提示词。*",
        "speaker_label": "音色预设",
        "speaker_none": "未选择音色，将使用默认声音。",
        "custom_voice_label": "自定义音色（.npz）",
        "text_temp_label": "文本温度",
        "text_temp_info": "控制文本生成的发散程度。越低越稳定，越高越有变化。",
        "text_temp_reset": "重置文本",
        "waveform_temp_label": "波形温度",
        "waveform_temp_info": "控制音频生成的发散程度。越低越稳定，越高越多样。",
        "waveform_temp_reset": "重置波形",
        "long_form_label": "长文本生成",
        "long_form_info": "按句子切分并串联生成，适合超过约 13 秒的长音频。",
        "output_filename_label": "输出文件名",
        "output_filename_placeholder": "留空则自动生成",
        "save_prompt_label": "保存为音色预设（.npz）",
        "save_prompt_info": "保存本次完整生成结果，后续可作为音色预设复用。",
        "generate_btn": "生成音频",
        "audio_label": "生成结果",
        "log_label": "进度日志",
        "history_hint": "点击表格行可回放对应音频。",
        "history_headers": ["时间", "文本", "音色", "时长"],
        "model_notice": (
            "### 模型下载提示\n"
            "首次使用时，Nice talking会从 Hugging Face 下载模型文件。这个过程可能需要几分钟，"
            "终端里出现倒计时或重试信息通常代表模型正在下载，请不要关闭窗口。\n\n"
            "模型权重来自 `suno/bark`，也就是 Suno AI Bark 原项目模型仓库。"
        ),
        "model_settings_hint": "配置模型加载选项。修改后需要点击“加载 / 重新加载模型”才会生效。",
        "use_small_label": "使用小模型",
        "use_small_info": "使用较小模型，显存需求约 8GB，而不是约 12GB。",
        "cpu_offload_label": "CPU 卸载",
        "cpu_offload_info": "在不同阶段之间把模型卸载到 CPU，适合低显存设备。",
        "mps_label": "启用 MPS",
        "mps_info": "启用 Apple Silicon 的 MPS 后端，仍属实验选项。",
        "load_models_btn": "加载 / 重新加载模型",
        "model_status_label": "模型状态",
        "model_status_placeholder": "点击“加载 / 重新加载模型”预加载模型",
        "empty_text": "请输入要生成的文本。",
        "loading_models": "正在加载模型...",
        "model_download_log": "如果这是首次运行，Bark 现在可能正在下载模型文件，请保持窗口打开。",
        "done": "完成！",
        "models_prepare": "正在准备加载模型。首次使用可能需要下载较大的模型文件，请耐心等待...",
        "models_loaded": "模型加载成功。",
        "models_failed": "模型加载失败",
        "generation_failed": "生成失败",
        "audio_saved": "音频已保存",
        "prompt_saved": "音色预设已保存",
    },
}


def t(lang: str, key: str):
    """Return localized UI text."""
    return UI_TEXT.get(lang, UI_TEXT["en"])[key]


def render_hero_copy(lang: str) -> str:
    """Render the product header copy."""
    return f"""
<div class="defi-hero-copy">
  <div class="defi-badge"><span></span>{t(lang, 'hero_badge')}</div>
  <h1 class="cyber-glitch" data-text="{t(lang, 'hero_title')}">{t(lang, 'hero_title')}</h1>
  <p class="cyber-type">{t(lang, 'hero_subtitle')}</p>
  <div class="defi-hero-note">{t(lang, 'hero_cta')}</div>
</div>
"""


def render_hero_orb() -> str:
    """Render the product header audio waveform orb."""
    return """
  <div class="defi-orb" aria-hidden="true">
    <div class="defi-orb__ring defi-orb__ring--outer"></div>
    <div class="defi-orb__ring defi-orb__ring--inner"></div>
    <div class="defi-orb__core">
      <span></span><span></span><span></span><span></span><span></span>
    </div>
    <div class="defi-orb__stat defi-orb__stat--top">24kHz</div>
    <div class="defi-orb__stat defi-orb__stat--bottom">LOCAL</div>
  </div>
"""


def render_status_strip(lang: str) -> str:
    """Render compact product metrics."""
    return f"""
<div class="cyber-status-strip">
  <div><strong>{t(lang, 'metric_engine')}</strong><span>GEN-CORE</span></div>
  <div><strong>{t(lang, 'metric_sample')}</strong><span>SAMPLE RATE</span></div>
  <div><strong>{t(lang, 'metric_runtime')}</strong><span>PRIVATE NODE</span></div>
</div>
"""


def render_credit(lang: str) -> str:
    """Render attribution text for the upstream project and secondary developer."""
    return f"""
<div class="cyber-credit">
  <span>{t(lang, 'credit')}</span>
  <a href="https://github.com/suno-ai/bark" target="_blank" rel="noopener noreferrer">suno-ai/bark</a>
</div>
"""


CYBER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Orbitron:wght@700;800;900&family=Share+Tech+Mono&display=swap');

:root {
    --cyber-bg: #0a0a0f;
    --cyber-fg: #e0e0e0;
    --cyber-card: #12121a;
    --cyber-muted: #1c1c2e;
    --cyber-muted-fg: #8b91a4;
    --cyber-accent: #00ff88;
    --cyber-secondary: #ff00ff;
    --cyber-tertiary: #00d4ff;
    --cyber-border: #2a2a3a;
    --cyber-input: #111119;
    --cyber-danger: #ff3366;
    --shadow-neon: 0 0 5px #00ff88, 0 0 14px #00ff8845;
    --shadow-neon-lg: 0 0 10px #00ff88, 0 0 26px #00ff8860, 0 0 50px #00ff8828;
    --shadow-magenta: 0 0 5px #ff00ff, 0 0 22px #ff00ff55;
    --shadow-cyan: 0 0 5px #00d4ff, 0 0 22px #00d4ff55;
    --chamfer: polygon(0 14px, 14px 0, calc(100% - 14px) 0, 100% 14px, 100% calc(100% - 14px), calc(100% - 14px) 100%, 14px 100%, 0 calc(100% - 14px));
    --chamfer-sm: polygon(0 8px, 8px 0, calc(100% - 8px) 0, 100% 8px, 100% calc(100% - 8px), calc(100% - 8px) 100%, 8px 100%, 0 calc(100% - 8px));
}

html,
body,
gradio-app,
.gradio-container {
    width: 100% !important;
    min-width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    overflow-x: hidden !important;
    background:
        radial-gradient(circle at 12% 8%, rgba(255, 0, 255, 0.14), transparent 28rem),
        radial-gradient(circle at 88% 18%, rgba(0, 212, 255, 0.13), transparent 28rem),
        radial-gradient(circle at 45% 100%, rgba(0, 255, 136, 0.09), transparent 30rem),
        var(--cyber-bg) !important;
    color: var(--cyber-fg) !important;
    font-family: "JetBrains Mono", "Fira Code", Consolas, monospace !important;
}

gradio-app,
gradio-app > div,
.gradio-container,
.gradio-container > .main,
.gradio-container .main,
.gradio-container .contain,
.gradio-container .app,
.gradio-container .wrap {
    max-width: none !important;
}

body::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 9999;
    pointer-events: none;
    background:
        repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0, 0, 0, 0.34) 2px, rgba(0, 0, 0, 0.34) 4px),
        linear-gradient(rgba(0, 255, 136, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 212, 255, 0.035) 1px, transparent 1px);
    background-size: auto, 54px 54px, 54px 54px;
    mix-blend-mode: screen;
    opacity: 0.38;
}

body::after {
    content: "";
    position: fixed;
    left: 0;
    right: 0;
    top: -20vh;
    height: 20vh;
    z-index: 9998;
    pointer-events: none;
    background: linear-gradient(180deg, transparent, rgba(0, 255, 136, 0.15), transparent);
    animation: scanline 6s linear infinite;
    opacity: 0.45;
}

.gradio-container {
    width: 100vw !important;
    max-width: none !important;
    min-height: 100vh !important;
    padding: clamp(18px, 1.45vw, 34px) clamp(18px, 1.35vw, 34px) !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    --body-text-color: #d9deea !important;
    --block-label-text-color: #dfffee !important;
    --block-title-text-color: #dfffee !important;
    --input-text-color: #e0e0e0 !important;
    --button-secondary-text-color: #dfffee !important;
    --checkbox-label-text-color: #d9deea !important;
    --neutral-700: #d9deea !important;
    --neutral-800: #e0e0e0 !important;
}

.main, .contain, .app {
    background: transparent !important;
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
}

* {
    border-radius: 0 !important;
    letter-spacing: 0.01em;
}

.cyber-hero,
.cyber-panel,
.cyber-terminal,
.cyber-hud,
.cyber-status-strip {
    position: relative;
    clip-path: var(--chamfer);
}

.cyber-hero {
    overflow: hidden;
    padding: 34px;
    margin-bottom: 18px;
    border: 1px solid rgba(0, 255, 136, 0.42);
    background:
        linear-gradient(135deg, rgba(18, 18, 26, 0.96), rgba(10, 10, 15, 0.88)),
        linear-gradient(90deg, rgba(0, 255, 136, 0.08) 1px, transparent 1px),
        linear-gradient(rgba(0, 212, 255, 0.08) 1px, transparent 1px);
    background-size: auto, 38px 38px, 38px 38px;
    box-shadow: var(--shadow-neon-lg);
}

.cyber-hero::before,
.cyber-panel::before,
.cyber-terminal::before,
.cyber-hud::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(120deg, rgba(0, 255, 136, 0.15), transparent 26%, rgba(255, 0, 255, 0.1) 72%, transparent);
    opacity: 0.58;
}

.cyber-hero__signal {
    color: var(--cyber-accent);
    font-family: "Share Tech Mono", monospace;
    font-size: 12px;
    letter-spacing: 0.26em;
    margin-bottom: 16px;
    text-transform: uppercase;
    text-shadow: 0 0 10px rgba(0, 255, 136, 0.55);
}

.cyber-hero__content {
    position: relative;
    z-index: 1;
    max-width: min(1280px, 82vw);
}

.cyber-glitch {
    position: relative;
    width: fit-content;
    margin: 0 0 16px;
    color: var(--cyber-fg);
    font-family: "Orbitron", "Share Tech Mono", monospace;
    font-size: clamp(54px, 8vw, 124px);
    font-weight: 900;
    line-height: 0.86;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    text-shadow: -2px 0 var(--cyber-secondary), 2px 0 var(--cyber-tertiary), 0 0 26px rgba(0, 255, 136, 0.42);
    animation: rgbShift 3.4s steps(2) infinite;
}

.cyber-glitch::before,
.cyber-glitch::after {
    content: attr(data-text);
    position: absolute;
    inset: 0;
    opacity: 0.65;
    pointer-events: none;
}

.cyber-glitch::before {
    color: var(--cyber-secondary);
    transform: translate(-2px, 1px);
    clip-path: inset(0 0 56% 0);
    animation: glitch 2.8s steps(3) infinite;
}

.cyber-glitch::after {
    color: var(--cyber-tertiary);
    transform: translate(2px, -1px);
    clip-path: inset(54% 0 0 0);
    animation: glitch 3.1s steps(4) infinite reverse;
}

.cyber-type {
    max-width: min(960px, 62vw);
    margin: 0;
    color: #c8ccd8;
    font-size: clamp(15px, 1.3vw, 20px);
    line-height: 1.75;
    text-shadow: 0 0 12px rgba(0, 212, 255, 0.24);
}

.cyber-cursor {
    display: inline-block;
    width: 10px;
    height: 1.05em;
    margin-left: 8px;
    vertical-align: -0.18em;
    background: var(--cyber-accent);
    box-shadow: var(--shadow-neon);
    animation: blink 1s step-end infinite;
}

.cyber-status-strip span,
label,
.wrap .label-wrap span {
    font-family: "Share Tech Mono", monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.16em;
}

.cyber-status-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1px;
    margin: 0 0 22px;
    border: 1px solid rgba(42, 42, 58, 0.95);
    background: var(--cyber-border);
    overflow: hidden;
}

.cyber-status-strip > div {
    padding: 15px 18px;
    background: rgba(18, 18, 26, 0.9);
}

.cyber-status-strip strong {
    display: block;
    color: var(--cyber-fg);
    font-family: "Orbitron", monospace;
    font-size: 16px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.cyber-status-strip span {
    display: block;
    margin-top: 4px;
    color: var(--cyber-accent);
    font-size: 10px;
}

.cyber-panel,
.cyber-terminal,
.cyber-hud {
    padding: 20px !important;
    border: 1px solid rgba(42, 42, 58, 0.95) !important;
    background: linear-gradient(180deg, rgba(18, 18, 26, 0.94), rgba(10, 10, 15, 0.94)) !important;
    box-shadow: 0 22px 60px rgba(0, 0, 0, 0.42);
}

.cyber-panel:hover,
.cyber-terminal:hover,
.cyber-hud:hover {
    border-color: rgba(0, 255, 136, 0.7) !important;
    box-shadow: var(--shadow-neon), 0 28px 70px rgba(0, 0, 0, 0.48);
}

.cyber-section-title h3 {
    margin: 0 0 14px !important;
    color: var(--cyber-accent) !important;
    font-family: "Orbitron", monospace !important;
    font-size: 18px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    text-shadow: 0 0 16px rgba(0, 255, 136, 0.34);
}

.cyber-section-title h3::before {
    content: "> ";
    color: var(--cyber-secondary);
    text-shadow: var(--shadow-magenta);
}

.cyber-main-grid {
    align-items: stretch !important;
    display: grid !important;
    grid-template-columns: minmax(560px, 1.35fr) minmax(420px, 0.9fr) !important;
    gap: clamp(18px, 1.8vw, 34px) !important;
}

.cyber-main-grid > .form {
    gap: 20px !important;
}

.cyber-main-grid > * {
    min-width: 0 !important;
}

.cyber-bottom-grid {
    display: grid !important;
    grid-template-columns: minmax(520px, 1fr) minmax(420px, 0.75fr) !important;
    gap: clamp(18px, 1.8vw, 34px) !important;
    margin-top: clamp(18px, 1.8vw, 34px) !important;
}

.cyber-bottom-grid > * {
    min-width: 0 !important;
}

.cyber-tip {
    color: var(--cyber-muted-fg);
    border-left: 2px solid var(--cyber-tertiary);
    padding: 8px 12px;
    background: rgba(0, 212, 255, 0.06);
}

.cyber-tip em {
    color: #b7bece !important;
}

.cyber-notice {
    border: 1px solid rgba(255, 0, 255, 0.36);
    padding: 14px 16px;
    background: rgba(255, 0, 255, 0.065);
    box-shadow: var(--shadow-magenta);
    clip-path: var(--chamfer-sm);
}

.cyber-notice h3 {
    margin-top: 0 !important;
    color: var(--cyber-secondary) !important;
    font-family: "Orbitron", monospace !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

textarea,
input,
.wrap input,
.wrap textarea,
.wrap select,
.dropdown,
.token,
.container .wrap,
.gradio-dropdown,
.gradio-textbox,
.gradio-slider,
.gradio-checkbox,
.gradio-radio,
.gradio-file,
.gradio-dataframe,
.gradio-audio {
    background: var(--cyber-input) !important;
    color: var(--cyber-fg) !important;
    border-color: var(--cyber-border) !important;
    font-family: "JetBrains Mono", monospace !important;
}

textarea,
input {
    border: 1px solid var(--cyber-border) !important;
    box-shadow: inset 0 0 0 1px rgba(0, 255, 136, 0.03) !important;
    clip-path: var(--chamfer-sm);
}

textarea:focus,
input:focus,
.focused {
    border-color: var(--cyber-accent) !important;
    box-shadow: var(--shadow-neon) !important;
    outline: 2px solid transparent !important;
}

label,
.wrap .label-wrap span,
.block-info {
    color: var(--cyber-muted-fg) !important;
    font-size: 11px !important;
}

.wrap .label-wrap,
.block .label-wrap,
.label-wrap,
.block-label {
    background: transparent !important;
    color: var(--cyber-accent) !important;
    border: 0 !important;
}

.block-info {
    line-height: 1.6 !important;
}

.wrap .label-wrap span {
    color: var(--cyber-accent) !important;
}

.wrap .label-wrap span,
.block-info,
.prose p,
.markdown p,
.markdown li,
.form span,
.form div,
.gradio-container span,
.gradio-container p {
    color: #d9deea;
}

.gradio-container .block-info,
.gradio-container .info,
.gradio-container small {
    color: #9aa3b8 !important;
}

.gradio-container [data-testid="block-label"],
.gradio-container .label-wrap > span {
    background: rgba(0, 255, 136, 0.055) !important;
    color: var(--cyber-accent) !important;
    border: 1px solid rgba(0, 255, 136, 0.38) !important;
    box-shadow: inset 0 0 0 1px rgba(0, 212, 255, 0.08), 0 0 8px rgba(0, 255, 136, 0.12) !important;
    text-shadow: 0 0 8px rgba(0, 255, 136, 0.35);
    font-family: "Share Tech Mono", monospace !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.16em !important;
    line-height: 1 !important;
    text-transform: uppercase !important;
    padding: 4px 8px !important;
    clip-path: var(--chamfer-sm);
}

button,
.gradio-button {
    min-height: 46px !important;
    border: 2px solid var(--cyber-accent) !important;
    background: transparent !important;
    color: var(--cyber-accent) !important;
    font-family: "Share Tech Mono", monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    clip-path: var(--chamfer-sm);
    transition: all 120ms steps(4) !important;
}

button:hover,
.gradio-button:hover {
    background: var(--cyber-accent) !important;
    color: var(--cyber-bg) !important;
    box-shadow: var(--shadow-neon-lg) !important;
    transform: translateY(-1px) skewX(-1deg);
}

.cyber-primary button {
    background: var(--cyber-accent) !important;
    color: var(--cyber-bg) !important;
    box-shadow: var(--shadow-neon);
    animation: glitch 4.6s steps(3) infinite;
}

.cyber-secondary button {
    border-color: var(--cyber-secondary) !important;
    color: var(--cyber-secondary) !important;
}

.cyber-secondary button:hover {
    background: var(--cyber-secondary) !important;
    color: var(--cyber-bg) !important;
    box-shadow: var(--shadow-magenta) !important;
}

.tabs,
.tab-nav,
.accordion,
.form,
.block,
.panel,
.prose,
.markdown {
    background: transparent !important;
    color: var(--cyber-fg) !important;
}

.accordion {
    border: 1px solid rgba(42, 42, 58, 0.85) !important;
    clip-path: var(--chamfer-sm);
    background: rgba(12, 12, 18, 0.78) !important;
}

.accordion > .label-wrap {
    color: var(--cyber-tertiary) !important;
    font-family: "Orbitron", monospace !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

.cyber-log textarea,
.cyber-log input {
    color: var(--cyber-accent) !important;
    font-size: 12px !important;
    line-height: 1.65 !important;
    text-shadow: 0 0 8px rgba(0, 255, 136, 0.24);
}

.cyber-log::before {
    content: "";
    position: absolute;
    top: 12px;
    left: 18px;
    width: 10px;
    height: 10px;
    border-radius: 50% !important;
    background: var(--cyber-danger);
    box-shadow: 18px 0 #ffaa00, 36px 0 var(--cyber-accent);
}

.cyber-log {
    position: relative;
    padding-top: 16px !important;
}

.cyber-output .audio-container,
.cyber-output audio {
    background: rgba(0, 212, 255, 0.05) !important;
    border-color: rgba(0, 212, 255, 0.35) !important;
    clip-path: var(--chamfer-sm);
}

table,
thead,
tbody,
tr,
td,
th {
    background: rgba(10, 10, 15, 0.8) !important;
    color: var(--cyber-fg) !important;
    border-color: rgba(42, 42, 58, 0.9) !important;
    font-family: "JetBrains Mono", monospace !important;
}

th {
    color: var(--cyber-accent) !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}

tr:hover td {
    background: rgba(0, 255, 136, 0.06) !important;
}

input[type="range"] {
    accent-color: var(--cyber-accent);
}

input[type="checkbox"],
input[type="radio"] {
    accent-color: var(--cyber-accent);
}

.cyber-language {
    position: fixed !important;
    top: 18px !important;
    right: 22px !important;
    z-index: 10050 !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 5px !important;
    border: 1px solid rgba(0, 212, 255, 0.48) !important;
    background: rgba(10, 10, 15, 0.88) !important;
    box-shadow: var(--shadow-cyan);
    clip-path: var(--chamfer-sm);
    backdrop-filter: blur(10px);
}

.cyber-language [data-testid="block-label"],
.cyber-language .block-info,
.cyber-language > label,
.cyber-language legend {
    display: none !important;
}

.cyber-language .wrap,
.cyber-language .wrap > div,
.cyber-language fieldset,
.cyber-language .radio,
.cyber-language [role="radiogroup"] {
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: 0 !important;
}

.cyber-language label {
    min-width: 46px !important;
    height: 32px !important;
    padding: 0 12px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: 1px solid rgba(0, 255, 136, 0.3) !important;
    background: rgba(18, 18, 26, 0.92) !important;
    color: var(--cyber-tertiary) !important;
    font-family: "Share Tech Mono", monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    clip-path: var(--chamfer-sm);
    cursor: pointer !important;
    transition: all 120ms steps(4) !important;
}

.cyber-language label:hover {
    color: var(--cyber-bg) !important;
    background: var(--cyber-tertiary) !important;
    box-shadow: var(--shadow-cyan);
}

.cyber-language input:checked + span,
.cyber-language label:has(input:checked),
.cyber-language label.selected {
    color: var(--cyber-bg) !important;
    background: var(--cyber-accent) !important;
    border-color: var(--cyber-accent) !important;
    box-shadow: var(--shadow-neon);
}

.cyber-language input {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

.cyber-language span {
    color: inherit !important;
    background: transparent !important;
    border: 0 !important;
}

.cyber-credit {
    margin-top: 20px;
    padding: 14px 18px;
    border: 1px solid rgba(0, 255, 136, 0.28);
    background: rgba(10, 10, 15, 0.72);
    color: #cbd3e3;
    font-family: "Share Tech Mono", monospace;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    clip-path: var(--chamfer-sm);
}

.cyber-credit a {
    color: var(--cyber-accent) !important;
    margin-left: 12px;
    text-decoration: none !important;
    text-shadow: var(--shadow-neon);
}

.gradio-container label,
.gradio-container legend,
.gradio-container .label-wrap,
.gradio-container .label-wrap *,
.gradio-container [data-testid="block-label"],
.gradio-container [data-testid="block-label"] *,
.gradio-container .wrap .label-wrap span {
    color: #dfffee !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #aab3c8 !important;
    opacity: 1 !important;
}

.gradio-container .block-info,
.gradio-container .info,
.gradio-container .svelte-1gfkn6j,
.gradio-container .svelte-1gfkn6j * {
    color: #aeb8cc !important;
}

.gradio-container .cyber-section-title *,
.gradio-container .cyber-notice h3,
.gradio-container th,
.gradio-container .accordion > .label-wrap,
.gradio-container .accordion > .label-wrap * {
    color: var(--cyber-accent) !important;
}

.gradio-container label.svelte-1b6s6s,
.gradio-container label.svelte-1b6s6s *,
.gradio-container label[data-testid="block-label"],
.gradio-container label[data-testid="block-label"] *,
.gradio-container .float,
.gradio-container .float * {
    color: #dfffee !important;
    fill: #dfffee !important;
    stroke: currentColor !important;
}

body .gradio-container.gradio-container label.svelte-1b6s6s[data-testid="block-label"],
body .gradio-container.gradio-container label.svelte-1b6s6s[data-testid="block-label"].float,
body .gradio-container.gradio-container .block label.svelte-1b6s6s[data-testid="block-label"] {
    color: var(--cyber-accent) !important;
    background: rgba(0, 255, 136, 0.055) !important;
    border: 1px solid rgba(0, 255, 136, 0.38) !important;
    box-shadow: inset 0 0 0 1px rgba(0, 212, 255, 0.08), 0 0 8px rgba(0, 255, 136, 0.12) !important;
    padding: 4px 8px !important;
    clip-path: var(--chamfer-sm);
}

body .gradio-container.gradio-container label.svelte-1b6s6s[data-testid="block-label"] span,
body .gradio-container.gradio-container label.svelte-1b6s6s[data-testid="block-label"] svg {
    color: var(--cyber-accent) !important;
    stroke: var(--cyber-accent) !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

@media (min-width: 1800px) {
    .cyber-hero {
        padding: 48px 56px;
    }
    .cyber-panel,
    .cyber-terminal,
    .cyber-hud {
        padding: 28px !important;
    }
    .cyber-main-grid {
        grid-template-columns: minmax(760px, 1.45fr) minmax(560px, 0.95fr) !important;
    }
    .cyber-bottom-grid {
        grid-template-columns: minmax(760px, 1.2fr) minmax(520px, 0.8fr) !important;
    }
    textarea {
        min-height: 190px !important;
    }
    .cyber-log textarea {
        min-height: 260px !important;
    }
}

@media (min-width: 2200px) {
    .cyber-main-grid {
        grid-template-columns: minmax(900px, 1.5fr) minmax(680px, 1fr) !important;
    }
}

.cyber-language label {
    color: var(--cyber-tertiary) !important;
}

.cyber-side-stack {
    gap: 18px !important;
}

footer {
    display: none !important;
}

@keyframes blink {
    50% { opacity: 0; }
}

@keyframes glitch {
    0%, 100% { transform: translate(0); }
    20% { transform: translate(-2px, 2px); }
    40% { transform: translate(2px, -2px); }
    60% { transform: translate(-1px, -1px); }
    80% { transform: translate(1px, 1px); }
}

@keyframes scanline {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(120vh); }
}

@keyframes rgbShift {
    0%, 100% { text-shadow: -2px 0 #ff00ff, 2px 0 #00d4ff, 0 0 26px rgba(0, 255, 136, 0.42); }
    50% { text-shadow: 2px 0 #ff00ff, -2px 0 #00d4ff, 0 0 30px rgba(0, 212, 255, 0.34); }
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 14px !important;
    }
    .cyber-hero {
        padding: 24px 18px;
    }
    .cyber-hero__content,
    .cyber-status-strip,
    .cyber-main-grid,
    .cyber-bottom-grid {
        grid-template-columns: 1fr;
    }
    .cyber-language {
        top: 10px !important;
        right: 10px !important;
        transform: scale(0.92);
        transform-origin: top right;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation: none !important;
        transition: none !important;
    }
}

/* Bitcoin DeFi theme override */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --defi-bg: #030304;
    --defi-surface: #0f1115;
    --defi-surface-2: rgba(255, 255, 255, 0.045);
    --defi-fg: #ffffff;
    --defi-muted: #94a3b8;
    --defi-border: rgba(255, 255, 255, 0.1);
    --defi-orange: #f7931a;
    --defi-burnt: #ea580c;
    --defi-gold: #ffd600;
    --defi-radius-xl: 16px;
    --defi-radius-lg: 12px;
    --defi-glow: 0 0 28px -6px rgba(247, 147, 26, 0.55);
    --defi-glow-soft: 0 0 60px -18px rgba(247, 147, 26, 0.28);
}

html,
body,
gradio-app,
.gradio-container {
    background:
        radial-gradient(circle at 18% 8%, rgba(247, 147, 26, 0.15), transparent 34rem),
        radial-gradient(circle at 88% 0%, rgba(255, 214, 0, 0.1), transparent 28rem),
        radial-gradient(circle at 50% 100%, rgba(234, 88, 12, 0.12), transparent 36rem),
        var(--defi-bg) !important;
    color: var(--defi-fg) !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

body::before {
    background-image:
        linear-gradient(to right, rgba(30, 41, 59, 0.44) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(30, 41, 59, 0.44) 1px, transparent 1px) !important;
    background-size: 50px 50px !important;
    mask-image: radial-gradient(circle at center, black 42%, transparent 100%);
    mix-blend-mode: normal !important;
    opacity: 0.42 !important;
}

body::after {
    display: none !important;
}

* {
    border-radius: var(--defi-radius-lg) !important;
    letter-spacing: 0 !important;
}

.cyber-hero,
.cyber-panel,
.cyber-terminal,
.cyber-hud,
.cyber-status-strip {
    clip-path: none !important;
    border-radius: 24px !important;
}

.cyber-hero {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.62fr);
    align-items: center;
    gap: clamp(32px, 5vw, 96px);
    min-height: clamp(300px, 30vw, 460px);
    padding: clamp(36px, 5vw, 86px) !important;
    border: 1px solid var(--defi-border) !important;
    background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.075), rgba(255, 255, 255, 0.025)),
        rgba(15, 17, 21, 0.72) !important;
    box-shadow: var(--defi-glow-soft), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(18px);
}

.cyber-hero::before,
.cyber-panel::before,
.cyber-terminal::before,
.cyber-hud::before {
    display: none !important;
}

.cyber-hero__content {
    max-width: 980px;
}

.defi-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 24px;
    padding: 9px 14px;
    border: 1px solid rgba(247, 147, 26, 0.28);
    background: rgba(247, 147, 26, 0.08);
    color: var(--defi-orange);
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.12em !important;
    text-transform: uppercase;
    border-radius: 999px !important;
    box-shadow: 0 0 22px -10px rgba(247, 147, 26, 0.8);
}

.defi-badge span {
    width: 8px;
    height: 8px;
    border-radius: 999px !important;
    background: var(--defi-orange);
    box-shadow: 0 0 0 6px rgba(247, 147, 26, 0.12), 0 0 16px rgba(247, 147, 26, 0.8);
}

.cyber-glitch {
    margin: 0 0 22px !important;
    color: #ffffff !important;
    background-image: linear-gradient(90deg, #ffffff 0%, #ffffff 44%, var(--defi-orange) 68%, var(--defi-gold) 100%) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-family: "Space Grotesk", "Inter", sans-serif !important;
    font-size: clamp(64px, 9vw, 148px) !important;
    font-weight: 700 !important;
    line-height: 0.86 !important;
    letter-spacing: -0.06em !important;
    text-shadow: none !important;
    animation: none !important;
}

.cyber-glitch::before,
.cyber-glitch::after {
    display: none !important;
}

.cyber-type {
    max-width: 820px !important;
    color: #dbe3ef !important;
    font-family: "Inter", sans-serif !important;
    font-size: clamp(17px, 1.25vw, 23px) !important;
    line-height: 1.65 !important;
    text-shadow: none !important;
}

.cyber-cursor {
    display: none !important;
}

.defi-hero-note {
    width: fit-content;
    margin-top: 28px;
    padding: 13px 18px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(0, 0, 0, 0.28);
    color: var(--defi-muted);
    font-family: "JetBrains Mono", monospace;
    font-size: 13px;
    border-radius: 999px !important;
}

.defi-orb {
    position: relative;
    width: min(30vw, 430px);
    min-width: 320px;
    aspect-ratio: 1;
    justify-self: center;
    animation: defiFloat 8s ease-in-out infinite;
}

.defi-orb__ring {
    position: absolute;
    inset: 8%;
    border: 1px solid rgba(247, 147, 26, 0.4);
    border-radius: 50% !important;
    box-shadow: 0 0 40px -12px rgba(247, 147, 26, 0.7), inset 0 0 30px -12px rgba(255, 214, 0, 0.7);
}

.defi-orb__ring--outer {
    animation: spin 12s linear infinite;
}

.defi-orb__ring--inner {
    inset: 21%;
    border-color: rgba(255, 214, 0, 0.36);
    animation: spinReverse 16s linear infinite;
}

.defi-orb__core {
    position: absolute;
    inset: 31%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: clamp(7px, 0.75vw, 12px);
    border-radius: 50% !important;
    background: radial-gradient(circle at 35% 28%, var(--defi-gold), var(--defi-orange) 48%, var(--defi-burnt));
    color: #1b0b00;
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(72px, 7vw, 118px);
    font-weight: 700;
    box-shadow: 0 0 60px rgba(247, 147, 26, 0.55), 0 0 120px rgba(255, 214, 0, 0.18);
}

.defi-orb__stat {
    position: absolute;
    padding: 10px 14px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(0, 0, 0, 0.42);
    color: #fff;
    font-family: "JetBrains Mono", monospace;
    font-size: 13px;
    font-weight: 600;
    border-radius: 999px !important;
    backdrop-filter: blur(12px);
    box-shadow: var(--defi-glow);
}

.defi-orb__stat--top {
    top: 16%;
    right: 2%;
}

.defi-orb__stat--bottom {
    left: 3%;
    bottom: 18%;
}

.cyber-status-strip {
    border: 1px solid var(--defi-border) !important;
    background: rgba(255, 255, 255, 0.04) !important;
    box-shadow: none !important;
    overflow: hidden;
}

.cyber-status-strip > div {
    background: rgba(15, 17, 21, 0.78) !important;
    padding: 20px 24px !important;
}

.cyber-status-strip strong {
    color: #fff !important;
    font-family: "Space Grotesk", sans-serif !important;
    font-size: 18px !important;
    letter-spacing: -0.02em !important;
}

.cyber-status-strip span {
    color: var(--defi-orange) !important;
    font-family: "JetBrains Mono", monospace !important;
}

.cyber-panel,
.cyber-terminal,
.cyber-hud {
    padding: clamp(22px, 2vw, 34px) !important;
    border: 1px solid var(--defi-border) !important;
    background: rgba(15, 17, 21, 0.78) !important;
    box-shadow: 0 0 50px -12px rgba(247, 147, 26, 0.12) !important;
    backdrop-filter: blur(16px);
    transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
}

.cyber-panel:hover,
.cyber-terminal:hover,
.cyber-hud:hover {
    transform: translateY(-2px);
    border-color: rgba(247, 147, 26, 0.45) !important;
    box-shadow: 0 0 44px -14px rgba(247, 147, 26, 0.32) !important;
}

.cyber-section-title h3 {
    color: #fff !important;
    font-family: "Space Grotesk", sans-serif !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    text-transform: none !important;
    text-shadow: none !important;
}

.cyber-section-title h3::before {
    content: "";
    display: inline-block;
    width: 10px;
    height: 10px;
    margin-right: 10px;
    border-radius: 999px !important;
    background: var(--defi-orange);
    box-shadow: 0 0 16px rgba(247, 147, 26, 0.9);
}

textarea,
input,
.wrap input,
.wrap textarea,
.wrap select,
.dropdown,
.token,
.container .wrap,
.gradio-dropdown,
.gradio-textbox,
.gradio-slider,
.gradio-checkbox,
.gradio-radio,
.gradio-file,
.gradio-dataframe,
.gradio-audio {
    background: rgba(0, 0, 0, 0.38) !important;
    color: #fff !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
    font-family: "Inter", sans-serif !important;
}

textarea,
input {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

textarea:focus,
input:focus,
.focused {
    border-color: rgba(247, 147, 26, 0.9) !important;
    box-shadow: 0 10px 28px -16px rgba(247, 147, 26, 0.75) !important;
}

.gradio-container [data-testid="block-label"],
.gradio-container .label-wrap > span,
body .gradio-container.gradio-container label.svelte-1b6s6s[data-testid="block-label"],
body .gradio-container.gradio-container label.svelte-1b6s6s[data-testid="block-label"].float,
body .gradio-container.gradio-container .block label.svelte-1b6s6s[data-testid="block-label"] {
    background: rgba(255, 255, 255, 0.055) !important;
    color: var(--defi-muted) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: none !important;
    font-family: "JetBrains Mono", monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 999px !important;
    clip-path: none !important;
}

button,
.gradio-button {
    min-height: 46px !important;
    border: 0 !important;
    background: linear-gradient(90deg, var(--defi-burnt), var(--defi-orange)) !important;
    color: #fff !important;
    font-family: "Inter", sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-radius: 999px !important;
    box-shadow: var(--defi-glow) !important;
}

button:hover,
.gradio-button:hover {
    background: linear-gradient(90deg, var(--defi-orange), var(--defi-gold)) !important;
    color: #170900 !important;
    transform: translateY(-1px) scale(1.015) !important;
    box-shadow: 0 0 36px -4px rgba(247, 147, 26, 0.72) !important;
}

.cyber-secondary button {
    background: transparent !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    color: #fff !important;
    box-shadow: none !important;
}

.cyber-secondary button:hover {
    border-color: rgba(247, 147, 26, 0.65) !important;
    background: rgba(247, 147, 26, 0.1) !important;
    color: #fff !important;
}

.cyber-param-group {
    gap: 8px !important;
}

.cyber-reset-button,
.cyber-reset-button button,
.cyber-reset-button .gradio-button {
    width: fit-content !important;
    min-width: 118px !important;
    min-height: 34px !important;
    padding: 0 14px !important;
    align-self: flex-start !important;
    border: 1px solid rgba(255, 214, 0, 0.34) !important;
    background: rgba(247, 147, 26, 0.08) !important;
    color: var(--defi-gold) !important;
    box-shadow: none !important;
    font-size: 12px !important;
    letter-spacing: 0.03em !important;
    text-transform: none !important;
}

.cyber-reset-button:hover,
.cyber-reset-button button:hover,
.cyber-reset-button .gradio-button:hover {
    border-color: rgba(247, 147, 26, 0.78) !important;
    background: rgba(247, 147, 26, 0.18) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

.accordion {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 16px !important;
    clip-path: none !important;
}

.cyber-tip {
    color: var(--defi-muted) !important;
    border-left: 2px solid rgba(247, 147, 26, 0.8) !important;
    background: rgba(247, 147, 26, 0.055) !important;
}

.cyber-inline-note {
    margin: 4px 0 14px !important;
    padding: 0 0 0 12px !important;
    border-left: 2px solid rgba(247, 147, 26, 0.75) !important;
    background: transparent !important;
    color: var(--defi-muted) !important;
    box-shadow: none !important;
}

.cyber-inline-note p,
.cyber-inline-note span {
    margin: 0 !important;
    color: var(--defi-muted) !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
}

.cyber-notice {
    border: 1px solid rgba(247, 147, 26, 0.28) !important;
    background: rgba(247, 147, 26, 0.07) !important;
    box-shadow: var(--defi-glow-soft) !important;
    border-radius: 16px !important;
    clip-path: none !important;
}

.cyber-notice h3 {
    color: var(--defi-orange) !important;
    font-family: "Space Grotesk", sans-serif !important;
    text-transform: none !important;
}

.cyber-log::before {
    background: #ef4444 !important;
    box-shadow: 18px 0 #f59e0b, 36px 0 #22c55e !important;
}

table,
thead,
tbody,
tr,
td,
th {
    background: rgba(0, 0, 0, 0.18) !important;
    border-color: rgba(255, 255, 255, 0.08) !important;
}

th {
    color: var(--defi-orange) !important;
}

tr:hover td {
    background: rgba(247, 147, 26, 0.06) !important;
}

input[type="range"],
input[type="checkbox"],
input[type="radio"] {
    accent-color: var(--defi-orange);
}

.cyber-language {
    border-color: rgba(247, 147, 26, 0.42) !important;
    background: rgba(3, 3, 4, 0.78) !important;
    box-shadow: 0 0 30px -10px rgba(247, 147, 26, 0.45) !important;
    border-radius: 999px !important;
    clip-path: none !important;
}

.cyber-language label {
    border-color: rgba(255, 255, 255, 0.12) !important;
    background: transparent !important;
    color: var(--defi-muted) !important;
    border-radius: 999px !important;
}

.cyber-language label:hover,
.cyber-language label:has(input:checked),
.cyber-language label.selected {
    color: #170900 !important;
    background: linear-gradient(90deg, var(--defi-orange), var(--defi-gold)) !important;
    box-shadow: var(--defi-glow) !important;
}

.cyber-credit {
    border-color: rgba(255, 255, 255, 0.1) !important;
    background: rgba(255, 255, 255, 0.04) !important;
    color: var(--defi-muted) !important;
    border-radius: 16px !important;
    clip-path: none !important;
}

.cyber-credit a {
    color: var(--defi-orange) !important;
    text-shadow: none !important;
}

@keyframes defiFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-18px); }
}

@keyframes spinReverse {
    to { transform: rotate(-360deg); }
}

@media (max-width: 1100px) {
    .cyber-hero {
        grid-template-columns: 1fr;
    }
    .defi-orb {
        width: min(78vw, 360px);
        min-width: 260px;
    }
}

@media (max-width: 700px) {
    .cyber-hero {
        padding: 28px 20px !important;
    }
    .defi-hero-note {
        width: auto;
        border-radius: 16px !important;
    }
    .defi-orb {
        display: none;
    }
}

/* Targeted QA refinements */
.cyber-hero > .form {
    display: grid !important;
    grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.62fr) !important;
    align-items: center !important;
    gap: clamp(32px, 5vw, 96px) !important;
    width: 100% !important;
}

.cyber-hero__content {
    min-width: 0 !important;
}

.cyber-hero-orb {
    display: flex !important;
    justify-content: center !important;
    min-width: 0 !important;
}

.cyber-language {
    position: static !important;
    inset: auto !important;
    transform: none !important;
    width: fit-content !important;
    max-width: 100% !important;
    margin: 24px 0 0 !important;
    padding: 5px !important;
    overflow: hidden !important;
    border: 1px solid rgba(247, 147, 26, 0.55) !important;
    background: rgba(3, 3, 4, 0.7) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 24px -10px rgba(247, 147, 26, 0.8) !important;
}

.cyber-language [data-testid="block-label"],
.cyber-language [data-testid="block-info"],
.cyber-language .block-info,
.cyber-language legend,
.cyber-language > label {
    display: none !important;
}

.cyber-language fieldset,
.cyber-language .wrap,
.cyber-language .wrap > div,
.cyber-language .radio,
.cyber-language [role="radiogroup"] {
    display: flex !important;
    align-items: center !important;
    gap: 5px !important;
    width: auto !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.cyber-language label {
    width: 58px !important;
    min-width: 58px !important;
    height: 40px !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    line-height: 1 !important;
    border-radius: 999px !important;
}

.cyber-language span {
    color: inherit !important;
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}

@media (max-width: 1100px) {
    .cyber-hero > .form {
        grid-template-columns: 1fr !important;
    }
}

@media (max-width: 700px) {
    .cyber-language {
        margin-top: 18px !important;
    }
}

.cyber-log::before {
    display: none !important;
}

.cyber-log {
    padding-top: 0 !important;
}

.cyber-input.gradio-dropdown,
.cyber-input .wrap.svelte-1sk0pyu,
.cyber-input .wrap-inner.svelte-1sk0pyu,
.cyber-input .secondary-wrap.svelte-1sk0pyu {
    background: rgba(3, 3, 4, 0.72) !important;
    border-color: rgba(247, 147, 26, 0.34) !important;
    border-radius: 12px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
}

.cyber-input .wrap.svelte-1sk0pyu {
    min-height: 44px !important;
    padding: 4px 10px !important;
}

.cyber-input input.border-none.svelte-1sk0pyu,
.cyber-input .wrap-inner.svelte-1sk0pyu input,
.cyber-input .secondary-wrap.svelte-1sk0pyu input {
    background: transparent !important;
    color: #ffffff !important;
    border: 0 !important;
    box-shadow: none !important;
    clip-path: none !important;
    font-family: "Inter", sans-serif !important;
    font-size: 14px !important;
}

.cyber-input .wrap.svelte-1sk0pyu:focus-within,
.cyber-input .wrap-inner.svelte-1sk0pyu:focus-within {
    border-color: rgba(247, 147, 26, 0.85) !important;
    box-shadow: 0 0 0 1px rgba(247, 147, 26, 0.25), 0 0 24px -12px rgba(247, 147, 26, 0.85) !important;
}

.cyber-input .dropdown-arrow,
.cyber-input svg,
.cyber-input .secondary-wrap.svelte-1sk0pyu svg {
    color: var(--defi-gold) !important;
    stroke: var(--defi-gold) !important;
}

.gradio-container [data-testid="block-label"],
.gradio-container [data-testid="block-info"],
.gradio-container label.container > span[data-testid="block-info"],
body .gradio-container.gradio-container label.svelte-1b6s6s[data-testid="block-label"],
body .gradio-container.gradio-container label.svelte-1b6s6s[data-testid="block-label"].float,
body .gradio-container.gradio-container .block label.svelte-1b6s6s[data-testid="block-label"] {
    background: rgba(247, 147, 26, 0.12) !important;
    color: #ffd98a !important;
    border: 1px solid rgba(247, 147, 26, 0.42) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 18px -12px rgba(247, 147, 26, 0.9) !important;
}

body .gradio-container.gradio-container .form span[data-testid="block-info"],
body .gradio-container.gradio-container label.container > span[data-testid="block-info"] {
    color: #ffd98a !important;
}

body .gradio-container.gradio-container label.svelte-1b6s6s[data-testid="block-label"] span,
body .gradio-container.gradio-container label.svelte-1b6s6s[data-testid="block-label"] svg {
    color: #ffd98a !important;
    stroke: #ffd98a !important;
}

.gradio-container input[type="checkbox"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    margin: 0 !important;
    border: 1px solid rgba(255, 214, 0, 0.42) !important;
    border-radius: 6px !important;
    background: rgba(3, 3, 4, 0.72) !important;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04) !important;
    cursor: pointer !important;
    clip-path: none !important;
}

.gradio-container input[type="checkbox"]:checked {
    border-color: rgba(255, 214, 0, 0.95) !important;
    background:
        linear-gradient(135deg, var(--defi-orange), var(--defi-gold)) !important;
    box-shadow: 0 0 0 4px rgba(247, 147, 26, 0.16), 0 0 20px -8px rgba(247, 147, 26, 0.95) !important;
}

.gradio-container input[type="checkbox"]:checked::after {
    content: "";
    display: block;
    width: 8px;
    height: 4px;
    margin: 4px auto 0;
    border-left: 2px solid #170900;
    border-bottom: 2px solid #170900;
    transform: rotate(-45deg);
}

.gradio-container input[type="checkbox"] + span,
.gradio-container input[type="checkbox"] ~ span,
.gradio-container label:has(input[type="checkbox"]) {
    color: #dbe3ef !important;
}

.gradio-container label:has(input[type="checkbox"]:checked) {
    color: #ffffff !important;
}

.cyber-tip,
.cyber-tip.prose,
.block.cyber-tip,
.gradio-container .cyber-tip {
    margin: 8px 0 14px !important;
    padding: 0 0 0 12px !important;
    border: 0 !important;
    border-left: 2px solid rgba(247, 147, 26, 0.78) !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.cyber-tip p,
.cyber-tip span,
.cyber-tip em {
    margin: 0 !important;
    background: transparent !important;
    color: #b8c3d8 !important;
    line-height: 1.55 !important;
}

.gradio-container .cyber-panel > .block:has(> button.label-wrap) {
    border: 1px solid rgba(247, 147, 26, 0.38) !important;
    background-color: rgba(15, 17, 21, 0.9) !important;
    background-image: linear-gradient(90deg, rgba(247, 147, 26, 0.22), rgba(255, 214, 0, 0.04)) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 12px 30px -24px rgba(247, 147, 26, 0.85) !important;
}

.gradio-container .cyber-panel > .block > button.label-wrap {
    min-height: 50px !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 16px !important;
    padding: 0 14px 0 18px !important;
    border: 0 !important;
    border-bottom: 1px solid rgba(247, 147, 26, 0.22) !important;
    border-radius: 0 !important;
    background-color: rgba(247, 147, 26, 0.13) !important;
    background-image: linear-gradient(90deg, rgba(247, 147, 26, 0.26), rgba(255, 214, 0, 0.06)) !important;
    box-shadow: none !important;
    clip-path: none !important;
    cursor: pointer !important;
}

.gradio-container .cyber-panel > .block > button.label-wrap span:not(.icon) {
    color: #ffffff !important;
    background: transparent !important;
    border: 0 !important;
    font-family: "Space Grotesk", "Inter", sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    line-height: 1.2 !important;
    text-transform: uppercase !important;
    text-shadow: 0 0 18px rgba(247, 147, 26, 0.22) !important;
}

.gradio-container .cyber-panel > .block > button.label-wrap .icon {
    flex: 0 0 34px !important;
    width: 34px !important;
    height: 34px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-left: auto !important;
    border: 1px solid rgba(247, 147, 26, 0.58) !important;
    border-radius: 999px !important;
    background: rgba(247, 147, 26, 0.12) !important;
    color: var(--defi-gold) !important;
    font-size: 13px !important;
    line-height: 1 !important;
    box-shadow: 0 0 18px -8px rgba(247, 147, 26, 0.9) !important;
}

.gradio-container .cyber-panel > .block > button.label-wrap:hover {
    background-color: rgba(247, 147, 26, 0.18) !important;
    background-image: linear-gradient(90deg, rgba(247, 147, 26, 0.34), rgba(255, 214, 0, 0.1)) !important;
    transform: none !important;
}

.defi-orb__core::before {
    content: "";
    position: absolute;
    inset: 14%;
    border-radius: 50% !important;
    background:
        radial-gradient(circle, rgba(255, 255, 255, 0.16), transparent 58%),
        linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px);
    background-size: auto, 30px 30px, 30px 30px;
    opacity: 0.42;
}

.defi-orb__core span {
    position: relative;
    z-index: 1;
    width: clamp(5px, 0.5vw, 8px);
    border-radius: 999px !important;
    background: linear-gradient(180deg, #fff7d6, var(--defi-gold), var(--defi-orange));
    box-shadow: 0 0 18px rgba(255, 214, 0, 0.75);
    animation: wavePulse 1.35s ease-in-out infinite;
}

.defi-orb__core span:nth-child(1) { height: 28%; animation-delay: -0.28s; }
.defi-orb__core span:nth-child(2) { height: 52%; animation-delay: -0.14s; }
.defi-orb__core span:nth-child(3) { height: 76%; animation-delay: 0s; }
.defi-orb__core span:nth-child(4) { height: 50%; animation-delay: 0.14s; }
.defi-orb__core span:nth-child(5) { height: 32%; animation-delay: 0.28s; }

@keyframes wavePulse {
    0%, 100% { transform: scaleY(0.62); opacity: 0.72; }
    50% { transform: scaleY(1.08); opacity: 1; }
}

.gradio-file button,
.gradio-file button.center,
.cyber-input .center,
div[id^="component-"] button.center {
    border-radius: 10px !important;
}

.gradio-container input[type="range"] {
    height: 28px !important;
    background: transparent !important;
    cursor: pointer !important;
}

.gradio-container input[type="range"]::-webkit-slider-runnable-track {
    height: 8px !important;
    border-radius: 999px !important;
    background: linear-gradient(90deg, var(--defi-orange), var(--defi-gold)) !important;
    box-shadow: 0 0 18px rgba(247, 147, 26, 0.45) !important;
}

.gradio-container input[type="range"]::-webkit-slider-thumb {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 22px !important;
    height: 22px !important;
    margin-top: -7px !important;
    border: 3px solid #ffffff !important;
    border-radius: 50% !important;
    background: var(--defi-orange) !important;
    box-shadow: 0 0 0 5px rgba(247, 147, 26, 0.22), 0 0 22px rgba(247, 147, 26, 0.85) !important;
}

.gradio-container input[type="range"]::-moz-range-track {
    height: 8px !important;
    border-radius: 999px !important;
    background: linear-gradient(90deg, var(--defi-orange), var(--defi-gold)) !important;
    box-shadow: 0 0 18px rgba(247, 147, 26, 0.45) !important;
}

.gradio-container input[type="range"]::-moz-range-thumb {
    width: 18px !important;
    height: 18px !important;
    border: 3px solid #ffffff !important;
    border-radius: 50% !important;
    background: var(--defi-orange) !important;
    box-shadow: 0 0 0 5px rgba(247, 147, 26, 0.22), 0 0 22px rgba(247, 147, 26, 0.85) !important;
}

.gradio-dataframe,
.gradio-dataframe .table-wrap,
.gradio-dataframe table,
.gradio-dataframe .table,
.gradio-dataframe button.disable_click,
.gradio-dataframe svelte-virtual-table-viewport,
.cyber-table,
.cyber-table .table-wrap,
.cyber-table table,
.cyber-table .table,
.cyber-table button.disable_click,
.cyber-table svelte-virtual-table-viewport {
    border-radius: 0 !important;
}

.gradio-dataframe .table-wrap,
.gradio-dataframe table,
.gradio-dataframe .table,
.cyber-table .table-wrap,
.cyber-table table,
.cyber-table .table {
    border: 1px solid #ffffff !important;
    border-color: #ffffff !important;
}

/* Final global control polish */
.gradio-container input[type="checkbox"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    margin: 0 !important;
    border: 1px solid rgba(255, 214, 0, 0.52) !important;
    border-radius: 6px !important;
    background: rgba(3, 3, 4, 0.72) !important;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04) !important;
    clip-path: none !important;
}

.gradio-container input[type="checkbox"]:checked {
    border-color: rgba(255, 214, 0, 0.98) !important;
    background: linear-gradient(135deg, var(--defi-orange), var(--defi-gold)) !important;
    box-shadow: 0 0 0 4px rgba(247, 147, 26, 0.16), 0 0 20px -8px rgba(247, 147, 26, 0.95) !important;
}

.gradio-container input[type="checkbox"]:checked::after {
    content: "";
    display: block;
    width: 8px;
    height: 4px;
    margin: 4px auto 0;
    border-left: 2px solid #170900;
    border-bottom: 2px solid #170900;
    transform: rotate(-45deg);
}

.cyber-tip,
.cyber-tip.prose,
.block.cyber-tip,
.gradio-container .cyber-tip,
.cyber-inline-note,
.cyber-inline-note.prose,
.block.cyber-inline-note,
.gradio-container .cyber-inline-note {
    border-top: 0 !important;
    border-right: 0 !important;
    border-bottom: 0 !important;
    border-left: 2px solid rgba(247, 147, 26, 0.78) !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
"""

# ── Speaker presets ──────────────────────────────────────────────────────────

LANG_MAP = {code: name for name, code in SUPPORTED_LANGS}

def _build_speaker_options():
    """Build speaker preset options organized by language and version."""
    options = [DEFAULT_SPEAKER]
    for lang_name, lang_code in SUPPORTED_LANGS:
        for ver in ["v2", "v1"]:
            for speaker_id in range(10):
                if ver == "v2":
                    label = f"[{lang_name}] v2/{lang_code}_speaker_{speaker_id}"
                    value = f"v2/{lang_code}_speaker_{speaker_id}"
                else:
                    label = f"[{lang_name}] {lang_code}_speaker_{speaker_id}"
                    value = f"{lang_code}_speaker_{speaker_id}"
                options.append((label, value))
    return options

SPEAKER_OPTIONS = _build_speaker_options()

# ── Long-form text splitting ─────────────────────────────────────────────────

def split_text_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks for long-form generation."""
    # Split on sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?。！？\n])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# ── Progress tracking ────────────────────────────────────────────────────────

class ProgressTracker:
    """Thread-safe progress tracker that captures tqdm output."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stage = "Idle"
        self._progress = 0.0
        self._message = ""
        self._log_lines = []

    def update(self, stage: str, progress: float, message: str = ""):
        with self._lock:
            self._stage = stage
            self._progress = min(max(progress, 0.0), 1.0)
            self._message = message
            ts = datetime.now().strftime("%H:%M:%S")
            self._log_lines.append(f"[{ts}] {stage}: {message}")

    def log(self, message: str):
        with self._lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self._log_lines.append(f"[{ts}] {message}")

    def get_status(self) -> tuple[str, float, str]:
        with self._lock:
            return self._stage, self._progress, self._message

    def get_log(self) -> str:
        with self._lock:
            return "\n".join(self._log_lines[-50:])  # Last 50 lines

    def reset(self):
        with self._lock:
            self._stage = "Idle"
            self._progress = 0.0
            self._message = ""
            self._log_lines = []


# Global progress tracker
_tracker = ProgressTracker()


def _patched_tqdm(iterable=None, total=None, disable=False, desc=None, **kwargs):
    """Replacement for tqdm that reports progress to our tracker."""
    stage = _tracker.get_status()[0]
    if iterable is not None:
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        count = 0
        for item in iterable:
            count += 1
            if total and not disable:
                pct = count / total
                _tracker.update(stage, pct, f"{desc or 'Processing'} ({count}/{total})")
            yield item
    else:
        # Used as pbar = tqdm.tqdm(total=...)
        class _FakePbar:
            def __init__(self):
                self.n = 0
                self.total = total or 768

            def update(self, n):
                self.n += n
                if self.total and not disable:
                    pct = self.n / self.total
                    _tracker.update(stage, pct, f"{desc or 'Processing'} ({self.n}/{self.total})")

            def close(self):
                pass

            def refresh(self):
                pass

            @property
            def total(self):
                return self._total

            @total.setter
            def total(self, val):
                self._total = val

        return _FakePbar()


# ── Generation history ───────────────────────────────────────────────────────

class GenerationHistory:
    """Stores generation results for later retrieval."""

    def __init__(self, max_items=50):
        self._items = []
        self._max = max_items
        self._next_id = 1

    def add(self, audio_path: str, text: str, speaker: str, duration_s: float):
        entry = {
            "id": self._next_id,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "audio_path": audio_path,
            "text": text[:100] + ("..." if len(text) > 100 else ""),
            "speaker": speaker or "Default",
            "duration": round(duration_s, 2),
        }
        self._next_id += 1
        self._items.insert(0, entry)
        if len(self._items) > self._max:
            self._items = self._items[:self._max]
        return entry

    def get_table(self) -> list[list]:
        return [
            [item["time"], item["text"], item["speaker"], f"{item['duration']}s"]
            for item in self._items
        ]

    def get_audio_by_row(self, row_idx: int) -> Optional[str]:
        if 0 <= row_idx < len(self._items):
            audio_path = self._items[row_idx]["audio_path"]
            if os.path.exists(audio_path):
                return audio_path
        return None


_history = GenerationHistory()

# ── Output directory ─────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Core generation with progress ────────────────────────────────────────────

def generate_with_progress(
    text: str,
    history_prompt,
    text_temp: float,
    waveform_temp: float,
    silent: bool = True,
):
    """Run the Bark pipeline with progress tracking."""
    import tqdm as tqdm_module

    # Monkey-patch tqdm for progress tracking
    original_tqdm = tqdm_module.tqdm
    tqdm_module.tqdm = _patched_tqdm

    try:
        _tracker.update("Text → Semantic", 0.0, "Starting text-to-semantic conversion...")
        semantic_tokens = generate_text_semantic(
            text,
            history_prompt=history_prompt,
            temp=text_temp,
            silent=silent,
            use_kv_caching=True,
        )

        _tracker.update("Semantic → Coarse", 0.0, "Generating coarse audio tokens...")
        coarse_tokens = generate_coarse(
            semantic_tokens,
            history_prompt=history_prompt,
            temp=waveform_temp,
            silent=silent,
            use_kv_caching=True,
        )

        _tracker.update("Coarse → Fine", 0.0, "Generating fine audio tokens...")
        fine_tokens = generate_fine(
            coarse_tokens,
            history_prompt=history_prompt,
            temp=0.5,
        )

        _tracker.update("Decoding", 0.0, "Decoding audio waveform...")
        audio_arr = codec_decode(fine_tokens)

        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        _tracker.update("Done", 1.0, "Generation complete!")
        return audio_arr, full_generation
    finally:
        tqdm_module.tqdm = original_tqdm


def generate_long_with_progress(
    text: str,
    history_prompt,
    text_temp: float,
    waveform_temp: float,
):
    """Generate long-form audio by splitting text into sentences."""
    sentences = split_text_into_sentences(text)
    if not sentences:
        raise ValueError("No valid text to generate.")

    _tracker.log(f"Long-form generation: {len(sentences)} sentence(s)")
    all_audio = []
    prev_full_gen = None

    for i, sentence in enumerate(sentences):
        _tracker.log(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:60]}...")

        # Use previous generation as history for continuity
        hp = prev_full_gen if prev_full_gen is not None else history_prompt

        import tqdm as tqdm_module
        original_tqdm = tqdm_module.tqdm
        tqdm_module.tqdm = _patched_tqdm

        try:
            _tracker.update(
                f"Sentence {i+1}/{len(sentences)}",
                i / len(sentences),
                f"Text → Semantic: {sentence[:40]}...",
            )
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=hp,
                temp=text_temp,
                silent=True,
                use_kv_caching=True,
            )

            _tracker.update(
                f"Sentence {i+1}/{len(sentences)}",
                i / len(sentences),
                "Semantic → Coarse...",
            )
            coarse_tokens = generate_coarse(
                semantic_tokens,
                history_prompt=hp,
                temp=waveform_temp,
                silent=True,
                use_kv_caching=True,
            )

            _tracker.update(
                f"Sentence {i+1}/{len(sentences)}",
                i / len(sentences),
                "Coarse → Fine...",
            )
            fine_tokens = generate_fine(
                coarse_tokens,
                history_prompt=hp,
                temp=0.5,
            )

            audio_arr = codec_decode(fine_tokens)
            all_audio.append(audio_arr)

            # Save full generation for next sentence's history
            prev_full_gen = {
                "semantic_prompt": semantic_tokens,
                "coarse_prompt": coarse_tokens,
                "fine_prompt": fine_tokens,
            }
        finally:
            tqdm_module.tqdm = original_tqdm

    _tracker.update("Done", 1.0, "Long-form generation complete!")
    # Concatenate with small silence gap between sentences
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # 250ms silence
    combined = np.concatenate([
        np.concatenate([audio, silence]) for audio in all_audio
    ])
    return combined, prev_full_gen


def resolve_history_prompt(speaker_preset, custom_npz_file):
    """Determine the history prompt selected in the UI."""
    if custom_npz_file is not None:
        history_prompt = custom_npz_file.name if hasattr(custom_npz_file, "name") else custom_npz_file
        return history_prompt, "Custom .npz"

    if not speaker_preset or speaker_preset == DEFAULT_SPEAKER:
        return None, "Default"

    valid_values = {
        opt[1]
        for opt in SPEAKER_OPTIONS
        if isinstance(opt, tuple)
    }
    if speaker_preset in valid_values:
        return speaker_preset, speaker_preset

    for opt in SPEAKER_OPTIONS:
        if isinstance(opt, tuple) and opt[0] == speaker_preset:
            return opt[1], opt[1]

    return None, "Default"


# ── Gradio callbacks ─────────────────────────────────────────────────────────

def on_generate(
    lang,
    text,
    speaker_preset,
    custom_npz_file,
    text_temp,
    waveform_temp,
    use_long_form,
    output_filename,
    save_full_prompt,
    progress=gr.Progress(track_tqdm=False),
):
    """Main generation callback."""
    if not text or not text.strip():
        gr.Warning(t(lang, "empty_text"))
        return None, _tracker.get_log(), _history.get_table()

    history_prompt, speaker_label = resolve_history_prompt(speaker_preset, custom_npz_file)

    _tracker.reset()
    _tracker.log(f"Generating audio for: {text[:80]}...")
    _tracker.log(f"Speaker: {speaker_label}, text_temp={text_temp}, waveform_temp={waveform_temp}")
    _tracker.log(t(lang, "model_download_log"))
    progress(0, desc=t(lang, "loading_models"))

    try:
        if use_long_form:
            audio_arr, full_generation = generate_long_with_progress(
                text, history_prompt, text_temp, waveform_temp
            )
        else:
            audio_arr, full_generation = generate_with_progress(
                text, history_prompt, text_temp, waveform_temp
            )

        if not output_filename:
            output_filename = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        if not output_filename.endswith(".wav"):
            output_filename += ".wav"

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        write_wav(output_path, SAMPLE_RATE, audio_arr)

        if save_full_prompt:
            prompt_path = output_path.replace(".wav", ".npz")
            if full_generation is not None:
                save_as_prompt(prompt_path, full_generation)
                _tracker.log(f"{t(lang, 'prompt_saved')}: {prompt_path}")
            else:
                _tracker.log("Voice preset was not saved because full generation data is unavailable.")

        duration_s = len(audio_arr) / SAMPLE_RATE
        _history.add(output_path, text, speaker_label, duration_s)
        _tracker.log(f"{t(lang, 'audio_saved')}: {output_path}")

        progress(1.0, desc=t(lang, "done"))
        return (SAMPLE_RATE, audio_arr), _tracker.get_log(), _history.get_table()

    except Exception as e:
        _tracker.log(f"Error: {str(e)}")
        logger.exception("Generation failed")
        raise gr.Error(f"{t(lang, 'generation_failed')}: {str(e)}")


def on_preload_models(lang, use_small, cpu_offload, enable_mps, progress=gr.Progress()):
    """Preload models with settings."""
    progress(0, desc=t(lang, "loading_models"))
    yield t(lang, "models_prepare")

    try:
        # Set environment variables
        os.environ["SUNO_USE_SMALL_MODELS"] = str(use_small)
        os.environ["SUNO_OFFLOAD_CPU"] = str(cpu_offload)
        os.environ["SUNO_ENABLE_MPS"] = str(enable_mps)

        # Reload module-level settings
        import bark.generation as gen
        gen.USE_SMALL_MODELS = use_small
        gen.OFFLOAD_CPU = cpu_offload
        gen.GLOBAL_ENABLE_MPS = enable_mps

        preload_models(
            text_use_small=use_small,
            coarse_use_small=use_small,
            fine_use_small=use_small,
        )
        progress(1.0, desc=t(lang, "done"))
        gr.Info(t(lang, "models_loaded"))
        yield t(lang, "models_loaded")
    except Exception as e:
        logger.exception("Model loading failed")
        yield f"{t(lang, 'models_failed')}: {str(e)}"
        raise gr.Error(f"{t(lang, 'models_failed')}: {str(e)}")


def on_history_select(evt: gr.SelectData, history_data):
    """Handle history table selection - return selected audio."""
    idx = evt.index
    row_idx = idx[0] if isinstance(idx, (list, tuple)) else idx
    audio_path = _history.get_audio_by_row(row_idx)
    if audio_path and os.path.exists(audio_path):
        import scipy.io.wavfile as wavfile
        sr, data = wavfile.read(audio_path)
        return (sr, data)
    return None


def on_get_speaker_info(lang, speaker_preset):
    """Show info about the selected speaker."""
    if not speaker_preset or speaker_preset == DEFAULT_SPEAKER:
        return t(lang, "speaker_none")
    for opt in SPEAKER_OPTIONS:
        if isinstance(opt, tuple) and speaker_preset in opt:
            value = opt[1]
            lang_code = value.split("/")[-1].split("_")[0] if "/" in value else value.split("_")[0]
            lang_name = LANG_MAP.get(lang_code, "Unknown")
            ver = "v2" if "v2/" in value else "v1"
            if lang == "zh":
                note = "v2 是推荐版本。" if ver == "v2" else "v2 预设通常会有更稳定的音色质量。"
                return (
                    f"**语言：** {lang_name} ({lang_code})\n"
                    f"**版本：** {ver}\n"
                    f"**预设 ID：** `{value}`\n\n"
                    f"{note}"
                )
            note = "v2 is the recommended version." if ver == "v2" else "v2 presets generally produce more consistent voice quality."
            return (
                f"**Language:** {lang_name} ({lang_code})\n"
                f"**Version:** {ver}\n"
                f"**Preset ID:** `{value}`\n\n"
                f"{note}"
            )
    return "Speaker info unavailable." if lang == "en" else "音色信息不可用。"


def on_language_change(lang, speaker_preset):
    """Update visible UI copy when switching languages."""
    return [
        gr.update(value=render_hero_copy(lang)),
        gr.update(label=t(lang, "text_label"), placeholder=t(lang, "text_placeholder")),
        gr.update(value=t(lang, "tip")),
        gr.update(label=t(lang, "speaker_label")),
        gr.update(value=on_get_speaker_info(lang, speaker_preset)),
        gr.update(label=t(lang, "custom_voice_label")),
        gr.update(label=t(lang, "text_temp_label"), info=t(lang, "text_temp_info")),
        gr.update(value=t(lang, "text_temp_reset")),
        gr.update(label=t(lang, "waveform_temp_label"), info=t(lang, "waveform_temp_info")),
        gr.update(value=t(lang, "waveform_temp_reset")),
        gr.update(label=t(lang, "long_form_label"), info=t(lang, "long_form_info")),
        gr.update(label=t(lang, "output_filename_label"), placeholder=t(lang, "output_filename_placeholder")),
        gr.update(label=t(lang, "save_prompt_label"), info=t(lang, "save_prompt_info")),
        gr.update(value=t(lang, "generate_btn")),
        gr.update(label=t(lang, "audio_label")),
        gr.update(label=t(lang, "log_label")),
        gr.update(value=t(lang, "history_hint")),
        gr.update(headers=t(lang, "history_headers")),
        gr.update(value=t(lang, "model_notice")),
        gr.update(value=t(lang, "model_settings_hint")),
        gr.update(label=t(lang, "use_small_label"), info=t(lang, "use_small_info")),
        gr.update(label=t(lang, "cpu_offload_label"), info=t(lang, "cpu_offload_info")),
        gr.update(label=t(lang, "mps_label"), info=t(lang, "mps_info")),
        gr.update(value=t(lang, "load_models_btn")),
        gr.update(label=t(lang, "model_status_label"), placeholder=t(lang, "model_status_placeholder")),
        gr.update(value=f"### {t(lang, 'panel_input')}"),
        gr.update(value=f"### {t(lang, 'panel_output')}"),
        gr.update(value=f"### {t(lang, 'panel_log')}"),
        gr.update(value=f"### {t(lang, 'panel_history')}"),
        gr.update(value=f"### {t(lang, 'panel_model')}"),
    ]


# ── Build Gradio UI ──────────────────────────────────────────────────────────

def create_app():
    with gr.Blocks(
        title=f"{APP_NAME} - Text to Audio",
        theme=gr.themes.Soft(),
        css=CYBER_CSS,
    ) as app:
        with gr.Row(elem_classes=["cyber-hero"]):
            with gr.Column(elem_classes=["cyber-hero__content"]):
                hero_markdown = gr.Markdown(render_hero_copy("en"), elem_classes=["cyber-hero-wrap"])
                language = gr.Radio(
                    choices=[("EN", "en"), ("CN", "zh")],
                    value="en",
                    label="",
                    show_label=False,
                    elem_classes=["cyber-language"],
                )
            gr.Markdown(render_hero_orb(), elem_classes=["cyber-hero-orb"])

        with gr.Row(elem_classes=["cyber-main-grid"]):
            # ── Left column: Input ───────────────────────────────────────
            with gr.Column(scale=3, elem_classes=["cyber-panel"]):
                input_title = gr.Markdown(f"### {t('en', 'panel_input')}", elem_classes=["cyber-section-title"])
                text_input = gr.Textbox(
                    label=t("en", "text_label"),
                    placeholder=t("en", "text_placeholder"),
                    lines=5,
                    max_lines=20,
                    elem_classes=["cyber-input"],
                )
                tip_markdown = gr.Markdown(t("en", "tip"), elem_classes=["cyber-tip"])

                with gr.Row():
                    speaker_dropdown = gr.Dropdown(
                        choices=SPEAKER_OPTIONS,
                        value=DEFAULT_SPEAKER,
                        label=t("en", "speaker_label"),
                        interactive=True,
                        scale=3,
                        elem_classes=["cyber-input"],
                    )

                speaker_info = gr.Markdown(
                    value=t("en", "speaker_none"),
                    elem_classes=["cyber-tip"],
                )

                custom_npz = gr.File(
                    label=t("en", "custom_voice_label"),
                    file_types=[".npz"],
                    type="filepath",
                    elem_classes=["cyber-input"],
                )

                with gr.Accordion("Generation Parameters / 生成参数", open=True):
                    with gr.Row():
                        with gr.Column(elem_classes=["cyber-param-group"]):
                            text_temp = gr.Slider(
                                minimum=0.0,
                                maximum=1.5,
                                value=0.7,
                                step=0.05,
                                label=t("en", "text_temp_label"),
                                info=t("en", "text_temp_info"),
                                elem_classes=["cyber-input"],
                            )
                            reset_text_temp = gr.Button(
                                t("en", "text_temp_reset"),
                                size="sm",
                                elem_classes=["cyber-reset-button"],
                            )
                        with gr.Column(elem_classes=["cyber-param-group"]):
                            waveform_temp = gr.Slider(
                                minimum=0.0,
                                maximum=1.5,
                                value=0.7,
                                step=0.05,
                                label=t("en", "waveform_temp_label"),
                                info=t("en", "waveform_temp_info"),
                                elem_classes=["cyber-input"],
                            )
                            reset_waveform_temp = gr.Button(
                                t("en", "waveform_temp_reset"),
                                size="sm",
                                elem_classes=["cyber-reset-button"],
                            )

                    with gr.Row():
                        use_long_form = gr.Checkbox(
                            label=t("en", "long_form_label"),
                            value=False,
                            info=t("en", "long_form_info"),
                            elem_classes=["cyber-input"],
                        )

                with gr.Accordion("Output Options / 输出选项", open=False):
                    with gr.Row():
                        output_filename = gr.Textbox(
                            label=t("en", "output_filename_label"),
                            placeholder=t("en", "output_filename_placeholder"),
                            scale=2,
                            elem_classes=["cyber-input"],
                        )
                        save_full_prompt = gr.Checkbox(
                            label=t("en", "save_prompt_label"),
                            value=False,
                            info=t("en", "save_prompt_info"),
                            elem_classes=["cyber-input"],
                        )

                generate_btn = gr.Button(
                    t("en", "generate_btn"),
                    variant="primary",
                    size="lg",
                    elem_classes=["cyber-primary"],
                )

            # ── Right column: Output ─────────────────────────────────────
            with gr.Column(scale=2, elem_classes=["cyber-side-stack"]):
                with gr.Column(elem_classes=["cyber-hud", "cyber-output"]):
                    output_title = gr.Markdown(f"### {t('en', 'panel_output')}", elem_classes=["cyber-section-title"])
                    audio_output = gr.Audio(
                        label=t("en", "audio_label"),
                        type="numpy",
                        interactive=False,
                        elem_classes=["cyber-output"],
                    )

                with gr.Column(elem_classes=["cyber-terminal", "cyber-log"]):
                    log_title = gr.Markdown(f"### {t('en', 'panel_log')}", elem_classes=["cyber-section-title"])
                    log_output = gr.Textbox(
                        label=t("en", "log_label"),
                        lines=10,
                        max_lines=20,
                        interactive=False,
                        elem_classes=["log-box"],
                    )

        with gr.Row(elem_classes=["cyber-bottom-grid"]):
            # ── History section ──────────────────────────────────────────────
            with gr.Column(elem_classes=["cyber-terminal"]):
                history_title = gr.Markdown(f"### {t('en', 'panel_history')}", elem_classes=["cyber-section-title"])
                history_hint = gr.Markdown(t("en", "history_hint"), elem_classes=["cyber-inline-note"])
                history_table = gr.Dataframe(
                    headers=t("en", "history_headers"),
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    elem_classes=["cyber-table"],
                )

            # ── Model settings section ───────────────────────────────────────
            with gr.Column(elem_classes=["cyber-hud"]):
                model_title = gr.Markdown(f"### {t('en', 'panel_model')}", elem_classes=["cyber-section-title"])
                model_notice = gr.Markdown(t("en", "model_notice"), elem_classes=["cyber-notice"])
                model_settings_hint = gr.Markdown(t("en", "model_settings_hint"), elem_classes=["cyber-inline-note"])
                with gr.Row():
                    use_small_models = gr.Checkbox(
                        label=t("en", "use_small_label"),
                        value=False,
                        info=t("en", "use_small_info"),
                        elem_classes=["cyber-input"],
                    )
                    cpu_offload = gr.Checkbox(
                        label=t("en", "cpu_offload_label"),
                        value=False,
                        info=t("en", "cpu_offload_info"),
                        elem_classes=["cyber-input"],
                    )
                    enable_mps = gr.Checkbox(
                        label=t("en", "mps_label"),
                        value=False,
                        info=t("en", "mps_info"),
                        elem_classes=["cyber-input"],
                    )
                load_models_btn = gr.Button(
                    t("en", "load_models_btn"),
                    variant="secondary",
                    elem_classes=["cyber-secondary"],
                )
                model_status = gr.Textbox(
                    label=t("en", "model_status_label"),
                    interactive=False,
                    placeholder=t("en", "model_status_placeholder"),
                    elem_classes=["cyber-log"],
                )

        # ── Event bindings ───────────────────────────────────────────────

        generate_btn.click(
            fn=on_generate,
            inputs=[
                language,
                text_input,
                speaker_dropdown,
                custom_npz,
                text_temp,
                waveform_temp,
                use_long_form,
                output_filename,
                save_full_prompt,
            ],
            outputs=[audio_output, log_output, history_table],
        )

        speaker_dropdown.change(
            fn=on_get_speaker_info,
            inputs=[language, speaker_dropdown],
            outputs=[speaker_info],
        )

        load_models_btn.click(
            fn=on_preload_models,
            inputs=[language, use_small_models, cpu_offload, enable_mps],
            outputs=[model_status],
        )

        language.change(
            fn=on_language_change,
            inputs=[language, speaker_dropdown],
            outputs=[
                hero_markdown,
                text_input,
                tip_markdown,
                speaker_dropdown,
                speaker_info,
                custom_npz,
                text_temp,
                reset_text_temp,
                waveform_temp,
                reset_waveform_temp,
                use_long_form,
                output_filename,
                save_full_prompt,
                generate_btn,
                audio_output,
                log_output,
                history_hint,
                history_table,
                model_notice,
                model_settings_hint,
                use_small_models,
                cpu_offload,
                enable_mps,
                load_models_btn,
                model_status,
                input_title,
                output_title,
                log_title,
                history_title,
                model_title,
            ],
        )

        reset_text_temp.click(
            fn=lambda: 0.7,
            inputs=[],
            outputs=[text_temp],
        )

        reset_waveform_temp.click(
            fn=lambda: 0.7,
            inputs=[],
            outputs=[waveform_temp],
        )

        history_table.select(
            fn=on_history_select,
            inputs=[history_table],
            outputs=[audio_output],
        )

    return app


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vioce - Text to Audio Web Interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server hostname")
    args = parser.parse_args()

    app = create_app()
    try:
        app.launch(
            server_name=args.server_name,
            server_port=args.port,
            share=args.share,
            inbrowser=True,
        )
    except (OSError, ValueError):
        # Fallback: port occupied or localhost not accessible — try share link
        print("Local launch failed, creating share link...")
        app.launch(
            server_name=args.server_name,
            server_port=args.port,
            share=True,
            inbrowser=True,
        )
