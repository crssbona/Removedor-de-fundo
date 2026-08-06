# -*- coding: utf-8 -*-
"""
Remocao de fundo de VIDEO, quadro a quadro, com IS-Net (modelo de segmentacao
de objeto saliente) rodando em onnxruntime / DirectML (GPU).

Pipeline:
  ffmpeg (decodifica em quadros rgb24)  ->  IS-Net gera a mascara (alpha)
  ->  composicao (fundo verde OU alpha transparente)  ->  ffmpeg (recodifica)

Saidas:
  - "mp4green": MP4 com o objeto sobre fundo de cor solida (chroma key p/ CapCut)
  - "webm":     WebM (VP9) com canal alpha de verdade (transparente)
  - "mov":      MOV ProRes 4444 com alpha (edicao profissional)

O audio original e remuxado de volta (opcional).
"""

import json
import os
import subprocess
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "isnet-general-use.onnx")
MODEL_SIZE = 1024  # IS-Net espera entrada 1024x1024

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

CHROMA = {
    "green":   (0, 177, 64),
    "blue":    (0, 71, 187),
    "magenta": (192, 38, 211),
}

_session = None
_session_lock = threading.Lock()


class AbortError(Exception):
    """Levantada para interromper o processamento (ex.: cliente desconectou)."""


def model_available():
    return os.path.exists(MODEL_PATH)


def gpu_available():
    try:
        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def get_session():
    global _session
    with _session_lock:
        if _session is None:
            providers = []
            if "DmlExecutionProvider" in ort.get_available_providers():
                providers.append("DmlExecutionProvider")
            providers.append("CPUExecutionProvider")
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)
        return _session


def using_gpu():
    try:
        return "DmlExecutionProvider" in get_session().get_providers()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Inferencia da mascara
# ---------------------------------------------------------------------------
def _preprocess(frame_rgb):
    im = Image.fromarray(frame_rgb).resize((MODEL_SIZE, MODEL_SIZE), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = a - 0.5                      # IS-Net: mean 0.5, std 1.0
    a = a.transpose(2, 0, 1)[None]   # (1, 3, H, W)
    return np.ascontiguousarray(a)


def _infer_mask(sess, in_name, frame_rgb, out_w, out_h):
    out = sess.run(None, {in_name: _preprocess(frame_rgb)})[0]
    pred = out[0]
    if pred.ndim == 3:
        pred = pred[0]
    mi, ma = float(pred.min()), float(pred.max())
    pred = (pred - mi) / (ma - mi + 1e-8)
    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR)
    return np.asarray(mask, dtype=np.float32) / 255.0  # (H, W) em [0,1]


# ---------------------------------------------------------------------------
# Sondagem do video
# ---------------------------------------------------------------------------
def probe(path, ffprobe):
    out = subprocess.run(
        [ffprobe, "-v", "error",
         "-show_entries", "stream=index,codec_type,width,height,r_frame_rate,nb_frames",
         "-show_entries", "format=duration",
         "-of", "json", path],
        capture_output=True, text=True, creationflags=_NO_WINDOW,
    )
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        raise ValueError("O arquivo nao tem faixa de video.")
    w, h = int(v["width"]), int(v["height"])
    num, den = (v.get("r_frame_rate") or "30/1").split("/")
    fps = (float(num) / float(den)) if float(den) else 30.0
    nb = v.get("nb_frames")
    if nb and str(nb).isdigit() and int(nb) > 0:
        total = int(nb)
    else:
        dur = float(data.get("format", {}).get("duration", 0) or 0)
        total = int(round(dur * fps)) if dur else 0
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return {"w": w, "h": h, "fps": fps, "total": total, "audio": has_audio}


def _encoder_cmd(ffmpeg, fmt, w, h, fps, input_path, keep_audio, has_audio, out_path):
    pix_in = "rgb24" if fmt == "mp4green" else "rgba"
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", pix_in, "-s", "%dx%d" % (w, h),
        "-r", "%.6f" % fps, "-i", "pipe:0",
    ]
    mux_audio = keep_audio and has_audio
    if mux_audio:
        cmd += ["-i", input_path, "-map", "0:v", "-map", "1:a:0?"]

    if fmt == "webm":
        cmd += ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24",
                "-auto-alt-ref", "0"]
        if mux_audio:
            cmd += ["-c:a", "libopus"]
    elif fmt == "mov":
        cmd += ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
        if mux_audio:
            cmd += ["-c:a", "pcm_s16le"]
    else:  # mp4green
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"]
        if mux_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]

    if mux_audio:
        cmd += ["-shortest"]
    cmd += [out_path]
    return cmd


# ---------------------------------------------------------------------------
# Processamento principal
# ---------------------------------------------------------------------------
def process(input_path, out_path, fmt, chroma, keep_audio,
            ffmpeg, ffprobe, on_progress=None, should_abort=None):
    """
    fmt: 'mp4green' | 'webm' | 'mov'
    chroma: 'green' | 'blue' | 'magenta' (so usado em mp4green)
    on_progress(i, total): chamado a cada quadro (pode levantar AbortError)
    should_abort(): se retornar True, interrompe
    Retorna o numero de quadros processados.
    """
    info = probe(input_path, ffprobe)
    w, h, fps, total, has_audio = info["w"], info["h"], info["fps"], info["total"], info["audio"]
    rgba = fmt in ("webm", "mov")
    bg = np.array(CHROMA.get(chroma, CHROMA["green"]), dtype=np.float32)

    sess = get_session()
    in_name = sess.get_inputs()[0].name

    dec = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", input_path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, creationflags=_NO_WINDOW,
    )
    enc = subprocess.Popen(
        _encoder_cmd(ffmpeg, fmt, w, h, fps, input_path, keep_audio, has_audio, out_path),
        stdin=subprocess.PIPE, creationflags=_NO_WINDOW,
    )

    frame_bytes = w * h * 3
    i = 0
    try:
        while True:
            if should_abort and should_abort():
                raise AbortError()
            buf = dec.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            mask = _infer_mask(sess, in_name, frame, w, h)
            m = mask[..., None]
            if rgba:
                out = np.empty((h, w, 4), np.uint8)
                out[..., :3] = frame
                out[..., 3] = (mask * 255).astype(np.uint8)
            else:
                comp = frame.astype(np.float32) * m + bg * (1.0 - m)
                out = comp.clip(0, 255).astype(np.uint8)
            enc.stdin.write(out.tobytes())
            i += 1
            if on_progress:
                on_progress(i, total)
    finally:
        try:
            if enc.stdin:
                enc.stdin.close()
        except Exception:
            pass
        try:
            dec.stdout.close()
        except Exception:
            pass

    enc.wait()
    dec.wait()
    if enc.returncode not in (0, None):
        raise RuntimeError("Falha ao codificar o video de saida (ffmpeg).")
    return i
