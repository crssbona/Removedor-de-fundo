# -*- coding: utf-8 -*-
"""
Setup automatico das Ferramentas de Midia.

Instala as dependencias de Python e baixa os modelos/ffmpeg necessarios.
E' IDEMPOTENTE: pode rodar quantas vezes quiser — o que ja estiver instalado
ou baixado e' pulado, sem dar erro. Rodado automaticamente pelo Iniciar.bat.

A PRIMEIRA execucao baixa varios GB (PyTorch + modelos) e demora.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(BASE, "bin")
MODELS = os.path.join(BASE, "models")
MARKER = os.path.join(BASE, ".setup_ok")
VERSION = "1"

FFMPEG_URL = "https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
MODEL_FILES = [
    ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
     "isnet-general-use.onnx", 100),
]
SAM2_MODELS = [
    ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
     "sam2.1_hiera_tiny.pt", 80),
    ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
     "sam2.1_hiera_base_plus.pt", 200),
]


def have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def pip(*args, env=None):
    return subprocess.call([sys.executable, "-m", "pip", "install", "--no-input", *args], env=env)


def has_nvidia():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


def dml_ok():
    try:
        import onnxruntime as o
        return "DmlExecutionProvider" in o.get_available_providers()
    except Exception:
        return False


def download(url, dest, min_mb=1):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_mb * 1024 * 1024:
        print("   ok (ja existe):", os.path.basename(dest))
        return True
    print("   baixando:", os.path.basename(dest), "...")
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
        return True
    except Exception as e:
        print("   FALHOU:", str(e)[:100])
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        return False


def ensure_ffmpeg():
    exe = os.path.join(BIN, "ffmpeg.exe")
    if os.path.exists(exe):
        print("   ok (ja existe): ffmpeg")
        return
    import tempfile
    import zipfile
    os.makedirs(BIN, exist_ok=True)
    zp = os.path.join(tempfile.gettempdir(), "ffmpeg_dl.zip")
    print("   baixando: ffmpeg (~40 MB)...")
    try:
        urllib.request.urlretrieve(FFMPEG_URL, zp)
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                b = os.path.basename(n)
                if b in ("ffmpeg.exe", "ffprobe.exe"):
                    with z.open(n) as src, open(os.path.join(BIN, b), "wb") as dst:
                        shutil.copyfileobj(src, dst)
        os.remove(zp)
    except Exception as e:
        print("   FALHOU o ffmpeg:", str(e)[:100])


def prewarm():
    """Baixa (best-effort) os modelos que se auto-baixam no 1o uso."""
    def _insight():
        if not have("insightface"):
            return
        from insightface.app import FaceAnalysis
        a = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        a.prepare(ctx_id=-1, det_size=(640, 640))

    def _mms():
        if not have("torchaudio"):
            return
        import torchaudio
        torchaudio.pipelines.MMS_FA.get_model()

    def _demucs():
        if not have("demucs"):
            return
        from demucs.pretrained import get_model
        get_model("htdemucs")

    def _matany():
        if not have("matanyone"):
            return
        from matanyone import InferenceCore
        InferenceCore("PeiqingYang/MatAnyone")

    for name, fn in [("rostos (InsightFace)", _insight), ("legenda (MMS)", _mms),
                     ("voz (Demucs)", _demucs), ("matte (MatAnyone)", _matany)]:
        try:
            print("   -", name, "...")
            fn()
        except Exception as e:
            print("     (pulado por ora:", str(e)[:80], ")")


def main():
    # ja configurado nesta versao? sai rapido.
    if os.path.exists(MARKER):
        try:
            if open(MARKER, encoding="utf-8").read().strip() == VERSION:
                return 0
        except Exception:
            pass

    print("=" * 62)
    print("  Preparando as Ferramentas de Midia")
    print("  (a PRIMEIRA vez baixa varios GB e pode demorar bastante)")
    print("=" * 62)

    nv = has_nvidia()
    print("GPU NVIDIA:", "sim (CUDA)" if nv else "nao (vai usar CPU / DirectML)")
    torch_idx = ["--index-url", "https://download.pytorch.org/whl/cu126"] if nv else []

    # 1) PyTorch (torch + torchvision + torchaudio)
    if not (have("torch") and have("torchvision") and have("torchaudio")):
        print("\n[1/6] PyTorch...")
        if pip("torch", "torchvision", "torchaudio", *torch_idx) != 0:
            print("!! Falha ao instalar o PyTorch."); return 1
    else:
        print("\n[1/6] PyTorch ja instalado.")

    # 2) bibliotecas principais
    main_mods = ("yt_dlp", "cv2", "numpy", "PIL", "scipy", "einops", "av",
                 "hickle", "gdown", "huggingface_hub", "onnx", "demucs", "omegaconf", "imageio")
    if not all(have(m) for m in main_mods):
        print("\n[2/6] Bibliotecas principais...")
        pip("yt-dlp", "numpy", "Pillow", "opencv-python", "scipy", "einops",
            "hydra-core", "omegaconf", "tqdm", "imageio", "av", "hickle", "gdown",
            "huggingface_hub", "safetensors", "onnx", "demucs")
    else:
        print("\n[2/6] Bibliotecas ja instaladas.")

    # 3) InsightFace (rostos) + garantir o onnxruntime-directml (GPU)
    print("\n[3/6] Reconhecimento facial + GPU (DirectML)...")
    if not have("insightface"):
        pip("insightface")
    if not dml_ok():
        # o insightface puxa o onnxruntime comum; troco pelo directml (usa a GPU)
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"])
        pip("--force-reinstall", "--no-deps", "onnxruntime-directml")

    # 4) SAM 2 + MatAnyone (precisam do git)
    print("\n[4/6] SAM 2 / MatAnyone (video 'preciso' e 'matte')...")
    git_ok = shutil.which("git") is not None
    if git_ok:
        if not have("sam2"):
            env = dict(os.environ, SAM2_BUILD_CUDA="0", SAM2_BUILD_ALLOW_ERRORS="1")
            pip("git+https://github.com/facebookresearch/sam2.git", env=env)
        if not have("matanyone"):
            pip("--no-deps", "git+https://github.com/pq-yang/MatAnyone")
            pip("git+https://github.com/cheind/py-thin-plate-spline")
    else:
        print("   AVISO: 'git' nao encontrado. Os modos 'Preciso' e 'Matte' do video")
        print("          nao serao instalados. Instale o Git (https://git-scm.com) e")
        print("          apague o arquivo .setup_ok para rodar de novo, se quiser.")

    # 5) ffmpeg + modelos em arquivo
    print("\n[5/6] ffmpeg e modelos...")
    ensure_ffmpeg()
    os.makedirs(MODELS, exist_ok=True)
    for url, name, mb in MODEL_FILES:
        download(url, os.path.join(MODELS, name), mb)
    if git_ok:
        for url, name, mb in SAM2_MODELS:
            download(url, os.path.join(MODELS, name), mb)

    # 6) baixar os modelos que se auto-baixam (best-effort)
    print("\n[6/6] Baixando modelos de IA (rostos, legenda, voz, matte)...")
    prewarm()

    try:
        with open(MARKER, "w", encoding="utf-8") as f:
            f.write(VERSION)
    except Exception:
        pass
    print("\n" + "=" * 62)
    print("  Tudo pronto! Iniciando...")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
