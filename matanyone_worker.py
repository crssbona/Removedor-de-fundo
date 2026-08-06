# -*- coding: utf-8 -*-
"""
Worker do MODO MATTE (MatAnyone).

Fluxo:
  1) seus cliques no 1o quadro  ->  SAM 2 (tiny) gera a mascara-semente
  2) MatAnyone propaga como ALPHA SUAVE (matte) o video inteiro
  3) composicao (mp4green / webm / mov) + audio original

Roda como processo separado. Uso:
    python matanyone_worker.py <job_dir> <fmt> <chroma> <audio:0|1>

Le:   <job_dir>/input.*  e  <job_dir>/points.json
Gera: <job_dir>/<nome>-sem-fundo.<ext>
Imprime: MSG <texto> / PROG <i> <total> / DONE <arquivo> / ERR <msg>

Licenca do MatAnyone: NTU S-Lab 1.0 (uso de pesquisa / nao-comercial).
"""

import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import vmatte  # reaproveita _encoder_cmd / CHROMA / probe / _bin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAM2_CKPT = os.path.join(MODELS_DIR, "sam2.1_hiera_tiny.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
MATANYONE_ID = "PeiqingYang/MatAnyone"
MAX_MIN_SIDE = 720   # teto do lado menor no matte (protege a VRAM de 6 GB)
N_WARMUP = 10
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def log(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _bin(name):
    import shutil
    local = os.path.join(BASE_DIR, "bin", name + (".exe" if os.name == "nt" else ""))
    return local if os.path.exists(local) else (shutil.which(name) or local)


def available():
    import importlib.util
    if not os.path.exists(SAM2_CKPT):
        return False, "checkpoint do SAM 2 (tiny) ausente"
    for mod in ("torch", "sam2", "matanyone"):
        if importlib.util.find_spec(mod) is None:
            return False, "dependencia ausente: %s" % mod
    return True, None


def _extract_frames(ffmpeg, input_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", input_path, "-q:v", "2",
         os.path.join(frames_dir, "%05d.jpg")],
        check=True, creationflags=_NO_WINDOW,
    )
    return sorted(os.path.join(frames_dir, n) for n in os.listdir(frames_dir)
                  if n.endswith(".jpg"))


def _seed_mask(frame0_rgb, objects, device):
    """Gera a mascara-semente do 1o quadro a partir dos cliques, via SAM 2."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    predictor = SAM2ImagePredictor(sam)
    h, w = frame0_rgb.shape[:2]
    union = np.zeros((h, w), dtype=bool)
    with torch.inference_mode():
        predictor.set_image(frame0_rgb)
        for clicks in objects:
            pts = np.array([[c["x"], c["y"]] for c in clicks], dtype=np.float32)
            lbls = np.array([1 if c.get("pos", True) else 0 for c in clicks], dtype=np.int32)
            masks, _scores, _low = predictor.predict(
                point_coords=pts, point_labels=lbls, multimask_output=False)
            union |= (masks[0] > 0)
    del predictor, sam
    if device == "cuda":
        torch.cuda.empty_cache()
    return (union.astype(np.uint8) * 255)


def run(job_dir, fmt, chroma, keep_audio):
    ok, err = available()
    if not ok:
        log("ERR " + err)
        return 1

    import torch
    from matanyone import InferenceCore

    ffmpeg = _bin("ffmpeg")
    ffprobe = _bin("ffprobe")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inp = next((os.path.join(job_dir, n) for n in os.listdir(job_dir)
                if n.startswith("input.")), None)
    if not inp:
        log("ERR video de entrada nao encontrado")
        return 1
    with open(os.path.join(job_dir, "points.json"), encoding="utf-8") as f:
        objects = [o for o in json.load(f).get("objects", []) if o]
    if not objects:
        log("ERR nenhum objeto marcado")
        return 1

    base = "video"
    try:
        with open(os.path.join(job_dir, "name.txt"), encoding="utf-8") as f:
            base = f.read().strip() or "video"
    except Exception:
        pass
    out_ext = {"mp4green": ".mp4", "webm": ".webm", "mov": ".mov"}.get(fmt, ".mp4")
    out_name = "%s-sem-fundo%s" % (base, out_ext)
    out_path = os.path.join(job_dir, out_name)

    info = vmatte.probe(inp, ffprobe)
    w, h, fps, total, has_audio = info["w"], info["h"], info["fps"], info["total"], info["audio"]

    log("PROG 0 %d" % (total or 0))
    log("MSG Extraindo quadros do video...")
    names = _extract_frames(ffmpeg, inp, os.path.join(job_dir, "frames"))
    nframes = len(names)
    if total <= 0:
        total = nframes

    # resolucao de trabalho do matte (teto no lado menor)
    scale = 1.0
    if min(w, h) > MAX_MIN_SIDE:
        scale = MAX_MIN_SIDE / float(min(w, h))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))

    log("MSG Segmentando o 1o quadro (SAM 2)...")
    frame0 = np.asarray(Image.open(names[0]).convert("RGB"))
    seed_full = _seed_mask(frame0, objects, device)  # (h,w) 0/255
    if seed_full.max() == 0:
        log("ERR o SAM 2 nao encontrou objeto nos cliques")
        return 1
    seed = seed_full
    if scale != 1.0:
        seed = np.asarray(Image.fromarray(seed_full).resize((sw, sh), Image.NEAREST))

    log("MSG Carregando MatAnyone...")
    proc = InferenceCore(MATANYONE_ID)
    mask_t = torch.from_numpy(seed).float().to(device)

    rgba = fmt in ("webm", "mov")
    bg = np.array(vmatte.CHROMA.get(chroma, vmatte.CHROMA["green"]), dtype=np.float32)
    enc = subprocess.Popen(
        vmatte._encoder_cmd(ffmpeg, fmt, w, h, fps, inp, keep_audio, has_audio, out_path),
        stdin=subprocess.PIPE, creationflags=_NO_WINDOW,
    )

    seq = [0] * N_WARMUP + list(range(nframes))  # aquece repetindo o 1o quadro
    done = 0
    log("MSG Gerando o matte (removendo o fundo)...")
    try:
        with torch.inference_mode():
            for ti, fidx in enumerate(seq):
                arr = np.asarray(Image.open(names[fidx]).convert("RGB"))
                src = arr
                if scale != 1.0:
                    src = np.asarray(Image.fromarray(arr).resize((sw, sh), Image.BILINEAR))
                img = torch.from_numpy(src).permute(2, 0, 1).float().to(device) / 255.0

                if ti == 0:
                    proc.step(img, mask_t, objects=[1])
                    op = proc.step(img, first_frame_pred=True)
                elif ti <= N_WARMUP:
                    op = proc.step(img, first_frame_pred=True)
                else:
                    op = proc.step(img)

                if ti < N_WARMUP:
                    continue
                alpha = proc.output_prob_to_mask(op).cpu().numpy().astype(np.float32)  # (sh,sw) 0..1
                if scale != 1.0:
                    alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8))
                                       .resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
                a = alpha[..., None]
                if rgba:
                    out = np.empty((h, w, 4), np.uint8)
                    out[..., :3] = arr
                    out[..., 3] = (alpha * 255).astype(np.uint8)
                else:
                    out = (arr.astype(np.float32) * a + bg * (1.0 - a)).clip(0, 255).astype(np.uint8)
                enc.stdin.write(out.tobytes())
                done += 1
                log("PROG %d %d" % (done, total))
    finally:
        try:
            enc.stdin.close()
        except Exception:
            pass
    enc.wait()

    if enc.returncode not in (0, None):
        log("ERR falha ao codificar a saida (ffmpeg)")
        return 1
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        log("ERR arquivo de saida nao gerado")
        return 1
    log("DONE " + out_name)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 5:
        log("ERR uso: matanyone_worker.py <job_dir> <fmt> <chroma> <audio>")
        sys.exit(2)
    jd, fmt, chroma, audio = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    try:
        sys.exit(run(jd, fmt, chroma, audio != "0"))
    except Exception as e:
        log("ERR " + str(e))
        sys.exit(1)
