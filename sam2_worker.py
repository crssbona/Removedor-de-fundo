# -*- coding: utf-8 -*-
"""
Worker do MODO PRECISO (SAM 2).

Recebe um video + pontos de clique (marcando 1 ou mais objetos no 1o quadro),
rastreia cada objeto pelo video inteiro com o SAM 2 e gera a saida com o fundo
removido (mesmos formatos do modo automatico: mp4green / webm / mov).

Roda como processo separado (evita carregar o PyTorch dentro do servidor):

    python sam2_worker.py <job_dir> <fmt> <chroma> <audio:0|1>

Le:   <job_dir>/input.*  e  <job_dir>/points.json
Gera: <job_dir>/<nome>-sem-fundo.<ext>
Imprime no stdout (o servidor relê e repassa como SSE):
    PROG <i> <total>
    DONE <nome_do_arquivo>
    ERR  <mensagem>
"""

import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import vmatte  # reaproveita _encoder_cmd / CHROMA / probe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# modelos SAM 2 suportados: chave -> (checkpoint, config)
MODELS = {
    "tiny":      ("sam2.1_hiera_tiny.pt",      "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
}


def model_paths(key):
    fname, cfg = MODELS.get(key, MODELS["tiny"])
    return os.path.join(MODELS_DIR, fname), cfg


def available_models():
    return [k for k, (f, _c) in MODELS.items()
            if os.path.exists(os.path.join(MODELS_DIR, f))]


def log(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# --- refino de borda: guided filter (He et al.) em numpy, O(1) por pixel ----
def _boxfilter(x, r):
    h, w = x.shape
    dst = np.zeros_like(x)
    cum = np.cumsum(x, axis=0)
    dst[0:r + 1, :] = cum[r:2 * r + 1, :]
    dst[r + 1:h - r, :] = cum[2 * r + 1:h, :] - cum[0:h - 2 * r - 1, :]
    dst[h - r:h, :] = np.tile(cum[h - 1:h, :], (r, 1)) - cum[h - 2 * r - 1:h - r - 1, :]
    cum = np.cumsum(dst, axis=1)
    dst[:, 0:r + 1] = cum[:, r:2 * r + 1]
    dst[:, r + 1:w - r] = cum[:, 2 * r + 1:w] - cum[:, 0:w - 2 * r - 1]
    dst[:, w - r:w] = np.tile(cum[:, w - 1:w], (1, r)) - cum[:, w - 2 * r - 1:w - r - 1]
    return dst


def _guided_filter(guide, src, r, eps):
    N = _boxfilter(np.ones_like(guide), r)
    mean_I = _boxfilter(guide, r) / N
    mean_p = _boxfilter(src, r) / N
    mean_Ip = _boxfilter(guide * src, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = _boxfilter(guide * guide, r) / N - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return (_boxfilter(a, r) / N) * guide + (_boxfilter(b, r) / N)


def _refine_alpha(frame_rgb, alpha, r=4, eps=1e-4):
    """Suaviza e alinha a borda do alpha ao contorno da imagem (anti-serrilhado)."""
    h, w = alpha.shape
    r = min(r, (min(h, w) - 1) // 2)
    if r < 1:
        return np.clip(alpha, 0.0, 1.0)
    gray = frame_rgb.mean(axis=2) / 255.0
    q = _guided_filter(gray.astype(np.float32), alpha.astype(np.float32), r, eps)
    return np.clip(q, 0.0, 1.0)


def sam2_available():
    if not available_models():
        return False, "nenhum checkpoint do SAM2 encontrado em models/"
    try:
        import torch  # noqa
        import sam2   # noqa
    except Exception as e:
        return False, "dependencias do SAM2 ausentes: %s" % e
    return True, None


def _extract_frames(ffmpeg, input_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", input_path, "-q:v", "2",
         os.path.join(frames_dir, "%05d.jpg")],
        check=True, creationflags=_NO_WINDOW,
    )
    return sorted(n for n in os.listdir(frames_dir) if n.endswith(".jpg"))


def run(job_dir, fmt, chroma, keep_audio, model="tiny"):
    ok, err = sam2_available()
    if not ok:
        log("ERR " + err)
        return 1

    ckpt, cfg = model_paths(model)
    if not os.path.exists(ckpt):  # cai pro tiny se o escolhido nao existir
        ckpt, cfg = model_paths("tiny")

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    ffmpeg = vmatte._bin("ffmpeg") if hasattr(vmatte, "_bin") else None
    ffmpeg = ffmpeg or os.path.join(BASE_DIR, "bin", "ffmpeg.exe")
    ffprobe = os.path.join(BASE_DIR, "bin", "ffprobe.exe")

    inp = next((os.path.join(job_dir, n) for n in os.listdir(job_dir)
                if n.startswith("input.")), None)
    if not inp:
        log("ERR video de entrada nao encontrado")
        return 1

    with open(os.path.join(job_dir, "points.json"), encoding="utf-8") as f:
        objects = json.load(f).get("objects", [])
    objects = [o for o in objects if o]
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
    frames_dir = os.path.join(job_dir, "frames")
    names = _extract_frames(ffmpeg, inp, frames_dir)
    nframes = len(names)
    if total <= 0:
        total = nframes
    log("MSG Carregando modelo e analisando o video...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda" else _NullCtx())
    with torch.inference_mode(), autocast:
        # offload p/ caber nos 6 GB da GTX 1660
        state = predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        for obj_id, clicks in enumerate(objects):
            pts = np.array([[c["x"], c["y"]] for c in clicks], dtype=np.float32)
            lbls = np.array([1 if c.get("pos", True) else 0 for c in clicks], dtype=np.int32)
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=obj_id, points=pts, labels=lbls,
            )

        # codificador de saida (reaproveita o do modo automatico)
        rgba = fmt in ("webm", "mov")
        bg = np.array(vmatte.CHROMA.get(chroma, vmatte.CHROMA["green"]), dtype=np.float32)
        enc = subprocess.Popen(
            vmatte._encoder_cmd(ffmpeg, fmt, w, h, fps, inp, keep_audio, has_audio, out_path),
            stdin=subprocess.PIPE, creationflags=_NO_WINDOW,
        )

        try:
            done = 0
            log("MSG Removendo o fundo...")
            for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
                # alpha SUAVE (sigmoid) e uniao dos objetos pelo maximo
                prob = torch.sigmoid(mask_logits.float())          # (n,1,H,W) em [0,1]
                alpha = prob.amax(dim=0).squeeze(0).cpu().numpy().astype(np.float32)
                if alpha.shape != (h, w):
                    alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8))
                                       .resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
                frame = np.asarray(Image.open(os.path.join(frames_dir, names[frame_idx]))
                                   .convert("RGB")).astype(np.float32)
                # refina a borda alinhando ao contorno da imagem (anti-serrilhado)
                alpha = _refine_alpha(frame, alpha)
                a = alpha[..., None]
                if rgba:
                    out = np.empty((h, w, 4), np.uint8)
                    out[..., :3] = frame.astype(np.uint8)
                    out[..., 3] = (alpha * 255).astype(np.uint8)
                else:
                    out = (frame * a + bg * (1.0 - a)).clip(0, 255).astype(np.uint8)
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


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    if len(sys.argv) < 5:
        log("ERR uso: sam2_worker.py <job_dir> <fmt> <chroma> <audio> [modelo]")
        sys.exit(2)
    jd, fmt, chroma, audio = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    model = sys.argv[5] if len(sys.argv) > 5 else "tiny"
    try:
        sys.exit(run(jd, fmt, chroma, audio != "0", model))
    except Exception as e:
        log("ERR " + str(e))
        sys.exit(1)
