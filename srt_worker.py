# -*- coding: utf-8 -*-
"""
Worker de LEGENDA PALAVRA-POR-PALAVRA (música com letra).

Recebe um áudio (música) + a letra e devolve:
  - legenda.srt   (uma palavra por legenda, sincronizada)
  - words.json    (palavras + tempos, para o preview)

Pipeline (tudo local, GPU):
  ffmpeg carrega o áudio -> Demucs isola os VOCAIS -> alinhamento forçado
  (torchaudio MMS_FA) da letra na voz -> tempo de cada palavra -> SRT.

Uso:
    python srt_worker.py <job_dir>

Le:   <job_dir>/input.*   (áudio)  e  <job_dir>/lyrics.txt
Gera: <job_dir>/legenda.srt  e  <job_dir>/words.json
Imprime: MSG <txt> / DONE / ERR <msg>
"""

import json
import os
import re
import subprocess
import sys
import unicodedata

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    for m in ("torch", "torchaudio", "demucs"):
        if importlib.util.find_spec(m) is None:
            return False, "dependencia ausente: %s" % m
    return True, None


def _load_audio(ffmpeg, path, sr, ch):
    out = subprocess.run(
        [ffmpeg, "-v", "error", "-i", path, "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "pipe:1"],
        capture_output=True, creationflags=_NO_WINDOW,
    ).stdout
    a = np.frombuffer(out, np.float32)
    if ch > 1:
        a = a.reshape(-1, ch).T          # (ch, T)
    else:
        a = a.reshape(1, -1)             # (1, T)
    return a.copy()


def _norm_word(w):
    """minúsculas, sem acento, só a-z e apóstrofo (o alinhador MMS trabalha assim)."""
    w = unicodedata.normalize("NFKD", w.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return re.sub(r"[^a-z']", "", w)


def _srt_time(t):
    t = max(0.0, t)
    h = int(t // 3600); m = int(t % 3600 // 60); s = int(t % 60); ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s += 1; ms = 0
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def run(job_dir):
    ok, err = available()
    if not ok:
        log("ERR " + err)
        return 1

    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from torchaudio.pipelines import MMS_FA as bundle

    ffmpeg = _bin("ffmpeg")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    inp = next((os.path.join(job_dir, n) for n in os.listdir(job_dir)
                if n.startswith("input.")), None)
    if not inp:
        log("ERR áudio não encontrado")
        return 1
    try:
        with open(os.path.join(job_dir, "lyrics.txt"), encoding="utf-8") as f:
            lyrics = f.read()
    except Exception:
        log("ERR letra não encontrada")
        return 1

    # palavras: guardo a original (p/ exibir) e a normalizada (p/ alinhar)
    disp, norm = [], []
    for tok in lyrics.split():
        n = _norm_word(tok)
        if n:
            disp.append(tok.strip())
            norm.append(n)
    if not norm:
        log("ERR a letra não tem palavras válidas")
        return 1

    # --- separa os vocais (Demucs) ---
    log("MSG Separando os vocais (Demucs)... pode demorar")
    dmodel = get_model("htdemucs")
    dmodel.to(dev).eval()
    mix = torch.from_numpy(_load_audio(ffmpeg, inp, dmodel.samplerate, 2)).to(dev)
    ref = mix.mean(0)
    mix = (mix - ref.mean()) / (ref.std() + 1e-8)
    with torch.inference_mode():
        sources = apply_model(dmodel, mix[None], device=dev, split=True, overlap=0.25, progress=False)[0]
    sources = sources * ref.std() + ref.mean()
    vocals = sources[dmodel.sources.index("vocals")]        # (2, T) @ 44100
    del dmodel, mix, sources
    if dev == "cuda":
        torch.cuda.empty_cache()

    # vocais -> mono 16k (o que o MMS_FA espera). htdemucs = 44100 Hz.
    import torchaudio.functional as AF
    voc16 = AF.resample(vocals.mean(0, keepdim=True).cpu(), 44100, bundle.sample_rate)

    # --- alinhamento forçado ---
    log("MSG Alinhando a letra ao áudio...")
    amodel = bundle.get_model().to(dev)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    with torch.inference_mode():
        emission, _ = amodel(voc16.to(dev))
    try:
        spans = aligner(emission[0], tokenizer(norm))
    except Exception as e:
        log("ERR falha no alinhamento (a letra bate com o áudio?): %s" % e)
        return 1
    ratio = voc16.size(1) / emission.size(1) / bundle.sample_rate

    log("MSG Gerando a legenda...")
    words = []
    for sp, w in zip(spans, disp):
        words.append({"word": w, "start": round(sp[0].start * ratio, 3),
                      "end": round(sp[-1].end * ratio, 3)})

    # SRT: cada palavra vira uma legenda; fica na tela até a próxima começar
    lines = []
    for i, wd in enumerate(words):
        a = wd["start"]
        b = words[i + 1]["start"] if i + 1 < len(words) else wd["end"] + 0.4
        if b <= a:
            b = a + 0.15
        lines.append("%d\n%s --> %s\n%s\n" % (i + 1, _srt_time(a), _srt_time(b), wd["word"]))
    with open(os.path.join(job_dir, "legenda.srt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    jobname = os.path.basename(os.path.normpath(job_dir))
    with open(os.path.join(job_dir, "words.json"), "w", encoding="utf-8") as f:
        json.dump({"words": words, "audio": "/files/%s/%s" % (jobname, os.path.basename(inp)),
                   "srt": "/files/%s/legenda.srt" % jobname}, f, ensure_ascii=False)
    log("DONE")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log("ERR uso: srt_worker.py <job_dir>")
        sys.exit(2)
    try:
        sys.exit(run(sys.argv[1]))
    except Exception as e:
        log("ERR " + str(e))
        sys.exit(1)
