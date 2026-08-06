# -*- coding: utf-8 -*-
"""
Worker de RECONHECIMENTO FACIAL em filmes/episódios (live-action).

Recebe fotos de referencia de N personagens + UM OU VARIOS videos (episodios),
e retorna, por episodio, em que minutos cada personagem aparece — sozinho,
junto no mesmo QUADRO e junto na mesma CENA.

Motor: InsightFace (SCRFD + ArcFace), ONNX na GPU (DirectML). So funciona bem
em rostos humanos REAIS.

Uso:
    python face_worker.py <job_dir> <fps> <sens>

Le:   <job_dir>/chars.json        {"names": [...]}
      <job_dir>/movies.json       {"names": [...]}   (nomes dos episodios)
      <job_dir>/refs/<i>/*.jpg     fotos do personagem i
      <job_dir>/input_<e>.*        episodio e (0,1,2,...)
Gera: <job_dir>/results.json       {"episodes": [...]}
      <job_dir>/thumbs/<i>.jpg
Imprime: MSG <txt> / PROG <i> <total> / DONE / ERR <msg>
"""

import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
WORK_H = 720
SCENE_DIFF = 14.0


def log(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _bin(name):
    import shutil
    local = os.path.join(BASE_DIR, "bin", name + (".exe" if os.name == "nt" else ""))
    return local if os.path.exists(local) else (shutil.which(name) or local)


def available():
    import importlib.util
    for m in ("insightface", "onnxruntime", "cv2"):
        if importlib.util.find_spec(m) is None:
            return False, "dependencia ausente: %s" % m
    return True, None


def _load_bgr(path):
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def probe(path, ffprobe):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, creationflags=_NO_WINDOW,
    )
    d = json.loads(out.stdout or "{}")
    st = d["streams"][0]
    return int(st["width"]), int(st["height"]), float(d.get("format", {}).get("duration", 0) or 0)


def _merge(times, gap):
    if not times:
        return []
    times = sorted(times)
    ranges = [[times[0], times[0]]]
    for t in times[1:]:
        if t - ranges[-1][1] <= gap:
            ranges[-1][1] = t
        else:
            ranges.append([t, t])
    return ranges


def _embed_refs(app, job_dir, names, thumbs_dir):
    os.makedirs(thumbs_dir, exist_ok=True)
    char_embs = []
    for i, name in enumerate(names):
        ref_dir = os.path.join(job_dir, "refs", str(i))
        embs, thumb_saved = [], False
        if os.path.isdir(ref_dir):
            for fn in sorted(os.listdir(ref_dir)):
                try:
                    img = _load_bgr(os.path.join(ref_dir, fn))
                except Exception:
                    continue
                faces = app.get(img)
                if not faces:
                    continue
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embs.append(face.normed_embedding)
                if not thumb_saved:
                    x1, y1, x2, y2 = [max(0, int(v)) for v in face.bbox]
                    crop = img[y1:y2, x1:x2][:, :, ::-1]
                    if crop.size:
                        Image.fromarray(crop).save(os.path.join(thumbs_dir, "%d.jpg" % i))
                        thumb_saved = True
        if not embs:
            raise ValueError("nenhum rosto detectado nas fotos de '%s'" % name)
        char_embs.append(np.array(embs, dtype=np.float32))
    return char_embs


def _scan_movie(app, char_embs, names, inp, ffmpeg, ffprobe, fps, thr, on_frame):
    """Varre um video e devolve (episodio_dict, nquadros)."""
    w, h, dur = probe(inp, ffprobe)
    sh = min(WORK_H, h)
    sw = int(round(w * sh / h / 2)) * 2
    interval = 1.0 / fps
    gap = interval * 3 + 0.1

    dec = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", inp, "-vf", "fps=%s,scale=%d:%d" % (fps, sw, sh),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE, creationflags=_NO_WINDOW,
    )
    fb = sw * sh * 3
    per_char = [[] for _ in names]
    frame_sets, scene_of = [], []
    scene_id, prev_small, idx = 0, None, 0
    try:
        while True:
            buf = dec.stdout.read(fb)
            if not buf or len(buf) < fb:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
            t = idx * interval
            small = frame[::max(1, sh // 36), ::max(1, sw // 64), :].mean(axis=2)
            if prev_small is not None and small.shape == prev_small.shape and \
                    np.abs(small - prev_small).mean() > SCENE_DIFF:
                scene_id += 1
            prev_small = small
            scene_of.append(scene_id)
            present = set()
            for face in app.get(frame):
                emb = face.normed_embedding
                best_i, best_sim = -1, thr
                for i, refs in enumerate(char_embs):
                    sim = float(np.max(refs @ emb))
                    if sim >= best_sim:
                        best_sim, best_i = sim, i
                if best_i >= 0:
                    present.add(best_i)
            for i in present:
                per_char[i].append(t)
            frame_sets.append((t, present))
            idx += 1
            on_frame()
    finally:
        try:
            dec.stdout.close()
        except Exception:
            pass
    dec.wait()

    characters = []
    for i, name in enumerate(names):
        ranges = _merge(per_char[i], gap)
        characters.append({
            "name": name,
            "total": round(sum(b - a + interval for a, b in ranges), 1),
            "appearances": len(ranges),
            "ranges": [[round(a, 1), round(b + interval, 1)] for a, b in ranges],
        })

    pair_frame = {}
    for t, present in frame_sets:
        pl = sorted(present)
        for a in range(len(pl)):
            for b in range(a + 1, len(pl)):
                pair_frame.setdefault((pl[a], pl[b]), []).append(t)
    together_frame = [{"chars": list(k), "ranges": [[round(x, 1), round(y + interval, 1)]
                       for x, y in _merge(v, gap)]} for k, v in sorted(pair_frame.items())]

    scene_chars, scene_span = {}, {}
    for (t, present), sid in zip(frame_sets, scene_of):
        scene_chars.setdefault(sid, set()).update(present)
        sp = scene_span.setdefault(sid, [t, t]); sp[1] = t
    pair_scene = {}
    for sid, chars in scene_chars.items():
        cl = sorted(chars); a0, b0 = scene_span[sid]
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                pair_scene.setdefault((cl[a], cl[b]), []).append([round(a0, 1), round(b0 + interval, 1)])
    together_scene = [{"chars": list(k), "ranges": v} for k, v in sorted(pair_scene.items())]

    ep = {"duration": round(dur, 1), "characters": characters,
          "together_frame": together_frame, "together_scene": together_scene}
    return ep, idx


def run(job_dir, fps, sens):
    ok, err = available()
    if not ok:
        log("ERR " + err)
        return 1
    import cv2  # noqa
    from insightface.app import FaceAnalysis

    fps = float(fps)
    ffmpeg, ffprobe = _bin("ffmpeg"), _bin("ffprobe")
    jobname = os.path.basename(os.path.normpath(job_dir))

    movies = sorted(n for n in os.listdir(job_dir) if n.startswith("input_")
                    or n.startswith("input."))
    if not movies:
        log("ERR nenhum video enviado")
        return 1
    with open(os.path.join(job_dir, "chars.json"), encoding="utf-8") as f:
        names = json.load(f).get("names", [])
    if not names:
        log("ERR nenhum personagem definido")
        return 1
    ep_names = []
    try:
        with open(os.path.join(job_dir, "movies.json"), encoding="utf-8") as f:
            ep_names = json.load(f).get("names", [])
    except Exception:
        pass

    thr = max(0.15, min(0.55, 0.5 - (float(sens) / 100.0) * 0.35))

    log("MSG Carregando o modelo de rostos...")
    app = FaceAnalysis(name="buffalo_l",
                       providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    log("MSG Analisando as fotos dos personagens...")
    try:
        char_embs = _embed_refs(app, job_dir, names, os.path.join(job_dir, "thumbs"))
    except ValueError as e:
        log("ERR " + str(e))
        return 1

    # total de quadros de todos os episodios (p/ a barra de progresso)
    grand = 0
    for mv in movies:
        try:
            _w, _h, dur = probe(os.path.join(job_dir, mv), ffprobe)
            grand += int(dur * fps)
        except Exception:
            pass
    grand = grand or 1
    log("PROG 0 %d" % grand)

    thumbs = [("/files/%s/thumbs/%d.jpg" % (jobname, i))
              if os.path.exists(os.path.join(job_dir, "thumbs", "%d.jpg" % i)) else None
              for i in range(len(names))]

    episodes = []
    done = [0]

    def on_frame():
        done[0] += 1
        log("PROG %d %d" % (min(done[0], grand), grand))

    for ei, mv in enumerate(movies):
        label = ep_names[ei] if ei < len(ep_names) else ("Episódio %d" % (ei + 1))
        log("MSG Analisando %s (%d/%d)..." % (label, ei + 1, len(movies)))
        ep, _n = _scan_movie(app, char_embs, names,
                             os.path.join(job_dir, mv), ffmpeg, ffprobe, fps, thr, on_frame)
        for i, c in enumerate(ep["characters"]):
            c["thumb"] = thumbs[i]
        ep["name"] = label
        ep["movie"] = "/files/%s/%s" % (jobname, mv)
        ep["index"] = ei
        episodes.append(ep)

    with open(os.path.join(job_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"fps_sample": fps, "threshold": round(thr, 3),
                   "char_names": names, "char_thumbs": thumbs,
                   "episodes": episodes}, f, ensure_ascii=False)
    log("DONE results.json")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        log("ERR uso: face_worker.py <job_dir> <fps> <sens>")
        sys.exit(2)
    try:
        sys.exit(run(sys.argv[1], sys.argv[2], sys.argv[3]))
    except Exception as e:
        log("ERR " + str(e))
        sys.exit(1)
