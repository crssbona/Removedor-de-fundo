# -*- coding: utf-8 -*-
"""
Servidor local do "Removedor de Fundo".

- Serve o site estatico (index.html) que remove fundo de imagens no navegador.
- Expoe uma pequena API que usa yt-dlp + ffmpeg para baixar videos do YouTube
  em MP4 (maxima qualidade) ou MP3, com progresso em tempo real (SSE).

Tudo roda apenas em 127.0.0.1 (somente nesta maquina). Para iniciar:

    python server.py

Depois abra http://localhost:8000 no navegador.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
BIN_DIR = os.path.join(BASE_DIR, "bin")

_port_arg = next((a for a in sys.argv[1:] if a.isdigit()), None)
PORT = int(_port_arg or os.environ.get("PORT", 8765))
NO_BROWSER = ("--no-browser" in sys.argv) or os.environ.get("NO_BROWSER") == "1"

os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# No Windows evita abrir uma janela de console a cada subprocesso.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def ffmpeg_path():
    """Caminho do ffmpeg: primeiro o portatil em ./bin, senao o do PATH."""
    local = os.path.join(BIN_DIR, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if os.path.exists(local):
        return BIN_DIR  # yt-dlp aceita o diretorio
    found = shutil.which("ffmpeg")
    return os.path.dirname(found) if found else None


def _bin(name):
    """Caminho completo de um executavel (ffmpeg/ffprobe): ./bin ou PATH."""
    local = os.path.join(BIN_DIR, name + (".exe" if os.name == "nt" else ""))
    return local if os.path.exists(local) else shutil.which(name)


def ffmpeg_bin():
    return _bin("ffmpeg")


def ffprobe_bin():
    return _bin("ffprobe")


# Carregamento preguicoso do motor de video (importa onnxruntime — pesado).
_vmatte_mod = False  # False = ainda nao tentado; None = falhou; modulo = ok
_vmatte_err = None


def get_vmatte():
    global _vmatte_mod, _vmatte_err
    if _vmatte_mod is False:
        try:
            import vmatte as _m
            _vmatte_mod = _m
        except Exception as e:
            _vmatte_mod = None
            _vmatte_err = str(e)
    return _vmatte_mod


# modelos SAM 2 suportados (chave -> checkpoint em models/)
_SAM_MODELS = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
}


def sam2_models():
    """Modelos SAM 2 cujo checkpoint esta baixado em models/."""
    md = os.path.join(BASE_DIR, "models")
    return [k for k, f in _SAM_MODELS.items() if os.path.exists(os.path.join(md, f))]


def _sam2_pkg():
    import importlib.util
    try:
        return (importlib.util.find_spec("sam2") is not None
                and importlib.util.find_spec("torch") is not None)
    except Exception:
        return False


def sam2_available():
    """SAM 2 pronto? (checa sem importar o torch — find_spec e' barato)."""
    return bool(sam2_models()) and _sam2_pkg()


def matanyone_available():
    """MatAnyone pronto? Precisa do pacote + SAM 2 (semente) + checkpoint tiny."""
    import importlib.util
    try:
        pkg = all(importlib.util.find_spec(m) is not None
                  for m in ("matanyone", "sam2", "torch"))
    except Exception:
        pkg = False
    tiny = os.path.join(BASE_DIR, "models", "sam2.1_hiera_tiny.pt")
    return pkg and os.path.exists(tiny)


def faces_available():
    """Reconhecimento facial pronto? (insightface + onnxruntime + opencv)."""
    import importlib.util
    try:
        return all(importlib.util.find_spec(m) is not None
                   for m in ("insightface", "onnxruntime", "cv2"))
    except Exception:
        return False


def srt_available():
    """Legenda por palavra pronta? (torch + torchaudio + demucs)."""
    import importlib.util
    try:
        return all(importlib.util.find_spec(m) is not None
                   for m in ("torch", "torchaudio", "demucs"))
    except Exception:
        return False


def ytdlp_cmd(*args):
    """Monta o comando do yt-dlp usando o mesmo Python (evita problemas de PATH)."""
    return [sys.executable, "-m", "yt_dlp", *args]


def ytdlp_version():
    try:
        out = subprocess.run(
            ytdlp_cmd("--version"),
            capture_output=True, text=True, creationflags=_NO_WINDOW, timeout=20,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def is_valid_url(url):
    return isinstance(url, str) and re.match(r"^https?://", url.strip()) is not None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "RemovedorDeFundo/1.0"
    protocol_version = "HTTP/1.0"  # framing por fechamento de conexao (bom p/ SSE)

    def log_message(self, fmt, *args):
        # log enxuto
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    # ---- roteamento -------------------------------------------------------
    def do_HEAD(self):
        # usado por checagens de prontidao (ex.: HEAD /)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/api/ping":
            return self.api_ping()
        if path == "/api/info":
            return self.api_info(qs)
        if path == "/api/download":
            return self.api_download(qs)
        if path == "/api/vmatte":
            return self.api_vmatte(qs)
        if path == "/api/vframe":
            return self.api_vframe(qs)
        if path == "/api/face/run":
            return self.api_face_run(qs)
        if path == "/api/face/results":
            return self.api_face_results(qs)
        if path == "/api/face/clip":
            return self.api_face_clip(qs)
        if path == "/api/face/supercut":
            return self.api_face_supercut(qs)
        if path == "/api/srt/run":
            return self.api_srt_run(qs)
        if path == "/api/srt/words":
            return self.api_srt_words(qs)
        if path.startswith("/files/"):
            return self.serve_download(path)
        return self.serve_static(path)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/vupload":
            return self.api_vupload(parsed)
        if parsed.path == "/api/vpoints":
            return self.api_vpoints(parsed)
        if parsed.path == "/api/face/job":
            return self.api_face_job()
        if parsed.path == "/api/face/names":
            return self.api_face_names(parsed)
        if parsed.path == "/api/face/ref":
            return self.api_face_ref(parsed)
        if parsed.path == "/api/face/movie":
            return self.api_face_movie(parsed)
        if parsed.path == "/api/face/movies":
            return self.api_face_movies(parsed)
        if parsed.path == "/api/srt/job":
            return self.api_srt_job()
        if parsed.path == "/api/srt/audio":
            return self.api_srt_audio(parsed)
        if parsed.path == "/api/srt/lyrics":
            return self.api_srt_lyrics(parsed)
        self.send_error(404, "Nao encontrado")

    # ---- API: ping --------------------------------------------------------
    def api_ping(self):
        v = get_vmatte()
        self.send_json({
            "ok": True,
            "ytdlp": ytdlp_version(),
            "ffmpeg": ffmpeg_path() is not None,
            "vmatte": bool(v and v.model_available()),
            "vmatte_gpu": bool(v and v.gpu_available()),
            "vmatte_error": None if v else _vmatte_err,
            "sam": sam2_available(),
            "sam_models": sam2_models(),
            "matanyone": matanyone_available(),
            "faces": faces_available(),
            "srt": srt_available(),
        })

    # ---- API: info (titulo / thumb / duracao) -----------------------------
    def api_info(self, qs):
        url = (qs.get("url") or [""])[0].strip()
        if not is_valid_url(url):
            return self.send_json({"ok": False, "error": "URL invalida."}, status=400)
        try:
            out = subprocess.run(
                ytdlp_cmd("--dump-single-json", "--no-playlist", "--no-warnings", "--", url),
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                creationflags=_NO_WINDOW, timeout=60,
            )
            if out.returncode != 0:
                msg = (out.stderr or "").strip().splitlines()
                return self.send_json(
                    {"ok": False, "error": msg[-1] if msg else "Falha ao obter informacoes."},
                    status=502,
                )
            data = json.loads(out.stdout)
            thumb = data.get("thumbnail")
            return self.send_json({
                "ok": True,
                "title": data.get("title"),
                "uploader": data.get("uploader") or data.get("channel"),
                "duration": data.get("duration"),
                "thumbnail": thumb,
            })
        except subprocess.TimeoutExpired:
            return self.send_json({"ok": False, "error": "Tempo esgotado ao obter informacoes."}, status=504)
        except Exception as e:
            return self.send_json({"ok": False, "error": str(e)}, status=500)

    # ---- API: download (SSE com progresso) --------------------------------
    def api_download(self, qs):
        url = (qs.get("url") or [""])[0].strip()
        fmt = (qs.get("format") or ["mp4"])[0].strip().lower()
        vq = (qs.get("vq") or ["1080"])[0].strip().lower()   # qualidade de video (mp4)
        aq = (qs.get("aq") or ["320"])[0].strip().lower()    # qualidade de audio (mp3)
        if not is_valid_url(url):
            return self.send_json({"ok": False, "error": "URL invalida."}, status=400)
        if fmt not in ("mp4", "mp3"):
            fmt = "mp4"
        if vq not in ("best", "2160", "1440", "1080", "720", "480", "360"):
            vq = "1080"
        if aq not in ("best", "320", "256", "192", "128"):
            aq = "320"

        # abre o stream SSE
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        job_id = uuid.uuid4().hex[:12]
        job_dir = os.path.join(DOWNLOADS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        ff = ffmpeg_path()
        common = [
            "--no-playlist", "--newline", "--encoding", "utf-8", "--no-mtime",
            "--progress-template",
            "PROG|%(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s",
            "-P", job_dir, "-o", "%(title)s.%(ext)s",
        ]
        if ff:
            common += ["--ffmpeg-location", ff]

        if fmt == "mp3":
            quality = "0" if aq == "best" else aq + "K"
            sel = ["-f", "ba/b", "-x", "--audio-format", "mp3", "--audio-quality", quality]
            want_ext = ".mp3"
        else:
            if vq == "best":
                # melhor video possivel (pode ser AV1/VP9 em 4K)
                sel = ["-f", "bv*+ba/b", "--merge-output-format", "mp4"]
            else:
                # respeita a resolucao escolhida; dentro dela prefere H.264/AAC (compativel)
                h = vq
                sel = [
                    "-f", "bv*[height<=%s]+ba/b[height<=%s]/b" % (h, h),
                    "-S", "res:%s,vcodec:h264,acodec:aac,br" % h,
                    "--merge-output-format", "mp4",
                ]
            want_ext = ".mp4"

        cmd = ytdlp_cmd(*common, *sel, "--", url)

        self._sse("status", {"message": "Iniciando..."})

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=_NO_WINDOW, bufsize=1,
            )
        except Exception as e:
            self._sse("error", {"message": "Nao foi possivel iniciar o yt-dlp: %s" % e})
            return

        tail = []  # ultimas linhas, para diagnostico de erro
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                tail.append(line)
                tail[:] = tail[-12:]

                if line.startswith("PROG|"):
                    _, pct, speed, eta = (line.split("|") + ["", "", ""])[:4]
                    self._sse("progress", {
                        "percent": _num(pct), "speed": speed.strip(), "eta": eta.strip(),
                    })
                elif "[Merger]" in line or "Merging formats" in line:
                    self._sse("status", {"message": "Juntando video e audio..."})
                elif "[ExtractAudio]" in line or "Extracting audio" in line:
                    self._sse("status", {"message": "Convertendo para MP3..."})
                elif "[download] Destination" in line:
                    self._sse("status", {"message": "Baixando..."})
                elif line.startswith("[youtube]") or line.startswith("[info]"):
                    self._sse("status", {"message": "Preparando..."})
        except (BrokenPipeError, ConnectionResetError):
            proc.kill()
            return  # cliente desconectou

        proc.wait()

        if proc.returncode != 0:
            self._sse("error", {"message": _clean_err(tail)})
            return

        final = _pick_output(job_dir, want_ext)
        if not final:
            self._sse("error", {"message": "O download terminou mas o arquivo nao foi encontrado."})
            return

        rel = "/files/%s/%s" % (job_id, urllib.parse.quote(final))
        self._sse("done", {"file": rel, "name": final})

    # ---- API: upload de video (POST cru) ----------------------------------
    def api_vupload(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        name = (qs.get("name") or ["video.mp4"])[0]
        base = os.path.splitext(os.path.basename(name))[0] or "video"
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv", ".flv", ".gif"):
            ext = ".mp4"
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return self.send_json({"ok": False, "error": "Arquivo vazio."}, status=400)

        job_id = uuid.uuid4().hex[:12]
        job_dir = os.path.join(DOWNLOADS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        in_path = os.path.join(job_dir, "input" + ext)
        remaining = length
        with open(in_path, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 512, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        with open(os.path.join(job_dir, "name.txt"), "w", encoding="utf-8") as f:
            f.write(base)
        self.send_json({"ok": True, "job": job_id, "name": base})

    # ---- API: remover fundo de video (SSE com progresso) ------------------
    def api_vmatte(self, qs):
        job = (qs.get("job") or [""])[0].strip()
        fmt = (qs.get("fmt") or ["mp4green"])[0].strip().lower()
        chroma = (qs.get("chroma") or ["green"])[0].strip().lower()
        audio = (qs.get("audio") or ["1"])[0].strip() != "0"
        engine = (qs.get("engine") or ["auto"])[0].strip().lower()
        sammodel = (qs.get("sammodel") or ["tiny"])[0].strip().lower()
        if not re.fullmatch(r"[0-9a-f]{6,32}", job or ""):
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        if fmt not in ("mp4green", "webm", "mov"):
            fmt = "mp4green"
        if chroma not in ("green", "blue", "magenta"):
            chroma = "green"
        if engine not in ("auto", "sam", "matanyone"):
            engine = "auto"
        if sammodel not in sam2_models():
            sammodel = (sam2_models() or ["tiny"])[0]

        job_dir = os.path.join(DOWNLOADS_DIR, job)
        if not os.path.isdir(job_dir):
            return self.send_json({"ok": False, "error": "Job nao encontrado."}, status=404)
        inp = next((os.path.join(job_dir, n) for n in os.listdir(job_dir)
                    if n.startswith("input.")), None)
        if not inp:
            return self.send_json({"ok": False, "error": "Video de entrada nao encontrado."}, status=404)
        ffmpeg, ffprobe = ffmpeg_bin(), ffprobe_bin()
        if not ffmpeg or not ffprobe:
            return self.send_json({"ok": False, "error": "ffmpeg/ffprobe nao encontrados."}, status=500)

        # validacao especifica por motor (antes de abrir o SSE)
        m = None
        if engine in ("sam", "matanyone"):
            avail = sam2_available() if engine == "sam" else matanyone_available()
            if not avail:
                return self.send_json({"ok": False, "error": "Motor '%s' indisponivel (modelo/dependencias)." % engine}, status=500)
            if not os.path.exists(os.path.join(job_dir, "points.json")):
                return self.send_json({"ok": False, "error": "Marque ao menos um objeto antes de processar."}, status=400)
        else:
            m = get_vmatte()
            if not m:
                return self.send_json({"ok": False, "error": "Motor de IA indisponivel: %s" % (_vmatte_err or "")}, status=500)
            if not m.model_available():
                return self.send_json({"ok": False, "error": "Modelo nao encontrado em models/."}, status=500)

        base = "video"
        try:
            with open(os.path.join(job_dir, "name.txt"), encoding="utf-8") as f:
                base = f.read().strip() or "video"
        except Exception:
            pass
        out_ext = {"mp4green": ".mp4", "webm": ".webm", "mov": ".mov"}[fmt]
        out_name = "%s-sem-fundo%s" % (base, out_ext)
        out_path = os.path.join(job_dir, out_name)

        # abre o stream SSE
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        if engine == "sam":
            nome = "Base+" if sammodel == "base_plus" else "Tiny"
            cmd = [sys.executable, os.path.join(BASE_DIR, "sam2_worker.py"),
                   job_dir, fmt, chroma, "1" if audio else "0", sammodel]
            return self._run_worker(job, cmd, "Carregando SAM 2 (%s)..." % nome)
        if engine == "matanyone":
            cmd = [sys.executable, os.path.join(BASE_DIR, "matanyone_worker.py"),
                   job_dir, fmt, chroma, "1" if audio else "0"]
            return self._run_worker(job, cmd, "Carregando MatAnyone...")

        # ---- motor automatico (IS-Net) ----
        self._sse("status", {"message": "Carregando modelo de IA..."})
        aborted = {"flag": False}

        def on_progress(i, total):
            try:
                self._sse("progress", {
                    "frame": i, "total": total,
                    "percent": round(i / total * 100, 1) if total else None,
                })
            except (BrokenPipeError, ConnectionResetError):
                aborted["flag"] = True
                raise m.AbortError()

        try:
            n = m.process(inp, out_path, fmt, chroma, audio, ffmpeg, ffprobe,
                          on_progress=on_progress, should_abort=lambda: aborted["flag"])
        except m.AbortError:
            return
        except Exception as e:
            try:
                self._sse("error", {"message": "Falha ao processar: %s" % e})
            except Exception:
                pass
            return

        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            self._sse("error", {"message": "O processamento terminou mas o arquivo nao foi gerado."})
            return
        rel = "/files/%s/%s" % (job, urllib.parse.quote(out_name))
        self._sse("done", {"file": rel, "name": out_name, "frames": n})

    def _run_worker(self, job, cmd, loading_msg):
        """Roda um worker de video (SAM 2 / MatAnyone) e repassa o progresso (SSE)."""
        self._sse("status", {"message": loading_msg})
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=_NO_WINDOW, bufsize=1, cwd=BASE_DIR,
            )
        except Exception as e:
            self._sse("error", {"message": "Nao foi possivel iniciar o SAM 2: %s" % e})
            return

        tail, done_sent, err_msg = [], False, None
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                tail.append(line); tail[:] = tail[-15:]
                if line.startswith("PROG "):
                    p = line.split()
                    i, total = int(p[1]), int(p[2])
                    self._sse("progress", {"frame": i, "total": total,
                                           "percent": round(i / total * 100, 1) if total else None})
                elif line.startswith("MSG "):
                    self._sse("status", {"message": line[4:].strip()})
                elif line.startswith("DONE "):
                    name = line[5:].strip()
                    rel = "/files/%s/%s" % (job, urllib.parse.quote(name))
                    self._sse("done", {"file": rel, "name": name})
                    done_sent = True
                elif line.startswith("ERR "):
                    err_msg = line[4:].strip()
        except (BrokenPipeError, ConnectionResetError):
            proc.kill()
            return
        proc.wait()
        if done_sent:
            return
        self._sse("error", {"message": err_msg or (tail[-1] if tail else "Falha no SAM 2.")})

    # ---- API: primeiro quadro do video (p/ marcar objetos) ----------------
    def api_vframe(self, qs):
        job = (qs.get("job") or [""])[0].strip()
        if not re.fullmatch(r"[0-9a-f]{6,32}", job or ""):
            return self.send_error(400, "Job invalido")
        job_dir = os.path.join(DOWNLOADS_DIR, job)
        inp = None
        if os.path.isdir(job_dir):
            inp = next((os.path.join(job_dir, n) for n in os.listdir(job_dir)
                        if n.startswith("input.")), None)
        if not inp:
            return self.send_error(404, "Video nao encontrado")
        fpath = os.path.join(job_dir, "firstframe.jpg")
        if not os.path.exists(fpath):
            ff = ffmpeg_bin()
            subprocess.run([ff, "-v", "error", "-i", inp, "-frames:v", "1", "-q:v", "2", fpath],
                           creationflags=_NO_WINDOW)
        if not os.path.exists(fpath):
            return self.send_error(500, "Falha ao extrair o quadro")
        with open(fpath, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- API: salvar os pontos de clique (objetos) ------------------------
    def api_vpoints(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        job = (qs.get("job") or [""])[0].strip()
        if not re.fullmatch(r"[0-9a-f]{6,32}", job or ""):
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        job_dir = os.path.join(DOWNLOADS_DIR, job)
        if not os.path.isdir(job_dir):
            return self.send_json({"ok": False, "error": "Job nao encontrado."}, status=404)
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            return self.send_json({"ok": False, "error": "JSON invalido."}, status=400)
        clean = []
        for o in data.get("objects", []):
            pts = [{"x": float(c["x"]), "y": float(c["y"]), "pos": bool(c.get("pos", True))}
                   for c in o if "x" in c and "y" in c]
            if pts:
                clean.append(pts)
        if not clean:
            return self.send_json({"ok": False, "error": "Nenhum objeto marcado."}, status=400)
        with open(os.path.join(job_dir, "points.json"), "w", encoding="utf-8") as f:
            json.dump({"objects": clean}, f)
        self.send_json({"ok": True, "objects": len(clean)})

    # ---- API: reconhecimento facial em filmes -----------------------------
    def api_face_job(self):
        job = uuid.uuid4().hex[:12]
        os.makedirs(os.path.join(DOWNLOADS_DIR, job), exist_ok=True)
        self.send_json({"ok": True, "job": job})

    def _face_dir(self, qs):
        job = (qs.get("job") or [""])[0].strip()
        if not re.fullmatch(r"[0-9a-f]{6,32}", job or ""):
            return None, None
        d = os.path.join(DOWNLOADS_DIR, job)
        return (job, d) if os.path.isdir(d) else (job, None)

    def _read_body(self, path):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return False
        remaining = length
        with open(path, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 512, remaining))
                if not chunk:
                    break
                f.write(chunk); remaining -= len(chunk)
        return True

    def api_face_names(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        job, d = self._face_dir(qs)
        if not d:
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            return self.send_json({"ok": False, "error": "JSON invalido."}, status=400)
        names = [(str(n).strip()[:60] or "Personagem") for n in data.get("names", [])]
        if not names:
            return self.send_json({"ok": False, "error": "Sem personagens."}, status=400)
        with open(os.path.join(d, "chars.json"), "w", encoding="utf-8") as f:
            json.dump({"names": names}, f, ensure_ascii=False)
        self.send_json({"ok": True, "count": len(names)})

    def api_face_ref(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        job, d = self._face_dir(qs)
        if not d:
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        try:
            ci = max(0, min(int((qs.get("ci") or ["0"])[0]), 99))
        except ValueError:
            ci = 0
        ext = os.path.splitext((qs.get("name") or ["img.jpg"])[0])[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            ext = ".jpg"
        rd = os.path.join(d, "refs", str(ci))
        os.makedirs(rd, exist_ok=True)
        if not self._read_body(os.path.join(rd, uuid.uuid4().hex[:8] + ext)):
            return self.send_json({"ok": False, "error": "Imagem vazia."}, status=400)
        self.send_json({"ok": True})

    def api_face_movie(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        job, d = self._face_dir(qs)
        if not d:
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        try:
            ei = max(0, min(int((qs.get("ei") or ["0"])[0]), 999))
        except ValueError:
            ei = 0
        ext = os.path.splitext((qs.get("name") or ["filme.mp4"])[0])[1].lower()
        if ext not in (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".ts"):
            ext = ".mp4"
        if not self._read_body(os.path.join(d, "input_%d%s" % (ei, ext))):
            return self.send_json({"ok": False, "error": "Filme vazio."}, status=400)
        self.send_json({"ok": True})

    def api_face_movies(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        job, d = self._face_dir(qs)
        if not d:
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            return self.send_json({"ok": False, "error": "JSON invalido."}, status=400)
        names = [(str(n).strip()[:80] or ("Episódio %d" % (i + 1)))
                 for i, n in enumerate(data.get("names", []))]
        with open(os.path.join(d, "movies.json"), "w", encoding="utf-8") as f:
            json.dump({"names": names}, f, ensure_ascii=False)
        self.send_json({"ok": True, "count": len(names)})

    def api_face_results(self, qs):
        job, d = self._face_dir(qs)
        rp = os.path.join(d, "results.json") if d else None
        if not rp or not os.path.isfile(rp):
            return self.send_error(404, "Resultados nao encontrados")
        with open(rp, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def api_face_run(self, qs):
        job, d = self._face_dir(qs)
        if not d:
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        if not faces_available():
            return self.send_json({"ok": False, "error": "Reconhecimento facial indisponivel."}, status=500)
        if not os.path.exists(os.path.join(d, "chars.json")):
            return self.send_json({"ok": False, "error": "Defina os personagens primeiro."}, status=400)
        if not any(n.startswith("input_") or n.startswith("input.") for n in os.listdir(d)):
            return self.send_json({"ok": False, "error": "Envie ao menos um vídeo primeiro."}, status=400)
        try:
            fps = str(max(0.25, min(4.0, float((qs.get("fps") or ["1"])[0]))))
            sens = str(max(0, min(100, int(float((qs.get("sens") or ["50"])[0])))))
        except ValueError:
            fps, sens = "1", "50"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        cmd = [sys.executable, os.path.join(BASE_DIR, "face_worker.py"), d, fps, sens]
        self._sse("status", {"message": "Carregando o modelo de rostos..."})
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace",
                                    creationflags=_NO_WINDOW, bufsize=1, cwd=BASE_DIR)
        except Exception as e:
            self._sse("error", {"message": "Nao foi possivel iniciar: %s" % e})
            return
        tail, done, err_msg = [], False, None
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                tail.append(line); tail[:] = tail[-15:]
                if line.startswith("PROG "):
                    p = line.split()
                    i, total = int(p[1]), int(p[2])
                    self._sse("progress", {"frame": i, "total": total,
                                           "percent": round(i / total * 100, 1) if total else None})
                elif line.startswith("MSG "):
                    self._sse("status", {"message": line[4:].strip()})
                elif line.startswith("DONE"):
                    self._sse("done", {"job": job})
                    done = True
                elif line.startswith("ERR "):
                    err_msg = line[4:].strip()
        except (BrokenPipeError, ConnectionResetError):
            proc.kill()
            return
        proc.wait()
        if not done:
            self._sse("error", {"message": err_msg or (tail[-1] if tail else "Falha na analise.")})

    # ---- API: cortar cena/take em qualidade original (sem re-encodar) ------
    def _face_input(self, d, ei=0):
        if not os.path.isdir(d):
            return None
        pref = "input_%d." % ei
        inp = next((os.path.join(d, n) for n in os.listdir(d) if n.startswith(pref)), None)
        return inp or next((os.path.join(d, n) for n in os.listdir(d) if n.startswith("input.")), None)

    def _extract_segment(self, ffmpeg, inp, start, dur, out_path, fmt=None):
        cmd = [ffmpeg, "-y", "-v", "error", "-ss", "%.3f" % start, "-i", inp,
               "-t", "%.3f" % dur, "-c", "copy", "-avoid_negative_ts", "make_zero"]
        if fmt:
            cmd += ["-f", fmt]
        cmd += [out_path]
        return subprocess.run(cmd, creationflags=_NO_WINDOW).returncode == 0

    def api_face_clip(self, qs):
        job, d = self._face_dir(qs)
        if not d:
            return self.send_error(400, "Job invalido")
        try:
            ei = max(0, int((qs.get("ei") or ["0"])[0]))
        except ValueError:
            ei = 0
        inp = self._face_input(d, ei)
        ffmpeg = ffmpeg_bin()
        if not inp or not ffmpeg:
            return self.send_error(404, "Filme nao encontrado")
        try:
            start = max(0.0, float((qs.get("start") or ["0"])[0]))
            end = float((qs.get("end") or ["0"])[0])
        except ValueError:
            return self.send_error(400, "Tempo invalido")
        label = re.sub(r"[^\w.-]", "_", (qs.get("name") or ["cena"])[0])[:50] or "cena"
        clips = os.path.join(d, "clips"); os.makedirs(clips, exist_ok=True)
        out = os.path.join(clips, "%s_%s.mp4" % (label, uuid.uuid4().hex[:6]))
        if not self._extract_segment(ffmpeg, inp, start, max(0.1, end - start), out) or not os.path.isfile(out):
            return self.send_error(500, "Falha ao cortar")
        with open(out, "rb") as f:
            body = f.read()
        fn = "%s_%s-%s.mp4" % (label, _hms(start), _hms(end))
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", "attachment; filename*=UTF-8''%s" % urllib.parse.quote(fn))
        self.end_headers()
        self.wfile.write(body)

    def _sel_ranges(self, res, sel):
        """Devolve (lista de (episodio_idx, [faixas]), rotulo)."""
        eps = res.get("episodes", [])
        names = res.get("char_names", [])
        try:
            if sel.startswith("allchar:"):
                i = int(sel[8:])
                segs = []
                for ep in eps:
                    r = [tuple(x) for x in ep["characters"][i]["ranges"]]
                    if r:
                        segs.append((ep["index"], r))
                return (segs, "%s-todos-eps" % names[i]) if segs else (None, None)
            if sel.startswith("ep:"):
                _, ei_s, rest = sel.split(":", 2)
                ei = int(ei_s)
                ep = next((e for e in eps if e["index"] == ei), None)
                if not ep:
                    return None, None
                if rest.startswith("char:"):
                    i = int(rest[5:])
                    return [(ei, [tuple(x) for x in ep["characters"][i]["ranges"]])], \
                           "%s-ep%d" % (names[i], ei + 1)
                if rest.startswith("frame:") or rest.startswith("scene:"):
                    key = "together_frame" if rest.startswith("frame:") else "together_scene"
                    a, b = [int(x) for x in rest.split(":", 1)[1].split("-")]
                    for p in ep.get(key, []):
                        if p["chars"] == [a, b]:
                            return [(ei, [tuple(x) for x in p["ranges"]])], \
                                   "%s+%s-ep%d" % (names[a], names[b], ei + 1)
        except Exception:
            pass
        return None, None

    def api_face_supercut(self, qs):
        job, d = self._face_dir(qs)
        if not d:
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        ffmpeg = ffmpeg_bin()
        rp = os.path.join(d, "results.json")
        if not ffmpeg or not os.path.isfile(rp):
            return self.send_json({"ok": False, "error": "Dados nao encontrados."}, status=404)
        with open(rp, encoding="utf-8") as f:
            res = json.load(f)
        seglist, label = self._sel_ranges(res, (qs.get("sel") or [""])[0])
        if not seglist:
            return self.send_json({"ok": False, "error": "Selecao invalida."}, status=400)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        clips = os.path.join(d, "clips"); os.makedirs(clips, exist_ok=True)
        total = sum(len(r) for _ei, r in seglist)
        self._sse("status", {"message": "Cortando %d cena(s) sem perder qualidade..." % total})
        segs, k = [], 0
        try:
            for ei, ranges in seglist:
                inp = self._face_input(d, ei)
                if not inp:
                    continue
                for (a, b) in ranges:
                    seg = os.path.join(clips, "seg_%04d.ts" % k)
                    if self._extract_segment(ffmpeg, inp, a, max(0.1, b - a), seg, fmt="mpegts"):
                        segs.append(seg)
                    k += 1
                    self._sse("progress", {"frame": k, "total": total,
                                           "percent": round(k / total * 100, 1)})
        except (BrokenPipeError, ConnectionResetError):
            return
        if not segs:
            self._sse("error", {"message": "Nenhuma cena pode ser cortada."})
            return
        self._sse("status", {"message": "Juntando as cenas..."})
        out_name = "%s-cenas.mp4" % (re.sub(r"[^\w.-]", "_", label)[:50] or "cenas")
        out = os.path.join(clips, out_name)
        rc = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", "concat:" + "|".join(segs),
                             "-c", "copy", out], creationflags=_NO_WINDOW).returncode
        for s in segs:
            try:
                os.remove(s)
            except Exception:
                pass
        if rc != 0 or not os.path.isfile(out):
            self._sse("error", {"message": "Falha ao juntar as cenas."})
            return
        rel = "/files/%s/clips/%s" % (job, urllib.parse.quote(out_name))
        self._sse("done", {"file": rel, "name": out_name})

    # ---- API: legenda palavra-por-palavra (musica + letra -> SRT) ---------
    def api_srt_job(self):
        job = uuid.uuid4().hex[:12]
        os.makedirs(os.path.join(DOWNLOADS_DIR, job), exist_ok=True)
        self.send_json({"ok": True, "job": job})

    def api_srt_audio(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        job, d = self._face_dir(qs)
        if not d:
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        ext = os.path.splitext((qs.get("name") or ["audio.mp3"])[0])[1].lower()
        if ext not in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
                       ".mp4", ".mkv", ".mov", ".webm"):
            ext = ".mp3"
        if not self._read_body(os.path.join(d, "input" + ext)):
            return self.send_json({"ok": False, "error": "Audio vazio."}, status=400)
        self.send_json({"ok": True})

    def api_srt_lyrics(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        job, d = self._face_dir(qs)
        if not d:
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8", "replace") if length else ""
        if not raw.strip():
            return self.send_json({"ok": False, "error": "Letra vazia."}, status=400)
        with open(os.path.join(d, "lyrics.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
        self.send_json({"ok": True})

    def api_srt_words(self, qs):
        job, d = self._face_dir(qs)
        wp = os.path.join(d, "words.json") if d else None
        if not wp or not os.path.isfile(wp):
            return self.send_error(404, "Resultado nao encontrado")
        with open(wp, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def api_srt_run(self, qs):
        job, d = self._face_dir(qs)
        if not d:
            return self.send_json({"ok": False, "error": "Job invalido."}, status=400)
        if not srt_available():
            return self.send_json({"ok": False, "error": "Legenda indisponivel (dependencias)."}, status=500)
        if not any(n.startswith("input.") for n in os.listdir(d)):
            return self.send_json({"ok": False, "error": "Envie o audio primeiro."}, status=400)
        if not os.path.exists(os.path.join(d, "lyrics.txt")):
            return self.send_json({"ok": False, "error": "Cole a letra primeiro."}, status=400)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        cmd = [sys.executable, os.path.join(BASE_DIR, "srt_worker.py"), d]
        self._sse("status", {"message": "Carregando os modelos..."})
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace",
                                    creationflags=_NO_WINDOW, bufsize=1, cwd=BASE_DIR)
        except Exception as e:
            self._sse("error", {"message": "Nao foi possivel iniciar: %s" % e})
            return
        tail, done, err_msg = [], False, None
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                tail.append(line); tail[:] = tail[-15:]
                if line.startswith("MSG "):
                    self._sse("status", {"message": line[4:].strip()})
                elif line.startswith("DONE"):
                    self._sse("done", {"job": job})
                    done = True
                elif line.startswith("ERR "):
                    err_msg = line[4:].strip()
        except (BrokenPipeError, ConnectionResetError):
            proc.kill()
            return
        proc.wait()
        if not done:
            self._sse("error", {"message": err_msg or (tail[-1] if tail else "Falha ao gerar a legenda.")})

    # ---- servir o arquivo baixado -----------------------------------------
    def serve_download(self, path):
        rel = urllib.parse.unquote(path[len("/files/"):])
        full = os.path.abspath(os.path.join(DOWNLOADS_DIR, rel))
        # impede path traversal
        if not full.startswith(os.path.abspath(DOWNLOADS_DIR) + os.sep) or not os.path.isfile(full):
            return self.send_error(404, "Arquivo nao encontrado")

        name = os.path.basename(full)
        ext = os.path.splitext(name)[1].lower().lstrip(".")
        ctype = {"mp4": "video/mp4", "mp3": "audio/mpeg", "m4a": "audio/mp4",
                 "webm": "video/webm", "mov": "video/quicktime", "mkv": "video/x-matroska",
                 "avi": "video/x-msvideo", "ts": "video/mp2t",
                 "wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg",
                 "opus": "audio/ogg", "aac": "audio/aac",
                 "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                 "json": "application/json", "srt": "text/plain; charset=utf-8"}.get(ext, "application/octet-stream")
        size = os.path.getsize(full)

        # suporte a Range (necessario p/ o <video> buscar/seek em arquivos grandes)
        start, end, partial = 0, size - 1, False
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                s, e = rng[6:].split("-", 1)
                if s.strip():
                    start = int(s)
                    end = int(e) if e.strip() else size - 1
                else:
                    start = max(0, size - int(e))
                start = max(0, start); end = min(end, size - 1)
                partial = start <= end
            except Exception:
                partial = False
        length = (end - start + 1) if partial else size

        self.send_response(206 if partial else 200)
        if partial:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        # inline: o <video> toca; os botoes de download usam o atributo download (forca baixar)
        self.send_header("Content-Disposition", "inline; filename*=UTF-8''%s" % urllib.parse.quote(name))
        self.end_headers()
        with open(full, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 256, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    # ---- arquivos estaticos -----------------------------------------------
    def serve_static(self, path):
        if path == "/" or path == "":
            path = "/index.html"
        full = os.path.abspath(os.path.join(BASE_DIR, path.lstrip("/")))
        if not full.startswith(BASE_DIR) or not os.path.isfile(full):
            return self.send_error(404, "Nao encontrado")
        ext = os.path.splitext(full)[1].lower()
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".json": "application/json",
                 ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
                 ".ico": "image/x-icon"}.get(ext, "application/octet-stream")
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- helpers ----------------------------------------------------------
    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, event, data):
        payload = "event: %s\ndata: %s\n\n" % (event, json.dumps(data))
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.flush()


def _num(s):
    m = re.search(r"[\d.]+", s or "")
    return float(m.group()) if m else None


def _pick_output(job_dir, want_ext):
    cands = []
    for n in os.listdir(job_dir):
        if n.endswith((".part", ".ytdl", ".temp")):
            continue
        cands.append(n)
    if not cands:
        return None
    exact = [n for n in cands if n.lower().endswith(want_ext)]
    pool = exact or cands
    pool.sort(key=lambda n: os.path.getsize(os.path.join(job_dir, n)), reverse=True)
    return pool[0]


def _hms(s):
    s = int(round(float(s)))
    return "%dm%02ds" % (s // 60, s % 60) if s < 3600 else "%dh%02dm%02ds" % (s // 3600, s % 3600 // 60, s % 60)


def _clean_err(tail):
    for line in reversed(tail):
        if "ERROR" in line:
            return re.sub(r"^\W*ERROR:\W*", "", line).strip()
    return tail[-1] if tail else "Falha desconhecida no download."


def main():
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    httpd.daemon_threads = True
    url = "http://localhost:%d" % PORT
    print("=" * 56)
    print("  Removedor de Fundo + Downloader  ->  %s" % url)
    print("  yt-dlp:", ytdlp_version() or "NAO ENCONTRADO")
    print("  ffmpeg:", "ok (%s)" % ffmpeg_path() if ffmpeg_path() else "NAO ENCONTRADO")
    _v = get_vmatte()
    if _v and _v.model_available():
        print("  video auto (IS-Net):", "ok %s" % ("[GPU]" if _v.gpu_available() else "[CPU]"))
    else:
        print("  video auto (IS-Net):", "modelo ausente" if _v else "deps ausentes (%s)" % _vmatte_err)
    print("  video preciso (SAM 2):", (", ".join(sam2_models()) if sam2_available() else "nao instalado"))
    print("  video matte (MatAnyone):", "ok" if matanyone_available() else "nao instalado")
    print("  rostos em filmes (InsightFace):", "ok" if faces_available() else "nao instalado")
    print("  legenda por palavra (Demucs+MMS):", "ok" if srt_available() else "nao instalado")
    print("  (Ctrl+C para encerrar)")
    print("=" * 56)
    # abre o navegador automaticamente (so quando rodado direto)
    if not NO_BROWSER:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrado.")


if __name__ == "__main__":
    main()
