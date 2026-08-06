# 🧰 Ferramentas de Mídia

Cinco ferramentas locais para edição, num só lugar — **tudo roda no seu PC** (sem enviar seus arquivos pra nuvem):

1. **🪄 Imagem** — remove o fundo de imagens (PNG transparente), direto no navegador.
2. **⬇️ YouTube** — baixa vídeos em MP4 (até 4K) ou MP3, com escolha de qualidade.
3. **🎬 Vídeo** — remove o fundo de vídeos com IA (automático, "preciso" por cliques com SAM 2, ou "matte" com MatAnyone). Saída pronta pro chroma key.
4. **🎭 Personagens** — a partir de fotos de referência, acha em que minutos cada personagem aparece num filme (live-action), com player e recorte das cenas em qualidade original.
5. **🎵 Legenda** — música + letra → SRT **palavra por palavra** sincronizado (separa os vocais com Demucs e alinha), com preview karaokê.

---

## ▶️ Como usar

1. Baixe/clone este repositório.
2. Dê **dois cliques em `Iniciar.bat`**.
3. Na **primeira vez**, ele baixa e instala tudo automaticamente (veja abaixo). Depois, abre sozinho em `http://localhost:8765`.

> ⚠️ **A primeira execução baixa vários GB** (PyTorch + modelos de IA) e **demora bastante** — deixe rodando. Das próximas vezes, abre rápido. Se algo já estiver baixado, ele pula sem erro.

---

## ✅ Pré-requisitos

| Item | Necessário? |
|---|---|
| **Windows** | Sim |
| **Python 3.10+** | Sim — instale de [python.org](https://www.python.org/downloads/) e **marque "Add Python to PATH"** |
| **GPU NVIDIA** | Opcional — recomendado. Sem ela, roda na CPU (bem mais lento) / DirectML |
| **Git** | Opcional — só para os modos **Preciso** e **Matte** do vídeo (SAM 2 / MatAnyone). Instale de [git-scm.com](https://git-scm.com) |

O `Iniciar.bat` cuida do resto: instala as bibliotecas de Python e baixa os modelos (ffmpeg, IS-Net, SAM 2, InsightFace, Demucs, MMS, MatAnyone).

---

## 🔒 Privacidade e uso

- Tudo é processado **localmente**. Seus arquivos não saem do seu computador.
- Use apenas conteúdo que você tem direito de usar. Baixar vídeos do YouTube pode violar os Termos de Serviço deles — a responsabilidade é sua.
- Alguns modelos (ex.: **MatAnyone**) têm **licença de pesquisa / uso não-comercial**. Reconhecimento facial serve para indexar personagens de filmes de ficção — não use para vigilância de pessoas reais.

---

## 🛠️ O que fica na pasta

- `index.html` — a interface (site)
- `server.py` — o servidor local
- `*_worker.py` — os motores de IA (vídeo, personagens, legenda…)
- `setup.py` — instalador automático (roda no `Iniciar.bat`)
- `bin/`, `models/`, `downloads/` — **baixados na 1ª execução** (não versionados)

Detalhes de cada aba estão no `LEIA-ME.txt`.
