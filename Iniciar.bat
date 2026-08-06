@echo off
chcp 65001 >nul
title Ferramentas de Midia
cd /d "%~dp0"

REM --- confere se o Python esta instalado ---
where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo  [X] Python nao encontrado.
  echo      Instale o Python 3.10 ou mais novo em: https://www.python.org/downloads/
  echo      IMPORTANTE: marque "Add Python to PATH" durante a instalacao.
  echo.
  pause
  exit /b 1
)

REM --- prepara tudo (idempotente; a 1a vez baixa varios GB e demora) ---
echo Verificando/instalando o necessario...
python setup.py
if errorlevel 1 (
  echo.
  echo  [X] Falha na preparacao. Veja as mensagens acima.
  pause
  exit /b 1
)

REM --- sobe o servidor (o navegador abre sozinho) ---
echo.
echo Iniciando o servidor... (feche esta janela para encerrar)
python server.py
echo.
echo O servidor foi encerrado.
pause
