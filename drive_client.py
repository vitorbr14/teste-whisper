"""Client da Google Drive API pro worker RunPod.

Baixa o arquivo original (baixar_do_drive) e sobe o .srt gerado pelo
pipeline (subir_para_drive). Remoção do original e tradução ficam fora
deste arquivo (ver CLAUDE.md do projeto).

Autenticação: service account, sem arquivo de credencial em disco nem na
imagem Docker. O JSON da service account chega via variável de ambiente
GOOGLE_SERVICE_ACCOUNT_JSON (configurada como RunPod Secret), lido e
parseado em memória a cada cold start.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# Escopo completo (não só readonly): upload/delete serão adicionados em
# etapas seguintes e reaproveitam essa mesma credencial/secret.
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

# Onde o worker RunPod espera os arquivos de entrada/saída do job.
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))

# Allowlist de caracteres aceitos num nome de arquivo sanitizado.
_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._\- ]")

_CHUNK_SIZE_BYTES = 10 * 1024 * 1024  # 10 MiB por chunk de download/upload

# Mimetype fixo pro upload — arquivo sempre é o .srt gerado pelo pipeline.
_SRT_MIMETYPE = "application/x-subrip"

_drive_service = None  # cache do client, reaproveitado entre invocações "quentes"


def _carregar_credenciais() -> service_account.Credentials:
    """Lê e parseia a service account a partir da env var. Nunca toca disco."""
    raw_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw_json:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON não configurada (RunPod Secret com o "
            "JSON da service account)."
        )

    try:
        info = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON não contém um JSON válido."
        ) from exc

    return service_account.Credentials.from_service_account_info(
        info, scopes=DRIVE_SCOPES
    )


def _build_drive_service():
    """Constrói (ou reaproveita) o client autenticado da Drive API v3."""
    global _drive_service
    if _drive_service is None:
        credentials = _carregar_credenciais()
        _drive_service = build(
            "drive", "v3", credentials=credentials, cache_discovery=False
        )
    return _drive_service


def _sanitizar_nome_arquivo(file_name: str) -> str:
    """Reduz file_name a um nome de arquivo seguro, sem path traversal.

    - descarta qualquer componente de diretório (basename apenas);
    - rejeita nome vazio ou "." / ".." após o basename;
    - troca qualquer caractere fora do allowlist por "_".
    """
    nome = Path(file_name).name

    if not nome or nome in (".", ".."):
        raise ValueError(f"Nome de arquivo inválido: {file_name!r}")

    nome_sanitizado = _SAFE_FILENAME_CHARS.sub("_", nome)

    if not nome_sanitizado:
        raise ValueError(f"Nome de arquivo inválido após sanitização: {file_name!r}")

    return nome_sanitizado


def _resolver_destino(nome_sanitizado: str) -> Path:
    """Monta o path final em WORKSPACE_DIR, com checagem de defesa em profundidade."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    destino = (WORKSPACE_DIR / nome_sanitizado).resolve()
    workspace_resolvido = WORKSPACE_DIR.resolve()

    if workspace_resolvido not in destino.parents and destino != workspace_resolvido:
        # Não deveria acontecer dado que _sanitizar_nome_arquivo já bloqueia
        # separadores de path, mas é barato garantir e falha alto.
        raise ValueError(f"Path de destino fora de WORKSPACE_DIR: {destino}")

    return destino


def baixar_do_drive(file_id: str, file_name: str) -> str:
    """Baixa o arquivo `file_id` do Drive para /workspace/<file_name sanitizado>.

    Retorna o caminho local do arquivo salvo. Lança RuntimeError se o
    download falhar por qualquer motivo (arquivo inexistente, permissão
    negada, erro de rede, erro de escrita local, etc.).
    """
    nome_sanitizado = _sanitizar_nome_arquivo(file_name)
    destino = _resolver_destino(nome_sanitizado)

    print(f"[drive_client] baixando file_id={file_id} -> {destino}")

    try:
        drive = _build_drive_service()
        request = drive.files().get_media(fileId=file_id)

        with open(destino, "wb") as arquivo_local:
            downloader = MediaIoBaseDownload(
                arquivo_local, request, chunksize=_CHUNK_SIZE_BYTES
            )
            concluido = False
            while not concluido:
                status, concluido = downloader.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    print(f"[drive_client] baixando {nome_sanitizado}: {pct}%")
    except HttpError as exc:
        raise RuntimeError(
            f"Falha ao baixar file_id={file_id} do Drive: {exc}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Falha ao escrever arquivo local em {destino}: {exc}"
        ) from exc

    print(f"[drive_client] download concluído: {destino}")
    return str(destino)


def subir_para_drive(local_path: str) -> str:
    """Sobe o arquivo local (o .srt gerado pelo pipeline) pra pasta de Saída.

    Pasta de destino vem de DRIVE_OUTPUT_FOLDER_ID (env var estática do
    worker, configurada direto no endpoint RunPod — não é secret, só um ID
    de pasta). Lida aqui dentro, não em constante de módulo, pra não quebrar
    o import deste arquivo em cenários que só usam baixar_do_drive.

    Retorna o ID (str) do arquivo criado no Drive. Lança RuntimeError se a
    env var não estiver configurada, o arquivo local não existir, ou o
    upload falhar por qualquer motivo.
    """
    folder_id = os.environ.get("DRIVE_OUTPUT_FOLDER_ID")
    if not folder_id:
        raise RuntimeError(
            "DRIVE_OUTPUT_FOLDER_ID não configurada (ID da pasta de Saída no Drive)."
        )

    origem = Path(local_path)
    if not origem.is_file():
        raise RuntimeError(f"Arquivo local não encontrado para upload: {origem}")

    print(f"[drive_client] subindo {origem.name} -> pasta {folder_id}")

    try:
        drive = _build_drive_service()
        metadata = {"name": origem.name, "parents": [folder_id]}
        media = MediaFileUpload(
            str(origem),
            mimetype=_SRT_MIMETYPE,
            resumable=True,
            chunksize=_CHUNK_SIZE_BYTES,
        )
        request = drive.files().create(body=metadata, media_body=media, fields="id")

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                print(f"[drive_client] subindo {origem.name}: {pct}%")
    except HttpError as exc:
        raise RuntimeError(f"Falha ao subir {origem} pro Drive: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Falha ao ler arquivo local {origem}: {exc}") from exc

    file_id = response["id"]
    print(f"[drive_client] upload concluído: file_id={file_id}")
    return file_id
