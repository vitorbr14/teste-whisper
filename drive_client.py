"""Client da Google Drive API pro worker RunPod.

Baixa o arquivo original (baixar_do_drive) usando Service Account e sobe
o .srt gerado pelo pipeline (subir_para_drive) usando OAuth da conta Google.

Credenciais nunca ficam em disco nem na imagem Docker:
- GOOGLE_SERVICE_ACCOUNT_JSON: Service Account usada no download.
- GOOGLE_OAUTH_CLIENT_ID: Client ID OAuth.
- GOOGLE_OAUTH_CLIENT_SECRET: Client Secret OAuth.
- GOOGLE_OAUTH_REFRESH_TOKEN: Refresh Token da conta que fará os uploads.

Remoção do original e tradução ficam fora deste arquivo.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))

_SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._\- ]")

_CHUNK_SIZE_BYTES = 10 * 1024 * 1024

_SRT_MIMETYPE = "application/x-subrip"


# São dois clients diferentes porque representam duas identidades diferentes.
_service_account_drive_service = None
_oauth_drive_service = None


# ---------------------------------------------------------------------------
# Service Account — usada para DOWNLOAD
# ---------------------------------------------------------------------------

def _carregar_credenciais_service_account() -> service_account.Credentials:
    """Carrega a Service Account a partir do Secret da RunPod."""
    raw_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not raw_json:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON não configurada "
            "(RunPod Secret com o JSON da service account)."
        )

    try:
        info = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON não contém um JSON válido."
        ) from exc

    return service_account.Credentials.from_service_account_info(
        info,
        scopes=DRIVE_SCOPES,
    )


def _build_service_account_drive_service():
    """Constrói/reaproveita o Drive client autenticado como Service Account."""
    global _service_account_drive_service

    if _service_account_drive_service is None:
        credentials = _carregar_credenciais_service_account()

        _service_account_drive_service = build(
            "drive",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )

    return _service_account_drive_service


# ---------------------------------------------------------------------------
# OAuth — usado para UPLOAD
# ---------------------------------------------------------------------------

def _carregar_credenciais_oauth() -> Credentials:
    """Monta as credenciais OAuth da conta Google responsável pelos uploads."""

    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("GOOGLE_OAUTH_REFRESH_TOKEN")

    if not client_id:
        raise RuntimeError("GOOGLE_OAUTH_CLIENT_ID não configurado.")

    if not client_secret:
        raise RuntimeError("GOOGLE_OAUTH_CLIENT_SECRET não configurado.")

    if not refresh_token:
        raise RuntimeError("GOOGLE_OAUTH_REFRESH_TOKEN não configurado.")

    return Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=DRIVE_SCOPES,
    )


def _build_oauth_drive_service():
    """Constrói/reaproveita o Drive client autenticado via OAuth."""
    global _oauth_drive_service

    if _oauth_drive_service is None:
        credentials = _carregar_credenciais_oauth()

        _oauth_drive_service = build(
            "drive",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )

    return _oauth_drive_service


# ---------------------------------------------------------------------------
# Utilidades de arquivo
# ---------------------------------------------------------------------------

def _sanitizar_nome_arquivo(file_name: str) -> str:
    """Reduz file_name a um nome de arquivo seguro, sem path traversal."""

    nome = Path(file_name).name

    if not nome or nome in (".", ".."):
        raise ValueError(f"Nome de arquivo inválido: {file_name!r}")

    nome_sanitizado = _SAFE_FILENAME_CHARS.sub("_", nome)

    if not nome_sanitizado:
        raise ValueError(
            f"Nome de arquivo inválido após sanitização: {file_name!r}"
        )

    return nome_sanitizado


def _resolver_destino(nome_sanitizado: str) -> Path:
    """Monta o path final dentro de WORKSPACE_DIR."""

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    destino = (WORKSPACE_DIR / nome_sanitizado).resolve()
    workspace_resolvido = WORKSPACE_DIR.resolve()

    if (
        workspace_resolvido not in destino.parents
        and destino != workspace_resolvido
    ):
        raise ValueError(
            f"Path de destino fora de WORKSPACE_DIR: {destino}"
        )

    return destino


# ---------------------------------------------------------------------------
# Download — Service Account
# ---------------------------------------------------------------------------

def baixar_do_drive(file_id: str, file_name: str) -> str:
    """Baixa o arquivo do Drive usando a Service Account."""

    nome_sanitizado = _sanitizar_nome_arquivo(file_name)
    destino = _resolver_destino(nome_sanitizado)

    print(f"[drive_client] baixando file_id={file_id} -> {destino}")

    try:
        # IMPORTANTE:
        # download continua usando Service Account.
        drive = _build_service_account_drive_service()

        request = drive.files().get_media(fileId=file_id)

        with open(destino, "wb") as arquivo_local:
            downloader = MediaIoBaseDownload(
                arquivo_local,
                request,
                chunksize=_CHUNK_SIZE_BYTES,
            )

            concluido = False

            while not concluido:
                status, concluido = downloader.next_chunk()

                if status:
                    pct = int(status.progress() * 100)
                    print(
                        f"[drive_client] baixando "
                        f"{nome_sanitizado}: {pct}%"
                    )

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


# ---------------------------------------------------------------------------
# Upload — OAuth da conta Google
# ---------------------------------------------------------------------------

def subir_para_drive(local_path: str) -> str:
    """Sobe o SRT usando OAuth da conta Google."""

    folder_id = os.environ.get("DRIVE_OUTPUT_FOLDER_ID")

    if not folder_id:
        raise RuntimeError(
            "DRIVE_OUTPUT_FOLDER_ID não configurada "
            "(ID da pasta de Saída no Drive)."
        )

    origem = Path(local_path)

    if not origem.is_file():
        raise RuntimeError(
            f"Arquivo local não encontrado para upload: {origem}"
        )

    print(
        f"[drive_client] subindo {origem.name} -> pasta {folder_id}"
    )

    try:
        # ESTA é a mudança importante:
        # upload agora usa OAuth, não Service Account.
        drive = _build_oauth_drive_service()

        metadata = {
            "name": origem.name,
            "parents": [folder_id],
        }

        media = MediaFileUpload(
            str(origem),
            mimetype=_SRT_MIMETYPE,
            resumable=True,
            chunksize=_CHUNK_SIZE_BYTES,
        )

        request = drive.files().create(
            body=metadata,
            media_body=media,
            fields="id",
        )

        response = None

        while response is None:
            status, response = request.next_chunk()

            if status:
                pct = int(status.progress() * 100)
                print(
                    f"[drive_client] subindo "
                    f"{origem.name}: {pct}%"
                )

    except HttpError as exc:
        raise RuntimeError(
            f"Falha ao subir {origem} pro Drive: {exc}"
        ) from exc

    except OSError as exc:
        raise RuntimeError(
            f"Falha ao ler arquivo local {origem}: {exc}"
        ) from exc

    file_id = response["id"]

    print(
        f"[drive_client] upload concluído: file_id={file_id}"
    )

    return file_id
