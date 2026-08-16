"""Entrypoint do worker RunPod Serverless.

Só orquestra: baixa o original do Drive, chama o pipeline de
transcrição/tradução (worker/pipeline.py, não implementado aqui) e sobe o
.srt resultante de volta pro Drive. Nenhuma lógica de negócio própria —
download, transcrição e upload vivem cada um no seu módulo.

Não deleta nem move o áudio original (etapa futura, fora deste arquivo).
"""

import runpod

from drive_client import baixar_do_drive, subir_para_drive
from pipeline import transcribe_file


def handler(job):
    job_input = job["input"]
    file_id = job_input["file_id"]
    file_name = job_input["file_name"]

    input_file = baixar_do_drive(file_id=file_id, file_name=file_name)

    result = transcribe_file(
        input_file=input_file,
        model_name="large-v3",
        chunk_seconds=600,
        force=True,
    )

    srt_path = result.get("srt_path")
    if not srt_path:
        raise RuntimeError(
            "transcribe_file não retornou srt_path (transcrição falhou?)"
        )

    result["srt_drive_id"] = subir_para_drive(srt_path)

    return result


runpod.serverless.start({"handler": handler})
