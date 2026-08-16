import runpod

from drive_client import baixar_do_drive
from pipeline import transcribe_file


def handler(job):
    job_input = job.get("input", {})

    file_id = job_input["file_id"]
    file_name = job_input["file_name"]

    print(f"[job] recebido: {file_name} ({file_id})")

    # 1. Baixa o arquivo real do Google Drive
    input_file = baixar_do_drive(
        file_id=file_id,
        file_name=file_name,
    )

    # 2. Transcreve com o pipeline que já existe
    result = transcribe_file(
        input_file=input_file,
        model_name="large-v3",
        chunk_seconds=600,
        force=True,
    )

    # Depois vamos adicionar aqui:
    # 3. upload do SRT pro Drive
    # 4. remoção/movimentação do original
    # 5. limpeza dos arquivos temporários

    return result


runpod.serverless.start({
    "handler": handler
})
