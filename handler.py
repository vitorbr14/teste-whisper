import runpod

from drive_client import baixar_do_drive
from pipeline import transcribe_file


def handler(job):
    job_input = job.get("input", {})

    file_id = job_input["file_id"]
    file_name = job_input["file_name"]

    print(
        f"Arquivo que disparou o job: "
        f"{file_name} "
        f"({file_id})"
    )

    # Baixa o arquivo real do Google Drive para /workspace
    input_file = baixar_do_drive(
        file_id=file_id,
        file_name=file_name,
    )

    # Transcreve o arquivo baixado
    result = transcribe_file(
        input_file=input_file,
        model_name="large-v3",
        chunk_seconds=600,
        force=True,
    )

    return result


runpod.serverless.start({
    "handler": handler
})
