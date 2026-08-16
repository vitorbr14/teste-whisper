import runpod
from pipeline import transcribe_file


def handler(job):
    job_input = job.get("input", {})

    print(
        f"Arquivo que disparou o job: "
        f"{job_input.get('file_name')} "
        f"({job_input.get('file_id')})"
    )

    # TEMPORÁRIO: ainda não baixa o arquivo real do Drive
    input_file = "/workspace/emo.mp3"

    return transcribe_file(
        input_file=input_file,
        model_name="large-v3",
        chunk_seconds=600,
        force=True,
    )


runpod.serverless.start({
    "handler": handler
})
