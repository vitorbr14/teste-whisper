import runpod
from pipeline import transcribe_file

def handler(job):
    job_input = job.get("input", {})

    input_file = job_input["input_file"]

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