import os
import time
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel


def load_dotenv(path: str = ".env") -> None:
    env_file = Path(path)

    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


LANGUAGE = "ja"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"


MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
    ".mp3", ".wav", ".flac", ".opus", ".m4a", ".aac", ".ogg",
}


# Parâmetros de resegmentação de legendas (ajustáveis)
MAX_CAPTION_CHARS = 40          # nº máx. de caracteres por legenda
MAX_CAPTION_DURATION = 6.0      # duração máx. de uma legenda, em segundos
MIN_PAUSE_FOR_BREAK = 0.5       # gap mínimo (s) entre palavras p/ considerar corte natural
SENTENCE_END_CHARS = "。！？.!?"
CLAUSE_BREAK_CHARS = "、,，"


# ---------------------------------------------------------------------------
# Núcleo de transcrição (reutilizável — sem input(), sem argparse, sem tradução)
# ---------------------------------------------------------------------------


def make_logger(log_file: Path, base_dir: Path):
    """Cria uma função de log isolada para um job específico."""

    def log(message: str) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] {message}"

        print(line)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log


def run_command(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    if millis == 1000:
        secs += 1
        millis = 0

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(output_file: Path, segments_data: list[dict]) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments_data, start=1):
            f.write(f"{i}\n")
            f.write(
                f"{format_srt_time(segment['start'])} --> "
                f"{format_srt_time(segment['end'])}\n"
            )
            f.write(f"{segment['text']}\n\n")


def reset_if_force(base_dir: Path, force: bool) -> None:
    if force and base_dir.exists():
        print(f"🧹 Limpando pasta antiga: {base_dir}")
        shutil.rmtree(base_dir)


def prepare_dirs(audio_dir: Path, chunks_dir: Path, srt_dir: Path) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    srt_dir.mkdir(parents=True, exist_ok=True)


def extract_audio(input_file: str, audio_dir: Path, force: bool, log) -> Path:
    output_audio = audio_dir / "audio.wav"

    if output_audio.exists() and not force:
        log(f"⏭️ Áudio já existe: {output_audio}")
        return output_audio

    log(f"🎧 Extraindo áudio de: {input_file}")

    run_command([
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_audio),
    ])

    log(f"✅ Áudio criado: {output_audio}")
    return output_audio


def chunk_audio(audio_file: Path, chunks_dir: Path, chunk_seconds: int, force: bool, log) -> list[Path]:
    existing_chunks = sorted(chunks_dir.glob("chunk_*.wav"))

    if existing_chunks and not force:
        log(f"⏭️ Chunks já existem: {len(existing_chunks)}")
        return existing_chunks

    for old_chunk in existing_chunks:
        old_chunk.unlink()

    log(f"✂️ Dividindo áudio em chunks de {chunk_seconds}s")

    output_pattern = chunks_dir / "chunk_%03d.wav"

    run_command([
        "ffmpeg",
        "-y",
        "-i", str(audio_file),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_pattern),
    ])

    chunks = sorted(chunks_dir.glob("chunk_*.wav"))

    log(f"✅ Total de chunks criados: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# Resegmentação de legendas (quebra segmentos grandes usando word timestamps)
# ---------------------------------------------------------------------------


def _split_words_into_groups(
    words: list[dict],
    max_chars: int = MAX_CAPTION_CHARS,
    max_duration: float = MAX_CAPTION_DURATION,
    pause_threshold: float = MIN_PAUSE_FOR_BREAK,
) -> list[list[dict]]:
    """Agrupa palavras (start/end/word) em blocos respeitando os limites de
    caracteres e duração, preferindo cortar em pontuação ou pausas naturais."""

    if not words:
        return []

    n = len(words)
    groups = []
    start = 0
    last_break = None
    i = 0

    while i < n:
        text_len = sum(len(w["word"]) for w in words[start:i + 1])
        duration = words[i]["end"] - words[start]["start"]

        if i > start and (text_len > max_chars or duration > max_duration):
            cut = last_break if last_break is not None else i - 1
            groups.append(words[start:cut + 1])
            start = cut + 1
            last_break = None
            continue

        word_text = words[i]["word"]
        ends_sentence = bool(word_text) and word_text[-1] in SENTENCE_END_CHARS
        ends_clause = bool(word_text) and word_text[-1] in CLAUSE_BREAK_CHARS
        gap_after = words[i + 1]["start"] - words[i]["end"] if i + 1 < n else None
        has_pause_after = gap_after is not None and gap_after >= pause_threshold

        if ends_sentence or ends_clause or has_pause_after:
            last_break = i

        i += 1

    if start < n:
        groups.append(words[start:])

    return groups


def _words_to_caption(group: list[dict]) -> dict:
    """Reconstrói uma legenda a partir de um grupo de palavras. A concatenação
    direta (sem espaço) é intencional: no faster-whisper cada word.word já
    vem com o espaçamento correto embutido (japonês normalmente não usa
    espaço entre palavras)."""

    return {
        "start": group[0]["start"],
        "end": group[-1]["end"],
        "text": "".join(w["word"] for w in group).strip(),
    }


def resegment_into_captions(
    segments,
    max_chars: int = MAX_CAPTION_CHARS,
    max_duration: float = MAX_CAPTION_DURATION,
    pause_threshold: float = MIN_PAUSE_FOR_BREAK,
) -> list[dict]:
    """Recebe os Segment do faster-whisper (com word_timestamps=True) e
    devolve uma lista plana de legendas {start, end, text}, quebrando
    segmentos grandes em legendas menores a partir dos timestamps de
    palavra. Segmentos já dentro dos limites não são fragmentados."""

    captions = []

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        words = segment.words or []
        segment_duration = segment.end - segment.start

        if not words or (len(text) <= max_chars and segment_duration <= max_duration):
            captions.append({"start": segment.start, "end": segment.end, "text": text})
            continue

        word_dicts = [{"start": w.start, "end": w.end, "word": w.word} for w in words]

        for group in _split_words_into_groups(word_dicts, max_chars, max_duration, pause_threshold):
            captions.append(_words_to_caption(group))

    return captions


def transcribe_chunk(
    model: WhisperModel,
    chunk_file: Path,
    chunk_index: int,
    chunk_seconds: int,
    srt_dir: Path,
    log,
) -> list[dict]:
    start_time = time.time()
    chunk_srt_file = srt_dir / f"{chunk_file.stem}.srt"

    try:
        log(f"🎙️ Transcrevendo: {chunk_file.name}")

        segments, info = model.transcribe(
            str(chunk_file),
            language=LANGUAGE,
            beam_size=10,
            # beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=True,
            initial_prompt=(
    "これはLiella!、ラブライブ！スーパースター!!、Love Live!に関する日本語の会話です。"
    "出演者や声優、メンバーの名前、愛称、グループ名が頻繁に登場します。"
    "Liella!、Aqours、ラブライブ！、ラブライブ！スーパースター!!、"
    "絵森彩、Aya Emori、Emori Aya、えもりん、エモリン。"
    "自然な日本語の会話で、声優、ライブ、イベント、アニメ、楽曲について話すことがあります。"
)
        )

        raw_segments = list(segments)

        offset = chunk_index * chunk_seconds

        local_segments = resegment_into_captions(raw_segments)

        global_segments = [
            {
                "start": caption["start"] + offset,
                "end": caption["end"] + offset,
                "text": caption["text"],
            }
            for caption in local_segments
        ]

        write_srt(chunk_srt_file, local_segments)

        elapsed = time.time() - start_time

        log(
            f"✅ Finalizado: {chunk_file.name} | "
            f"SRT individual: {chunk_srt_file.name} | "
            f"Idioma: {info.language} | "
            f"Duração: {info.duration:.2f}s | "
            f"Offset: {offset}s | "
            f"Tempo: {elapsed:.2f}s ({elapsed / 60:.2f} min)"
        )

        return global_segments

    except Exception as e:
        elapsed = time.time() - start_time
        log(f"❌ Erro em {chunk_file.name}: {str(e)} | Tempo até erro: {elapsed:.2f}s")
        return []


def write_final_srt(all_segments: list[dict], final_srt: Path, log) -> None:
    log(f"🧩 Gerando SRT final: {final_srt}")

    all_segments = sorted(all_segments, key=lambda segment: segment["start"])
    write_srt(final_srt, all_segments)

    log(f"✅ SRT final criado: {final_srt}")


def transcribe_file(
    input_file: str,
    model_name: str = "large-v3",
    chunk_seconds: int = 600,
    force: bool = False,
) -> dict:
    """
    Executa a transcrição completa (extração de áudio, chunking, transcrição
    com faster-whisper e montagem do SRT final) para um único arquivo.

    Não faz tradução, não pergunta nada ao usuário e não depende de estado
    global: cada chamada calcula seus próprios paths a partir de input_file,
    então pode ser chamada várias vezes (com arquivos diferentes) no mesmo
    processo sem interferência entre chamadas — é o que permite reaproveitar
    esta função tanto no CLI local quanto em um futuro worker Serverless.
    """

    overall_start = time.time()

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")

    output_name = input_path.stem

    base_dir = Path("process") / output_name
    audio_dir = base_dir / "audio"
    chunks_dir = base_dir / "chunks"
    srt_dir = base_dir / "srt"
    final_srt = base_dir / f"{output_name}.srt"
    log_file = base_dir / "process.log"

    log = make_logger(log_file, base_dir)

    reset_if_force(base_dir, force)
    prepare_dirs(audio_dir, chunks_dir, srt_dir)

    log(f"📄 Arquivo: {input_file}")
    log("🌐 Idioma fixo: japonês")
    log(f"🧠 Modelo: {model_name}")
    log(f"🖥️ Device: {DEVICE}")
    log(f"⚡ Compute: {COMPUTE_TYPE}")
    log(f"✂️ Chunk: {chunk_seconds}s")
    log(f"🧹 Force: {force}")

    audio_file = extract_audio(str(input_path), audio_dir, force, log)
    chunks = chunk_audio(audio_file, chunks_dir, chunk_seconds, force, log)

    if not chunks:
        log("⚠️ Nenhum chunk foi encontrado.")
        return {
            "srt_path": None,
            "input_file": str(input_path),
            "model": model_name,
            "chunk_seconds": chunk_seconds,
            "chunks": 0,
            "segments": 0,
            "elapsed_seconds": time.time() - overall_start,
            "log_file": str(log_file),
        }

    log(f"📦 Total para transcrever: {len(chunks)} chunks")
    log("🧠 Carregando modelo Whisper")

    model = WhisperModel(
        model_name,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    all_segments = []

    for index, chunk in enumerate(chunks):
        log(f"📍 Processando chunk {index + 1}/{len(chunks)}")
        segments = transcribe_chunk(model, chunk, index, chunk_seconds, srt_dir, log)
        all_segments.extend(segments)

    if all_segments:
        write_final_srt(all_segments, final_srt, log)
    else:
        log("⚠️ Nenhum segmento foi gerado. SRT final não criado.")

    overall_elapsed = time.time() - overall_start

    log(
        f"🏁 Tudo concluído | "
        f"Chunks: {len(chunks)} | "
        f"Tempo total: {overall_elapsed:.2f}s ({overall_elapsed / 60:.2f} min)"
    )

    return {
        "srt_path": str(final_srt) if all_segments else None,
        "input_file": str(input_path),
        "model": model_name,
        "chunk_seconds": chunk_seconds,
        "chunks": len(chunks),
        "segments": len(all_segments),
        "elapsed_seconds": overall_elapsed,
        "log_file": str(log_file),
    }


# ---------------------------------------------------------------------------
# CLI local — só coleta argumentos/opções e chama transcribe_file()
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcreve japonês com faster-whisper (gera SRT). Tradução não é feita aqui."
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Arquivo de vídeo ou áudio. Se omitido, lista os arquivos da pasta atual pra escolher",
    )

    parser.add_argument(
        "--model",
        default=None,
        choices=["small", "medium", "large-v3"],
        help="Modelo Whisper. Se omitido, pergunta interativamente.",
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=600,
        help="Tamanho dos chunks em segundos. Padrão: 600 segundos / 10 minutos",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recria áudio, chunks e SRTs do zero",
    )

    return parser.parse_args()


def choose_input_file() -> str:
    candidates = sorted(
        p for p in Path(".").iterdir()
        if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
    )

    if not candidates:
        print("❌ Nenhum arquivo de áudio/vídeo (mp4, mp3, flac, opus, wav, m4a, aac, ogg, mkv, mov, avi, webm) encontrado na pasta atual.")
        raise SystemExit(1)

    print("\nArquivos encontrados na pasta atual:")
    for i, path in enumerate(candidates, start=1):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{i} - {path.name} ({size_mb:.1f} MB)")

    while True:
        choice = input(f"\nEscolha o arquivo (1-{len(candidates)}): ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return str(candidates[int(choice) - 1])

        print("❌ Opção inválida.")


def choose_model() -> str:
    while True:
        print("\nEscolha o modelo Whisper:")
        print("1 - small    [mais rápido]")
        print("2 - medium   [equilibrado / recomendado]")
        print("3 - large-v3 [melhor qualidade, bem mais lento]\n")

        choice = input("Digite 1, 2 ou 3 (Enter = medium): ").strip()

        if choice == "":
            return "medium"

        if choice == "1":
            return "small"

        if choice == "2":
            return "medium"

        if choice == "3":
            return "large-v3"

        print("❌ Opção inválida. Escolha 1, 2 ou 3.")


def main():
    args = parse_args()

    input_file = args.input_file or choose_input_file()

    if not os.path.exists(input_file):
        print(f"❌ Arquivo não encontrado: {input_file}")
        return

    model_name = args.model or choose_model()

    result = transcribe_file(
        input_file=input_file,
        model_name=model_name,
        chunk_seconds=args.chunk,
        force=args.force,
    )

    if result["srt_path"]:
        print(f"\n✅ SRT final: {result['srt_path']}")
    else:
        print("\n⚠️ Nenhum SRT foi gerado.")


if __name__ == "__main__":
    main()
