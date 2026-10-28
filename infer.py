"""
Concurrent inference via SGLang.

Two input modes are supported:
  1. Dataset images: pass --image_dir and each image is sent as one request.
  2. PDF pages: pass --pdf and each converted page is sent as one request.
"""

import argparse
import base64
import json
import math
import os
import secrets
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SERVED_MODEL_NAME = "Unlimited-OCR"
SERVER_URL = "http://127.0.0.1:10000"
HOST = "127.0.0.1"
PORT = 10000
SERVER_TIMEOUT = 300
PDF_DPI = 300
ATTENTION_BACKEND = "fa3"
PAGE_SIZE = 1
MEM_FRACTION_STATIC = 0.8
PROMPT = "document parsing."
TEMPERATURE = 0
CONTEXT_LENGTH = 32768
REQUEST_TIMEOUT = 1200
MAX_RETRIES = 5
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_PDF_PAGE_PIXELS = 20_000_000
MAX_PDF_TOTAL_PIXELS = 100_000_000
MAX_CONCURRENCY = 8


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> list[str]:
    import fitz

    doc = fitz.open(pdf_path)
    image_paths = []
    try:
        if len(doc) > MAX_PDF_PAGES:
            raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit")
        scale = dpi / 72
        mat = fitz.Matrix(scale, scale)
        total_pixels = 0
        for i, page in enumerate(doc):
            width = math.ceil(page.rect.width * scale)
            height = math.ceil(page.rect.height * scale)
            page_pixels = width * height
            if width <= 0 or height <= 0 or page_pixels > MAX_PDF_PAGE_PIXELS:
                raise ValueError(f"PDF page {i + 1} exceeds the pixel limit")
            total_pixels += page_pixels
            if total_pixels > MAX_PDF_TOTAL_PIXELS:
                raise ValueError("PDF exceeds the total pixel limit")
            out_path = os.path.join(output_dir, f"page_{i + 1:04d}.png")
            page.get_pixmap(matrix=mat).save(out_path)
            image_paths.append(out_path)
    finally:
        doc.close()
    return image_paths


def encode_image(image_path: str) -> dict:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        image = f.read(MAX_IMAGE_BYTES + 1)
    if len(image) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES}-byte limit")
    data = base64.b64encode(image).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_content(image_path: str) -> list[dict]:
    return [{"type": "text", "text": PROMPT}, encode_image(image_path)]


def server_ready(server_url: str, api_key: str) -> bool:
    session = requests.Session()
    session.trust_env = False
    try:
        resp = session.get(
            f"{server_url}/v1/models",
            headers={"Authorization": "Bearer " + api_key},
            timeout=5,
        )
        if resp.status_code != 200:
            return False
        return any(model.get("id") == SERVED_MODEL_NAME for model in resp.json().get("data", []))
    except (requests.RequestException, ValueError, AttributeError):
        return False
    finally:
        session.close()


def start_server(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.server_log)) or ".", exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.api_key = secrets.token_urlsafe(32)

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        args.model_dir,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--attention-backend",
        ATTENTION_BACKEND,
        "--page-size",
        str(PAGE_SIZE),
        "--mem-fraction-static",
        str(MEM_FRACTION_STATIC),
        "--context-length",
        str(CONTEXT_LENGTH),
        "--disable-overlap-schedule",
        "--skip-server-warmup",
        "--api-key",
        args.api_key,
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]

    print(f"Starting SGLang server on GPU {args.gpu}, port {PORT} ...")
    log_file = open(args.server_log, "w", encoding="utf-8")
    process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    process._log_file = log_file
    print(f"Server PID: {process.pid}")

    start = time.time()
    while time.time() - start < SERVER_TIMEOUT:
        if process.poll() is not None:
            log_file.flush()
            raise RuntimeError(f"SGLang server exited early. Check {args.server_log}")
        if server_ready(SERVER_URL, args.api_key):
            print(f"Server ready ({time.time() - start:.0f}s)")
            return process
        time.sleep(3)

    stop_server(process)
    raise TimeoutError(f"Timed out waiting for SGLang server. Check {args.server_log}")


def stop_server(process):
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    process._log_file.close()


def collect_stream_silent(resp, output_file: str | None) -> dict:
    chunks = []
    token_count = 0
    first_token_time = None
    f = open(output_file, "w", encoding="utf-8") if output_file else None
    try:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError):
                continue
            if not delta:
                continue
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1
            chunks.append(delta)
            if f:
                f.write(delta)
    finally:
        if f:
            f.close()

    end_time = time.time()
    decode_time = (end_time - first_token_time) if first_token_time and token_count > 1 else 0
    return {"tokens": token_count, "decode_time": decode_time, "text": "".join(chunks)}


def infer_one(image_path: str, output_file: str | None, args, idx: int) -> dict:
    name = ascii(os.path.basename(image_path))
    try:
        content = build_content(image_path)
    except Exception as e:
        print(f"  [{idx}] {name}: FAILED ({ascii(str(e))})")
        return {"tokens": 0, "decode_time": 0, "text": ""}

    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": TEMPERATURE,
        "skip_special_tokens": False,
        "stream": True,
        "images_config": {"image_mode": args.image_mode},
    }

    session = requests.Session()
    session.trust_env = False
    try:
        for attempt in range(MAX_RETRIES):
            try:
                resp = session.post(
                    f"{SERVER_URL}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + args.api_key,
                    },
                    data=json.dumps(payload),
                    timeout=REQUEST_TIMEOUT,
                    stream=True,
                )
                if resp.status_code == 502 and attempt < MAX_RETRIES - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                resp.raise_for_status()
                result = collect_stream_silent(resp, output_file)
                print(f"  [{idx}] {name}: {result['tokens']} tokens, {result['decode_time']:.1f}s")
                return result
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  [{idx}] {name}: retry {attempt + 1}/{MAX_RETRIES} ({ascii(str(e))})")
                    time.sleep(3 * (attempt + 1))
                    continue
                print(f"  [{idx}] {name}: FAILED ({ascii(str(e))})")
                return {"tokens": 0, "decode_time": 0, "text": ""}
    finally:
        session.close()


def collect_dataset_images(image_dir: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    image_files = []
    for root, _, files in os.walk(image_dir):
        for name in files:
            if name.lower().endswith(exts):
                image_files.append(os.path.join(root, name))
    return sorted(image_files, key=lambda f: os.path.getsize(f), reverse=True)


def build_jobs(args, temp_dir: str | None = None) -> list[tuple[str, str | None]]:
    if args.pdf:
        image_files = pdf_to_images(args.pdf, temp_dir, dpi=PDF_DPI)
        prefix = os.path.splitext(os.path.basename(args.pdf))[0]
        jobs = []
        for i, image_path in enumerate(image_files):
            output_file = None
            if args.output_dir:
                output_file = os.path.join(args.output_dir, f"{prefix}_page_{i + 1:04d}.md")
            jobs.append((image_path, output_file))
        return jobs

    if not args.image_dir:
        raise ValueError("Either --image_dir or --pdf is required")
    image_files = collect_dataset_images(args.image_dir)

    jobs = []
    for image_path in image_files:
        output_file = None
        if args.output_dir:
            rel = os.path.relpath(image_path, args.image_dir)
            stem = os.path.splitext(rel)[0].replace(os.sep, "__")
            output_file = os.path.join(args.output_dir, f"{stem}.md")
        jobs.append((image_path, output_file))
    return jobs


def run(args):
    temp_dir = tempfile.TemporaryDirectory(prefix="pdf_ocr_") if args.pdf else None
    try:
        jobs = build_jobs(args, temp_dir.name if temp_dir else None)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

        mode = "pdf_pages" if args.pdf else "dataset_images"
        concurrency = min(args.concurrency, MAX_CONCURRENCY)
        print(f"Mode: {mode}, requests={len(jobs)}, concurrency={concurrency}, image_mode={args.image_mode}")

        wall_start = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(infer_one, image_path, output_file, args, i + 1): image_path
                for i, (image_path, output_file) in enumerate(jobs)
            }
            for future in as_completed(futures):
                results.append(future.result())

        wall_time = time.time() - wall_start
        total_tokens = sum(r["tokens"] for r in results)
        successful = sum(1 for r in results if r["tokens"] > 0)
        print(f"\n{'=' * 60}")
        print("Concurrent Results:")
        print(f"  Requests: {successful}/{len(jobs)}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Wall time: {wall_time:.2f}s")
        if wall_time > 0:
            print(f"  System TPS: {total_tokens / wall_time:.2f} tokens/s")
        if successful > 0:
            avg_decode = sum(r["decode_time"] for r in results if r["tokens"] > 0) / successful
            avg_tokens = total_tokens / successful
            print(f"  Avg tokens/request: {avg_tokens:.0f}")
            print(f"  Avg decode_time/request: {avg_decode:.2f}s")
        print(f"{'=' * 60}")
    finally:
        if temp_dir:
            temp_dir.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(
        description="SGLang concurrent inference for image datasets or PDF pages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image_dir", default="", help="Directory of images for dataset concurrency mode")
    parser.add_argument("--pdf", default="", help="PDF file; each page is converted and sent as one concurrent request")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--concurrency", type=int, default=8, help="Maximum concurrent requests (capped at 8)")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model_dir", default="baidu/Unlimited-OCR")
    parser.add_argument("--image_mode", choices=("gundam", "base"), default="gundam")
    parser.add_argument("--server_log", default="./log/sglang_server.log")
    return parser.parse_args()


def main():
    args = parse_args()
    server_process = start_server(args)
    try:
        run(args)
    finally:
        stop_server(server_process)


if __name__ == "__main__":
    main()
