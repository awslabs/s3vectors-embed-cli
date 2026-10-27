#!/usr/bin/env python3
"""Gap validation for s3vectors-embed-cli PR #18 (request_builder / response_parser refactor).

Covers what run pr18-20261008 did not:

  G1  Bedrock-side failures reaching the async result parser (corrupt media)
  G2  Unsupported and odd media extensions (.tiff, .m4a, no extension, .PNG, .JPG)
  G3  Nova limits and segmentation arithmetic (sync > 30 s, request body > 25 MB,
      durationSeconds / maxLengthChars overrides verified from segment metadata,
      oversized SRC-CONTENT metadata)
  G4  Batch runs with a failing file, pooled-store boundaries, overrides inside
      batch workers
  G5  --filter combined with a nested override; query-side protected-field and
      flat-key behavior
  G6  Cross-account --src-bucket-owner (only when S3V_CROSS_ACCOUNT_* are set)
  G7  Video segment structure, AUDIO_VIDEO_SEPARATE (only when S3V_VIDEO_S3 is set)

Pass criteria are strict: every put is reconciled against the index by count
and metadata; segment runs must be contiguous (index 0..n-1, start[i+1] ==
end[i]) and non-final segments must have exactly the configured length, which
is what proves an override reached Bedrock; queries assert the identity of the
top hit; negative cases assert a non-zero exit, no traceback and nothing
stored. "Probe" cases send input the CLI cannot validate client-side (bad
extensions, oversized input); they pass when the CLI fails cleanly (no
traceback, state consistent with the exit code) and the report records what
Bedrock said, which is the point of running them.

The script generates its own text, PNG and WAV fixtures and uploads what the
async cases need. It creates the vector bucket and indexes if they do not
exist (with S3VECTORS-EMBED-SRC-CONTENT non-filterable, as the README asks).

Required environment
    S3V_REPO             path to the s3vectors-embed-cli checkout to test
                         (default: two directories up from this file)
    S3V_VECTOR_BUCKET    vector bucket name (created if missing)
    S3V_INPUT_BUCKET     ordinary S3 bucket; fixtures go under S3V_INPUT_PREFIX/<run>/
    S3V_ASYNC_OUTPUT     s3://bucket/prefix for async output; <run>/<case>/ is appended

Optional environment
    S3V_REGION               default us-east-1
    S3V_INPUT_PREFIX         default pr18gap
    S3V_INDEX_PREFIX         default pr18gap  (indexes <prefix>-nova, <prefix>-titan-v2, <prefix>-twelvelabs)
    S3V_NOVA_DIM             default 1024     (256 | 384 | 1024 | 3072)
    S3V_VIDEO_S3             s3:// URI of a video <= 30 s WITH an audio track (enables G7)
    S3V_VIDEO_DURATION       its duration in seconds (enables exact segment-count checks)
    S3V_JPEG                 local path or s3:// URI of a real JPEG (enables the .JPG case)
    S3V_CROSS_ACCOUNT_S3     s3:// URI of an image in a bucket owned by another account (enables G6)
    S3V_CROSS_ACCOUNT_OWNER  that account's 12-digit id
    S3V_CASE_TIMEOUT         seconds per CLI invocation, default 600
    S3V_PYTHON               interpreter for the CLI, default: this one

Usage
    python3 gap_validation.py [--out DIR] [--only g1,g3,...] [--cleanup] [--keep-fixtures]

Exit code 0 when every case passed or was skipped, 1 when any case failed,
2 on a setup problem.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import wave
import zlib
from typing import Any, Dict, List, Optional, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover
    sys.exit("boto3 is required (it is a dependency of the CLI under test)")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("S3V_REPO") or os.path.abspath(os.path.join(HERE, "..", ".."))
REGION = os.environ.get("S3V_REGION", "us-east-1")
VB = os.environ.get("S3V_VECTOR_BUCKET")
INPUT_BUCKET = os.environ.get("S3V_INPUT_BUCKET")
INPUT_PREFIX = os.environ.get("S3V_INPUT_PREFIX", "pr18gap").strip("/")
INDEX_PREFIX = os.environ.get("S3V_INDEX_PREFIX", "pr18gap")
NOVA_DIM = int(os.environ.get("S3V_NOVA_DIM", "1024"))
ASYNC_OUTPUT_ROOT = (os.environ.get("S3V_ASYNC_OUTPUT") or "").rstrip("/")
VIDEO_S3 = os.environ.get("S3V_VIDEO_S3")
VIDEO_DURATION = float(os.environ["S3V_VIDEO_DURATION"]) if os.environ.get("S3V_VIDEO_DURATION") else None
JPEG_SRC = os.environ.get("S3V_JPEG")
CROSS_S3 = os.environ.get("S3V_CROSS_ACCOUNT_S3")
CROSS_OWNER = os.environ.get("S3V_CROSS_ACCOUNT_OWNER")
CASE_TIMEOUT = int(os.environ.get("S3V_CASE_TIMEOUT", "600"))
PYTHON = os.environ.get("S3V_PYTHON") or sys.executable

NOVA = "amazon.nova-2-multimodal-embeddings-v1:0"
TITAN_V2 = "amazon.titan-embed-text-v2:0"
TL = "twelvelabs.marengo-embed-2-7-v1:0"
INDEX = {NOVA: f"{INDEX_PREFIX}-nova", TITAN_V2: f"{INDEX_PREFIX}-titan-v2", TL: f"{INDEX_PREFIX}-twelvelabs"}
DIM = {NOVA: NOVA_DIM, TITAN_V2: 1024, TL: 1024}

RUN = time.strftime("gap-%Y%m%d-%H%M") + "-" + uuid.uuid4().hex[:4]

META_TYPE = "S3VECTORS-EMBED-TYPE"
META_IDX = "S3VECTORS-EMBED-SEGMENT-IDX"
META_START = "S3VECTORS-EMBED-START-SEC"
META_END = "S3VECTORS-EMBED-END-SEC"
META_CSTART = "S3VECTORS-EMBED-START-CHAR-POS"
META_CEND = "S3VECTORS-EMBED-END-CHAR-POS"
META_SRC = "S3VECTORS-EMBED-SRC-LOCATION"
META_CONTENT = "S3VECTORS-EMBED-SRC-CONTENT"
META_TRUNC = "S3VECTORS-EMBED-TRUNCATED-CHAR-LEN"

s3 = boto3.client("s3", region_name=REGION)
s3v = boto3.client("s3vectors", region_name=REGION, endpoint_url=f"https://s3vectors.{REGION}.api.aws")


def die(msg: str) -> None:
    print(f"setup error: {msg}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Case framework
# ---------------------------------------------------------------------------


class Case:
    """One CLI invocation plus its assertions.

    kind: positive  - must exit 0; assertions on stored vectors / output
          negative  - must exit non-zero with a client-side message, nothing stored
          probe     - input the CLI cannot validate itself; passes when the CLI
                      fails cleanly and state matches the exit code; what
                      Bedrock said is recorded as an observation
    """

    def __init__(self, cid: str, group: str, kind: str, purpose: str, expect: str = ""):
        self.id, self.group, self.kind, self.purpose, self.expect = cid, group, kind, purpose, expect
        self.checks: List[Tuple[str, bool, str]] = []
        self.observations: List[Tuple[str, str]] = []
        self.command = ""
        self.exit_code: Optional[int] = None
        self.output = ""
        self.elapsed = 0.0
        self.payload: Any = None
        self.verdict = "PENDING"
        self.skip_reason = ""
        self.vectors: List[Dict[str, Any]] = []

    @property
    def prefix(self) -> str:
        return f"{RUN}/{self.id}/"

    def check(self, label: str, ok: Any, observed: Any = "") -> bool:
        self.checks.append((label, bool(ok), str(observed)))
        return bool(ok)

    def observe(self, label: str, value: Any) -> None:
        self.observations.append((label, str(value)))

    def finish(self) -> None:
        if self.verdict == "SKIP":
            return
        self.verdict = "PASS" if self.checks and all(ok for _, ok, _ in self.checks) else "FAIL"
        marker = {"PASS": "PASS", "FAIL": "FAIL"}[self.verdict]
        print(f"{marker}  {self.id}  ({self.elapsed:.0f}s)", flush=True)
        if self.verdict == "FAIL":
            for label, ok, observed in self.checks:
                if not ok:
                    print(f"      x {label}: {observed[:300]}", flush=True)

    def skip(self, reason: str) -> None:
        self.verdict, self.skip_reason = "SKIP", reason
        print(f"SKIP  {self.id}  ({reason})", flush=True)


CASES: List[Case] = []


def new_case(*args, **kwargs) -> Case:
    case = Case(*args, **kwargs)
    CASES.append(case)
    return case


# ---------------------------------------------------------------------------
# CLI runner and AWS helpers
# ---------------------------------------------------------------------------


def run_cli(case: Case, *args: str, timeout: int = CASE_TIMEOUT) -> None:
    cmd = [PYTHON, "-m", "s3vectors.cli", "--region", REGION, *args]
    env = dict(os.environ, PYTHONPATH=REPO, COLUMNS="10000", NO_COLOR="1", TERM="dumb", PYTHONWARNINGS="ignore")
    case.command = shlex.join(cmd)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
        stderr += f"\n[harness] killed after {timeout}s"
        code = -9
    case.elapsed = time.time() - t0
    case.exit_code = code
    case.output = stdout + ("\n--- stderr ---\n" + stderr if stderr.strip() else "")
    case.payload = extract_json(stdout)


def extract_json(text: str) -> Any:
    """Last JSON object printed at the start of a line (the CLI result)."""
    decoder = json.JSONDecoder()
    found = None
    for m in re.finditer(r"^\{", text, re.M):
        try:
            found, _ = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
    return found


def list_vectors(index: str, prefix: str) -> List[Dict[str, Any]]:
    out, token = [], None
    while True:
        kw: Dict[str, Any] = {"vectorBucketName": VB, "indexName": index, "returnMetadata": True, "maxResults": 500}
        if token:
            kw["nextToken"] = token
        resp = s3v.list_vectors(**kw)
        out.extend(v for v in resp.get("vectors", []) if v["key"].startswith(prefix))
        token = resp.get("nextToken")
        if not token:
            return out


def stored(case: Case, model: str, expect: Optional[int] = None) -> List[Dict[str, Any]]:
    """Vectors under the case prefix; retries briefly when the count is not yet the expected one."""
    for _ in range(4):
        case.vectors = list_vectors(INDEX[model], case.prefix)
        if expect is None or len(case.vectors) == expect:
            break
        time.sleep(2)
    return case.vectors


def parse_s3(uri: str) -> Tuple[str, str]:
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def count_objects(uri_prefix: str) -> int:
    bucket, key = parse_s3(uri_prefix)
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key):
        n += len(page.get("Contents", []))
    return n


def upload(local: str, key: str) -> str:
    full = f"{INPUT_PREFIX}/{RUN}/{key}"
    s3.upload_file(local, INPUT_BUCKET, full)
    return f"s3://{INPUT_BUCKET}/{full}"


def async_out(case: Case) -> str:
    return f"{ASYNC_OUTPUT_ROOT}/{RUN}/{case.id}/"


def params(obj: Any) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------


def expect_success(case: Case, model: str, n_vectors: Optional[int] = None) -> List[Dict[str, Any]]:
    case.check("exit code 0", case.exit_code == 0, case.exit_code)
    case.check("no traceback in output", "Traceback" not in case.output)
    vectors = stored(case, model, n_vectors)
    if n_vectors is not None:
        case.check(f"index holds {n_vectors} vector(s) under the case prefix", len(vectors) == n_vectors, len(vectors))
    reported = None
    if isinstance(case.payload, dict):
        reported = case.payload.get("totalVectors", 1 if "key" in case.payload else None)
    if reported is not None:
        case.check("CLI-reported vector count matches the index", reported == len(vectors), f"reported {reported}, index {len(vectors)}")
    return vectors


def expect_client_rejection(case: Case, model: Optional[str], message: str, output_uri: Optional[str] = None) -> None:
    case.check("exit code non-zero", case.exit_code not in (0, None), case.exit_code)
    case.check("no traceback in output", "Traceback" not in case.output)
    case.check(f"message contains {message!r}", message in case.output, case.output.strip()[-400:])
    if model:
        case.check("nothing stored in the index", len(stored(case, model, 0)) == 0, len(case.vectors))
    if output_uri:
        n = count_objects(output_uri)
        case.check("nothing written to the async output prefix", n == 0, n)


def expect_clean_outcome(case: Case, model: Optional[str], expected: str, output_uri: Optional[str] = None) -> None:
    """Probe cases: the CLI must fail or succeed cleanly; what Bedrock did is recorded."""
    case.check("no traceback in output", "Traceback" not in case.output)
    case.check("CLI did not hang (finished inside the case timeout)", case.exit_code != -9, case.exit_code)
    vectors = stored(case, model) if model else []
    if case.exit_code == 0:
        case.check("exit 0 and vectors were stored", len(vectors) >= 1, len(vectors))
        outcome = "accepted"
    else:
        case.check("exit non-zero and nothing stored", len(vectors) == 0, len(vectors))
        case.check("an error message was printed", "Error" in case.output or "error" in case.output.lower())
        outcome = "rejected"
    case.observe("outcome", f"{outcome} (expected: {expected})")
    error_lines = [l for l in case.output.splitlines() if l.startswith("Error")]
    if error_lines:
        case.observe("error message", error_lines[-1][:600])
    if output_uri:
        case.observe("objects in async output prefix", count_objects(output_uri))


def check_segments(case: Case, vectors: List[Dict[str, Any]], *, start_key: str, end_key: str,
                   length: Optional[float], length_mode: str = "eq", total: Optional[float] = None,
                   tol: float = 0.06, label: str = "") -> None:
    """Contiguity and length assertions over a run of segment vectors."""
    tag = f"[{label}] " if label else ""
    if not vectors:
        case.check(f"{tag}at least one segment to inspect", False, "no vectors")
        return
    vs = sorted(vectors, key=lambda v: v["metadata"].get(META_IDX, -1))
    idxs = [v["metadata"].get(META_IDX) for v in vs]
    case.check(f"{tag}segment indexes are 0..n-1", idxs == list(range(len(vs))), idxs)
    starts = [v["metadata"].get(start_key) for v in vs]
    ends = [v["metadata"].get(end_key) for v in vs]
    if not case.check(f"{tag}every segment carries {start_key}/{end_key}",
                      all(s is not None and e is not None for s, e in zip(starts, ends)), list(zip(starts, ends))):
        return
    case.check(f"{tag}first segment starts at 0", abs(starts[0]) <= tol, starts[0])
    gaps = [abs(starts[i + 1] - ends[i]) for i in range(len(vs) - 1)]
    case.check(f"{tag}segments are continuous (start[i+1] == end[i])", all(g <= tol for g in gaps), gaps)
    lens = [e - s for s, e in zip(starts, ends)]
    if length is not None and len(vs) > 1:
        body = lens[:-1]
        if length_mode == "eq":
            case.check(f"{tag}non-final segments are exactly {length} long (configured value reached Bedrock)",
                       all(abs(l - length) <= tol for l in body), body)
        else:
            case.check(f"{tag}every segment is <= {length} long (configured value reached Bedrock)",
                       all(l <= length + tol for l in lens), lens)
    if total is not None:
        # Bedrock may round the final media boundary up to the next whole second (a 6.0 s clip reported 0-5, 5-7).
        slack = max(tol, 0.6) if length is None or length_mode == "le" else max(tol, 1.1)
        case.check(f"{tag}last segment ends at the input length ({total}, +/- {slack})", abs(ends[-1] - total) <= slack, ends[-1])
        if length is not None and length_mode == "eq":
            expected = math.ceil(total / length)
            case.check(f"{tag}segment count == ceil({total}/{length}) == {expected}", len(vs) == expected, len(vs))
        if length is not None and length_mode == "le":
            minimum = math.ceil(total / length)
            case.check(f"{tag}segment count >= ceil({total}/{length}) == {minimum}", len(vs) >= minimum, len(vs))
    case.observe("segments", [(round(s, 2), round(e, 2)) for s, e in zip(starts, ends)])


def batch_summary(case: Case) -> Dict[str, Any]:
    p = case.payload if isinstance(case.payload, dict) else {}
    return {k: p.get(k) for k in ("totalFiles", "processedFiles", "failedFiles", "totalVectors", "errors")}


def stored_batch_sizes(case: Case) -> List[int]:
    return sorted(int(m) for m in re.findall(r"STORED BATCH: (\d+) vectors", case.output))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SENTENCES = {
    "t1": "Kangaroos are marsupials that move by hopping on powerful hind legs.",
    "t2": "The espresso machine heats water to just below boiling before forcing it through the grounds.",
    "t3": "Rockets launch satellites into orbit above the atmosphere.",
    "t4": "Old-growth forests store carbon in trunks, roots and soil for centuries.",
    "t5": "Desert sand dunes migrate slowly downwind, a few meters every year.",
}
OCEAN_A = "The ocean covers more than seventy percent of the planet and regulates climate through currents."
OCEAN_B = "Deep ocean trenches reach depths of nearly eleven kilometers below the surface."
SPACE = "Rockets launch satellites into orbit above the atmosphere."
FILLER = ("Amazon Nova multimodal embeddings turn text, images, video and audio into vectors for semantic search. "
          "This sentence exists to fill the test document with realistic prose for segmentation testing. ")


def write_text(path: str, n_chars: int) -> None:
    text = (FILLER * (n_chars // len(FILLER) + 1))[:n_chars]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def png_bytes(w: int, h: int, seed: int, level: int = 9) -> bytes:
    rnd = random.Random(seed)
    r0, g0, b0 = (rnd.randrange(256) for _ in range(3))
    rows = []
    for y in range(h):
        row = bytearray(b"\x00")
        for x in range(w):
            row += bytes(((r0 + x * 255 // w) & 255, (g0 + y * 255 // h) & 255, (b0 + (x + y) * 127 // (w + h)) & 255))
        rows.append(bytes(row))
    return _png(w, h, b"".join(rows), level)


def noise_png_bytes(w: int, h: int) -> bytes:
    """Incompressible image, stored uncompressed, so the file is ~ w*h*3 bytes."""
    raw = b"".join(b"\x00" + os.urandom(w * 3) for _ in range(h))
    return _png(w, h, raw, 0)


def _png(w: int, h: int, raw: bytes, level: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, level)) + chunk(b"IEND", b""))


def write_wav(path: str, seconds: float, rate: int = 16000) -> None:
    """Mono 16-bit tone with slow amplitude modulation so segments differ."""
    n = int(seconds * rate)
    frames = bytearray()
    two_pi = 2 * math.pi
    for i in range(n):
        env = 0.55 + 0.45 * math.sin(two_pi * 0.2 * i / rate)
        frames += struct.pack("<h", int(11000 * env * math.sin(two_pi * 440 * i / rate)))
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(bytes(frames))


def write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


class Fixtures:
    def __init__(self, root: str):
        self.root = root
        self.local: Dict[str, str] = {}
        self.s3: Dict[str, str] = {}

    def path(self, *parts: str) -> str:
        p = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def build(self) -> None:
        print("generating fixtures ...", flush=True)
        L = self.local
        good = [png_bytes(256, 256, seed) for seed in (1, 2, 3)]
        for i, data in enumerate(good, 1):
            write_bytes(self.path("img", f"good-{i}.png"), data)
            L[f"good-{i}.png"] = self.path("img", f"good-{i}.png")
        corrupt_png = b"\x89PNG\r\n\x1a\n" + os.urandom(4096)
        write_bytes(self.path("img", "corrupt.png"), corrupt_png)
        L["corrupt.png"] = self.path("img", "corrupt.png")
        write_bytes(self.path("odd", "Logo.PNG"), good[0]);               L["Logo.PNG"] = self.path("odd", "Logo.PNG")
        write_bytes(self.path("odd", "noext"), good[0]);                  L["noext"] = self.path("odd", "noext")
        write_bytes(self.path("odd", "scan.tiff"), good[0]);              L["scan.tiff"] = self.path("odd", "scan.tiff")
        write_bytes(self.path("big", "noise-28mb.png"), noise_png_bytes(3050, 3050))
        L["noise-28mb.png"] = self.path("big", "noise-28mb.png")

        write_wav(self.path("aud", "tone-35s.wav"), 35.0);                L["tone-35s.wav"] = self.path("aud", "tone-35s.wav")
        write_wav(self.path("aud", "tone-5s.wav"), 5.0);                  L["tone-5s.wav"] = self.path("aud", "tone-5s.wav")
        shutil.copy(L["tone-5s.wav"], self.path("odd", "voice.m4a"));     L["voice.m4a"] = self.path("odd", "voice.m4a")
        write_bytes(self.path("aud", "corrupt.wav"), b"RIFF" + os.urandom(8192))
        L["corrupt.wav"] = self.path("aud", "corrupt.wav")
        write_bytes(self.path("vid", "corrupt.mp4"), b"\x00\x00\x00\x18ftypmp42" + os.urandom(8192))
        L["corrupt.mp4"] = self.path("vid", "corrupt.mp4")

        for name, sentence in SENTENCES.items():
            p = self.path("txt5", f"{name}.txt")
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(sentence)
            L[f"{name}.txt"] = p
        write_text(self.path("long", "long-9k.txt"), 9000);              L["long-9k.txt"] = self.path("long", "long-9k.txt")
        write_text(self.path("long", "long-20k.txt"), 20000);            L["long-20k.txt"] = self.path("long", "long-20k.txt")
        write_text(self.path("long", "long-45k.txt"), 45000);            L["long-45k.txt"] = self.path("long", "long-45k.txt")

        # Batch directories
        mixed = self.path("batch-mixed", "x")
        os.remove(mixed) if os.path.exists(mixed) else None
        for name, data in (("good-1.png", good[0]), ("good-2.png", good[1]), ("scan.tiff", good[2]),
                           ("chart.bmp", good[2]), ("notes.txt", b"not an image")):
            write_bytes(self.path("batch-mixed", name), data)
        L["batch-mixed/"] = os.path.dirname(mixed)
        for i, data in enumerate(good, 1):
            write_bytes(self.path("batch-corrupt", f"good-{i}.png"), data)
        write_bytes(self.path("batch-corrupt", "corrupt.png"), corrupt_png)
        L["batch-corrupt/"] = os.path.dirname(self.path("batch-corrupt", "x"))
        L["txt5/"] = os.path.dirname(L["t1.txt"])

        # Optional JPEG
        self.jpeg = None
        if JPEG_SRC:
            dest = self.path("odd", "Photo.JPG")
            if JPEG_SRC.startswith("s3://"):
                b, k = parse_s3(JPEG_SRC)
                s3.download_file(b, k, dest)
            else:
                shutil.copy(JPEG_SRC, dest)
            self.jpeg = dest
        else:
            try:
                from PIL import Image  # type: ignore
                img = Image.new("RGB", (256, 256))
                img.putdata([((x * 255) // 256, (y * 255) // 256, 128) for y in range(256) for x in range(256)])
                dest = self.path("odd", "Photo.JPG")
                img.save(dest, "JPEG", quality=85)
                self.jpeg = dest
            except Exception:
                self.jpeg = None

        print("uploading fixtures ...", flush=True)
        S = self.s3
        S["tone-35s.wav"] = upload(L["tone-35s.wav"], "audio/tone-35s.wav")
        S["corrupt.mp4"] = upload(L["corrupt.mp4"], "video/corrupt.mp4")
        S["corrupt.png"] = upload(L["corrupt.png"], "images/corrupt.png")
        S["good-1.png"] = upload(L["good-1.png"], "images/good-1.png")
        S["long-20k.txt"] = upload(L["long-20k.txt"], "text/long-20k.txt")
        S["long-45k.txt"] = upload(L["long-45k.txt"], "text/long-45k.txt")
        for name in SENTENCES:
            S[f"{name}.txt"] = upload(L[f"{name}.txt"], f"txt5/{name}.txt")
        S["txt5/"] = f"s3://{INPUT_BUCKET}/{INPUT_PREFIX}/{RUN}/txt5/"
        upload(L["tone-35s.wav"], "audio-batch/good-35s.wav")
        upload(L["corrupt.wav"], "audio-batch/corrupt.wav")
        S["audio-batch/"] = f"s3://{INPUT_BUCKET}/{INPUT_PREFIX}/{RUN}/audio-batch/"


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

created_indexes: List[str] = []
created_bucket = False


def ensure_infra() -> None:
    global created_bucket
    try:
        s3v.get_vector_bucket(vectorBucketName=VB)
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("NotFoundException", "ResourceNotFoundException", "404"):
            raise
        s3v.create_vector_bucket(vectorBucketName=VB)
        created_bucket = True
        print(f"created vector bucket {VB}")
    for model, index in INDEX.items():
        try:
            info = s3v.get_index(vectorBucketName=VB, indexName=index)["index"]
            if info["dimension"] != DIM[model]:
                die(f"index {index} exists with dimension {info['dimension']}, need {DIM[model]}; "
                    f"set S3V_INDEX_PREFIX to a fresh value or S3V_NOVA_DIM to match")
            nonfilt = info.get("metadataConfiguration", {}).get("nonFilterableMetadataKeys", [])
            if META_CONTENT not in nonfilt:
                print(f"warning: {index} does not mark {META_CONTENT} non-filterable; long-text cases may fail on the 2 KB filterable limit")
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("NotFoundException", "ResourceNotFoundException", "404"):
                raise
            s3v.create_index(vectorBucketName=VB, indexName=index, dataType="float32", dimension=DIM[model],
                             distanceMetric="cosine",
                             metadataConfiguration={"nonFilterableMetadataKeys": [META_CONTENT]})
            created_indexes.append(index)
            print(f"created index {index} (dimension {DIM[model]}, {META_CONTENT} non-filterable)")
    s3.head_bucket(Bucket=INPUT_BUCKET)
    s3.head_bucket(Bucket=parse_s3(ASYNC_OUTPUT_ROOT)[0])


def cleanup(fx: Fixtures, keep_fixtures: bool, delete_infra: bool) -> None:
    for root in (f"s3://{INPUT_BUCKET}/{INPUT_PREFIX}/{RUN}/", f"{ASYNC_OUTPUT_ROOT}/{RUN}/"):
        bucket, key = parse_s3(root)
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objs})
        print(f"deleted objects under {root}")
    if delete_infra:
        for index in created_indexes:
            s3v.delete_index(vectorBucketName=VB, indexName=index)
            print(f"deleted index {index}")
        if created_bucket and len(created_indexes) == len(INDEX):
            s3v.delete_vector_bucket(vectorBucketName=VB)
            print(f"deleted vector bucket {VB}")
    if not keep_fixtures:
        shutil.rmtree(fx.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def put_args(model: str, case: Case, *rest: str) -> List[str]:
    return ["put", "--vector-bucket-name", VB, "--index-name", INDEX[model], "--model-id", model,
            "--key-prefix", case.prefix, *rest]


def query_args(model: str, *rest: str) -> List[str]:
    return ["query", "--vector-bucket-name", VB, "--index-name", INDEX[model], "--model-id", model,
            "--return-distance", *rest]


def group_g1(fx: Fixtures) -> None:
    """Bedrock-side failures: the async parser must fail closed, errors must be readable."""
    c = new_case("g1-nova-async-corrupt-video", "G1", "probe",
                 "Corrupt .mp4 (valid ftyp header, random payload) submitted async. Exercises the parser's "
                 "fail-closed path (manifest FAILURE / failureReason) or the job-level Failed path; records which.",
                 expect="rejected")
    run_cli(c, *put_args(NOVA, c, "--video", fx.s3["corrupt.mp4"], "--async-output-s3-uri", async_out(c)))
    expect_clean_outcome(c, NOVA, "rejected", async_out(c))
    c.check("error names the async job or a failure reason", "Async embedding" in c.output or "failed" in c.output.lower())
    c.observe("failure path", classify_async_failure(c.output))
    c.finish()

    c = new_case("g1-nova-async-corrupt-image", "G1", "probe",
                 "Corrupt .png submitted async; same purpose as the video case for the image modality.",
                 expect="rejected")
    run_cli(c, *put_args(NOVA, c, "--image", fx.s3["corrupt.png"], "--async-output-s3-uri", async_out(c)))
    expect_clean_outcome(c, NOVA, "rejected", async_out(c))
    c.observe("failure path", classify_async_failure(c.output))
    c.finish()

    c = new_case("g1-nova-sync-corrupt-image-local", "G1", "negative",
                 "Corrupt .png through the sync path: Bedrock ValidationException must surface as a clean "
                 "'Bedrock embedding failed' error with nothing stored (baseline for comparing error quality).")
    run_cli(c, *put_args(NOVA, c, "--image", fx.local["corrupt.png"]))
    expect_client_rejection(c, NOVA, "Bedrock embedding failed")
    c.finish()

    c = new_case("g1-tl-async-corrupt-image", "G1", "probe",
                 "TwelveLabs with a corrupt image: the job should fail; the CLI must report Bedrock's failure "
                 "message and store nothing (unchanged flow from main).", expect="rejected")
    run_cli(c, *put_args(TL, c, "--image", fx.s3["corrupt.png"], "--async-output-s3-uri", async_out(c)))
    expect_clean_outcome(c, TL, "rejected", async_out(c))
    c.observe("failure path", classify_async_failure(c.output))
    c.finish()


def classify_async_failure(output: str) -> str:
    if "failed result(s)" in output or "reason=" in output or "failed segment(s)" in output:
        return "manifest-level: parser raised from a FAILURE entry / failed segment (fail-closed path exercised)"
    if "Async embedding failed:" in output:
        return "job-level: Bedrock reported the job Failed before any manifest was read"
    if "ValidationException" in output:
        return "submit-level: StartAsyncInvoke rejected the request"
    return "none / unexpected (see output)"


def group_g2(fx: Fixtures) -> None:
    """Extensions the builder derives 'format' from."""
    for cid, flag, key, note in (
        ("g2-nova-tiff-image-local", "--image", "scan.tiff", "PNG bytes named .tiff -> builder sends format 'tiff'"),
        ("g2-nova-m4a-audio-local", "--audio", "voice.m4a", "WAV bytes named .m4a -> builder sends format 'm4a'"),
        ("g2-nova-noext-image-local", "--image", "noext", "file without extension -> builder sends format ''"),
    ):
        c = new_case(cid, "G2", "probe",
                     f"{note}. Nova does not list this format; expect a Bedrock rejection. The recorded message "
                     "tells us whether a client-side format check belongs in the builder.", expect="rejected")
        run_cli(c, *put_args(NOVA, c, flag, fx.local[key]))
        expect_clean_outcome(c, NOVA, "rejected")
        c.finish()

    c = new_case("g2-nova-upper-png-local", "G2", "positive",
                 "Uppercase extension Logo.PNG: builder must lowercase to format 'png' (visible in --debug payload) "
                 "and the put must succeed with a filename-derived key.")
    run_cli(c, "--debug", *put_args(NOVA, c, "--image", fx.local["Logo.PNG"], "--filename-as-key"))
    vectors = expect_success(c, NOVA, 1)
    c.check('payload carries "format": "png"', '"format": "png"' in c.output)
    if vectors:
        c.check("key is <prefix>Logo.PNG", vectors[0]["key"] == c.prefix + "Logo.PNG", vectors[0]["key"])
        c.check("metadata TYPE=IMAGE", vectors[0]["metadata"].get(META_TYPE) == "IMAGE", vectors[0]["metadata"].get(META_TYPE))
    c.finish()

    c = new_case("g2-nova-upper-jpg-local", "G2", "positive",
                 "Real JPEG named Photo.JPG: builder must map jpg -> 'jpeg' (visible in --debug payload).")
    if not fx.jpeg:
        c.skip("no JPEG available: set S3V_JPEG or install Pillow")
    else:
        run_cli(c, "--debug", *put_args(NOVA, c, "--image", fx.jpeg))
        expect_success(c, NOVA, 1)
        c.check('payload carries "format": "jpeg"', '"format": "jpeg"' in c.output)
        c.finish()

    c = new_case("g2-nova-batch-image-mixed-extensions", "G2", "positive",
                 "Image batch over a directory holding good-1.png, good-2.png, scan.tiff, chart.bmp and notes.txt. "
                 "The orchestrator's extension filter admits tiff/bmp and (being a union over all modalities) "
                 "txt, so three files reach Bedrock as images and fail there. Expect 2 processed, 3 failed, "
                 "3 named errors, the run still exits 0 and the 2 good vectors are stored.")
    run_cli(c, *put_args(NOVA, c, "--image", os.path.join(fx.local["batch-mixed/"], "*"), "--filename-as-key"))
    expect_success(c, NOVA, 2)
    s = batch_summary(c)
    c.check("processedFiles == 2", s["processedFiles"] == 2, s)
    c.check("failedFiles == 3", s["failedFiles"] == 3, s)
    errors = s.get("errors") or []
    c.check("one error per failed file, naming it", all(any(n in e for e in errors) for n in ("scan.tiff", "chart.bmp", "notes.txt")) and len(errors) == 3, errors)
    c.observe("errors", errors)
    c.finish()


def group_g3(fx: Fixtures) -> None:
    """Nova limits and segmentation arithmetic."""
    c = new_case("g3-nova-sync-audio-35s-local", "G3", "probe",
                 "35 s WAV through the sync path (documented sync limit 30 s). Nothing client-side enforces it; "
                 "expect Bedrock to reject. Records the message a user would see.", expect="rejected")
    run_cli(c, *put_args(NOVA, c, "--audio", fx.local["tone-35s.wav"]))
    expect_clean_outcome(c, NOVA, "rejected")
    c.finish()

    c = new_case("g3-nova-sync-image-28mb-local", "G3", "probe",
                 "~28 MB uncompressed PNG inline (base64 ~37 MB, over the 25 MB InvokeModel body limit). "
                 "Expect a rejection; records whether the error is usable or an opaque transport failure.",
                 expect="rejected")
    run_cli(c, *put_args(NOVA, c, "--image", fx.local["noise-28mb.png"]))
    expect_clean_outcome(c, NOVA, "rejected")
    c.finish()

    c = new_case("g3-nova-sync-text-9k-local", "G3", "probe",
                 "9,000-char text inline with the default truncationMode END. Documents what Nova does past the "
                 "documented 8192-char value limit: truncate (TRUNCATED-CHAR-LEN in metadata) or reject.",
                 expect="accepted (truncated) or rejected")
    run_cli(c, *put_args(NOVA, c, "--text", fx.local["long-9k.txt"]))
    expect_clean_outcome(c, NOVA, "accepted or rejected")
    if c.vectors:
        c.observe(META_TRUNC, c.vectors[0]["metadata"].get(META_TRUNC, "absent"))
    c.finish()

    c = new_case("g3-nova-async-audio-35s-default", "G3", "positive",
                 "35 s WAV async with the CLI default segmentationConfig {durationSeconds: 5} sent explicitly. "
                 "Expect exactly 7 contiguous segments of 5 s, the last ending at 35 s.")
    run_cli(c, *put_args(NOVA, c, "--audio", fx.s3["tone-35s.wav"], "--async-output-s3-uri", async_out(c)))
    vectors = expect_success(c, NOVA, 7)
    check_segments(c, vectors, start_key=META_START, end_key=META_END, length=5, total=35.0)
    c.check("all vectors typed AUDIO", all(v["metadata"].get(META_TYPE) == "AUDIO" for v in vectors))
    c.finish()

    c = new_case("g3-nova-async-audio-35s-duration-10", "G3", "positive",
                 "Same file with nested override segmentedEmbeddingParams.audio.segmentationConfig.durationSeconds=10. "
                 "Expect 4 segments with non-final length exactly 10 s: the override reached Bedrock.")
    run_cli(c, *put_args(NOVA, c, "--audio", fx.s3["tone-35s.wav"], "--async-output-s3-uri", async_out(c),
                         "--bedrock-inference-params",
                         params({"segmentedEmbeddingParams": {"audio": {"segmentationConfig": {"durationSeconds": 10}}}})))
    vectors = expect_success(c, NOVA, 4)
    check_segments(c, vectors, start_key=META_START, end_key=META_END, length=10, total=35.0)
    c.finish()

    c = new_case("g3-nova-async-text-20k-default", "G3", "positive",
                 "20,000-char text async with the empty segmentationConfig wrapper, so Bedrock applies its own "
                 "maxLengthChars default. Expect contiguous segments covering chars 0..20000, every vector carrying "
                 "the full 20,000-char SRC-CONTENT. The observed default segment length is recorded (a self-run "
                 "measured ~8192 chars, cut at word boundaries, not the 32000 noted in the pr18-20261008 report).")
    run_cli(c, *put_args(NOVA, c, "--text", fx.s3["long-20k.txt"], "--async-output-s3-uri", async_out(c)))
    vectors = expect_success(c, NOVA)
    c.check("at least one segment", len(vectors) >= 1, len(vectors))
    check_segments(c, vectors, start_key=META_CSTART, end_key=META_CEND, length=None, total=20000, tol=0)
    c.check("every vector carries the full 20,000-char SRC-CONTENT",
            vectors and all(len(v["metadata"].get(META_CONTENT, "")) == 20000 for v in vectors),
            [len(v["metadata"].get(META_CONTENT, "")) for v in vectors])
    c.check("TYPE=TEXT on every vector", vectors and all(v["metadata"].get(META_TYPE) == "TEXT" for v in vectors))
    if len(vectors) > 1:
        first = min(vectors, key=lambda v: v["metadata"].get(META_IDX, 0))["metadata"]
        c.observe("Bedrock default segment length (chars)", first.get(META_CEND))
    c.finish()

    c = new_case("g3-nova-async-text-20k-maxlen-8000", "G3", "positive",
                 "Same text with nested override segmentedEmbeddingParams.text.segmentationConfig.maxLengthChars=8000. "
                 "Expect >= 3 contiguous segments, each <= 8000 chars, covering 0..20000: the override reached Bedrock.")
    run_cli(c, *put_args(NOVA, c, "--text", fx.s3["long-20k.txt"], "--async-output-s3-uri", async_out(c),
                         "--bedrock-inference-params",
                         params({"segmentedEmbeddingParams": {"text": {"segmentationConfig": {"maxLengthChars": 8000}}}})))
    vectors = expect_success(c, NOVA)
    c.check("at least 3 segments", len(vectors) >= 3, len(vectors))
    check_segments(c, vectors, start_key=META_CSTART, end_key=META_CEND, length=8000, length_mode="le", total=20000, tol=0)
    c.finish()

    c = new_case("g3-nova-async-text-45k-metadata-limit", "G3", "probe",
                 "45,000-char text async. The embedding job should succeed, but the CLI copies the whole text into "
                 "S3VECTORS-EMBED-SRC-CONTENT on every vector, exceeding the 40 KB per-vector metadata limit, so "
                 "PutVectors should fail after the job already ran. Pre-existing on main; documents the behavior.",
                 expect="rejected at PutVectors")
    run_cli(c, *put_args(NOVA, c, "--text", fx.s3["long-45k.txt"], "--async-output-s3-uri", async_out(c)))
    expect_clean_outcome(c, NOVA, "rejected at PutVectors (metadata too large)", async_out(c))
    c.observe("failure stage", "PutVectors" if "put_vectors" in c.output else ("Bedrock" if "embedding failed" in c.output.lower() else "see output"))
    c.finish()


def group_g4(fx: Fixtures) -> None:
    """Batch robustness."""
    c = new_case("g4-nova-batch-image-one-corrupt-batchsize-2", "G4", "positive",
                 "Local image batch: 3 good PNGs + corrupt.png, --batch-size 2. Expect 3 processed, 1 failed with "
                 "an error naming corrupt.png, 3 vectors in the index, and the pooled store flushed as 2 + 1.")
    run_cli(c, *put_args(NOVA, c, "--image", os.path.join(fx.local["batch-corrupt/"], "*"), "--filename-as-key",
                         "--batch-size", "2", "--max-workers", "2"))
    expect_success(c, NOVA, 3)
    s = batch_summary(c)
    c.check("processedFiles == 3 and failedFiles == 1", (s["processedFiles"], s["failedFiles"]) == (3, 1), s)
    c.check("error names corrupt.png", any("corrupt.png" in e for e in (s.get("errors") or [])), s.get("errors"))
    c.check("pooled store flushed as [1, 2]", stored_batch_sizes(c) == [1, 2], stored_batch_sizes(c))
    c.finish()

    c = new_case("g4-titan-v2-batch-s3-batchsize-2", "G4", "positive",
                 "S3 text batch of 5 objects with --batch-size 2 on Titan v2 (flat model, unchanged payload). "
                 "Expect 5 vectors keyed by filename and the pooled store flushed as 2 + 2 + 1.")
    run_cli(c, *put_args(TITAN_V2, c, "--text", fx.s3["txt5/"] + "*", "--filename-as-key", "--batch-size", "2"))
    vectors = expect_success(c, TITAN_V2, 5)
    c.check("keys are t1..t5.txt", sorted(v["key"][len(c.prefix):] for v in vectors) == [f"t{i}.txt" for i in range(1, 6)],
            sorted(v["key"] for v in vectors))
    c.check("pooled store flushed as [1, 2, 2]", stored_batch_sizes(c) == [1, 2, 2], stored_batch_sizes(c))
    c.finish()

    c = new_case("g4-nova-async-audio-batch-duration-10-one-corrupt", "G4", "positive",
                 "S3 audio batch (per-file storage path): good-35s.wav + corrupt.wav, async, with the nested "
                 "durationSeconds=10 override. Expect 1 processed / 1 failed, 4 vectors (10 s segments, so the "
                 "override was applied inside the batch worker), error naming corrupt.wav.")
    run_cli(c, *put_args(NOVA, c, "--audio", fx.s3["audio-batch/"] + "*", "--async-output-s3-uri", async_out(c),
                         "--bedrock-inference-params",
                         params({"segmentedEmbeddingParams": {"audio": {"segmentationConfig": {"durationSeconds": 10}}}})))
    vectors = expect_success(c, NOVA, 4)
    s = batch_summary(c)
    c.check("processedFiles == 1 and failedFiles == 1", (s["processedFiles"], s["failedFiles"]) == (1, 1), s)
    c.check("error names corrupt.wav", any("corrupt.wav" in e for e in (s.get("errors") or [])), s.get("errors"))
    check_segments(c, vectors, start_key=META_START, end_key=META_END, length=10, total=35.0)
    c.observe("failure path for corrupt.wav", classify_async_failure(" ".join(s.get("errors") or [])))
    c.finish()

    c = new_case("g4-nova-batch-text-override-in-every-worker", "G4", "positive",
                 "Local text batch of 5 with --debug and nested override truncationMode=NONE. Expect 5 vectors and "
                 "the override present in all 5 request payloads printed by the workers.")
    run_cli(c, "--debug", *put_args(NOVA, c, "--text", os.path.join(fx.local["txt5/"], "*"), "--filename-as-key",
                                    "--bedrock-inference-params",
                                    params({"singleEmbeddingParams": {"text": {"truncationMode": "NONE"}}})))
    expect_success(c, NOVA, 5)
    n = c.output.count('"truncationMode": "NONE"')
    c.check('"truncationMode": "NONE" appears in 5 payloads', n == 5, n)
    c.check("no payload still carries the default END", '"truncationMode": "END"' not in c.output)
    c.finish()


def group_g5(fx: Fixtures) -> None:
    """Query: filter plus nested override, protected fields, flat keys."""
    keys = {}
    for cid, key, text, category in (("g5-put-ocean-a", "ocean-a", OCEAN_A, "ocean"),
                                      ("g5-put-ocean-b", "ocean-b", OCEAN_B, "ocean"),
                                      ("g5-put-space", "space", SPACE, "space")):
        c = new_case(cid, "G5", "positive", f"Seed vector '{key}' with --metadata category={category} (plus run id) and a fixed --key.")
        run_cli(c, *put_args(NOVA, c, "--text-value", text, "--key", key, "--metadata", params({"category": category, "run": RUN})))
        vectors = expect_success(c, NOVA, 1)
        if vectors:
            c.check("key is <prefix><key>", vectors[0]["key"] == c.prefix + key, vectors[0]["key"])
            c.check("category stored", vectors[0]["metadata"].get("category") == category, vectors[0]["metadata"])
            keys[key] = vectors[0]["key"]
        c.finish()

    c = new_case("g5-nova-query-exact-roundtrip", "G5", "positive",
                 "Query with the exact ocean-a text, no filter. ocean-a must be the top hit and clearly ahead of "
                 "ocean-b. Distance is not ~0: the CLI embeds puts with embeddingPurpose GENERIC_INDEX and queries "
                 "with TEXT_RETRIEVAL, so identical text yields different vectors by design (observed ~0.2).")
    # Scope to this run: a shared index may hold identical seeds from earlier runs (exact ties).
    run_cli(c, *query_args(NOVA, "--text-value", OCEAN_A, "--k", "3", "--filter", params({"run": {"$eq": RUN}})))
    results = query_results(c)
    top = results[0] if results else None
    c.check("top hit is ocean-a", top and top["Key"] == keys.get("ocean-a"), top and top["Key"])
    by_key = {r["Key"]: r.get("distance") for r in results}
    da, db = by_key.get(keys.get("ocean-a")), by_key.get(keys.get("ocean-b"))
    c.check("ocean-a is at least 0.1 closer than ocean-b", da is not None and db is not None and db - da >= 0.1, (da, db))
    c.observe("distances", {"ocean-a": da, "ocean-b": db})
    c.finish()

    c = new_case("g5-nova-query-filter-plus-nested-override", "G5", "positive",
                 "Same ocean-a text with --filter category==space AND a nested override (truncationMode NONE). "
                 "Expect only category=space results, ocean-a absent, the space vector on top: filter and override "
                 "are applied together.")
    run_cli(c, *query_args(NOVA, "--text-value", OCEAN_A, "--k", "3",
                           "--filter", params({"$and": [{"category": {"$eq": "space"}}, {"run": {"$eq": RUN}}]}),
                           "--bedrock-inference-params", params({"singleEmbeddingParams": {"text": {"truncationMode": "NONE"}}})))
    results = query_results(c)
    c.check("exit 0 with >= 1 result", c.exit_code == 0 and len(results) >= 1, (c.exit_code, len(results)))
    c.check("every result has category=space", results and all(r.get("metadata", {}).get("category") == "space" for r in results),
            [r.get("metadata", {}).get("category") for r in results])
    c.check("ocean-a is not returned", all(r["Key"] != keys.get("ocean-a") for r in results))
    c.check("top hit is the space vector", results and results[0]["Key"] == keys.get("space"), results and results[0]["Key"])
    c.finish()

    c = new_case("g5-nova-query-protected-override-rejected", "G5", "negative",
                 "Query with an override on singleEmbeddingParams.embeddingDimension: rejected client-side before "
                 "any AWS client is created.")
    run_cli(c, *query_args(NOVA, "--text-value", "anything", "--bedrock-inference-params",
                           params({"singleEmbeddingParams": {"embeddingDimension": 256}})))
    expect_client_rejection(c, None, "Cannot override system-controlled parameters")
    c.finish()

    c = new_case("g5-nova-query-flat-key-rejected-by-bedrock", "G5", "probe",
                 "Query with the old flat shape {truncationMode: START}. PR #18 no longer relocates it, so it lands "
                 "at the request root and Bedrock rejects it ('extraneous key'). Documents the breaking change on "
                 "the query/sync path.", expect="rejected")
    run_cli(c, *query_args(NOVA, "--text-value", "anything", "--bedrock-inference-params", params({"truncationMode": "START"})))
    expect_clean_outcome(c, None, "rejected by Bedrock (extraneous key)")
    c.check("Bedrock named the extraneous key", "extraneous key" in c.output, c.output.strip()[-300:])
    c.finish()


def query_results(case: Case) -> List[Dict[str, Any]]:
    return case.payload.get("results", []) if isinstance(case.payload, dict) else []


def query_top(case: Case) -> Optional[Dict[str, Any]]:
    results = query_results(case)
    return results[0] if results else None


def group_g6(fx: Fixtures) -> None:
    """Cross-account bucketOwner."""
    c = new_case("g6-nova-async-cross-account-with-owner", "G6", "positive",
                 "Async image from a bucket in another account with --src-bucket-owner set: bucketOwner is sent in "
                 "the source and the put succeeds.")
    if not (CROSS_S3 and CROSS_OWNER):
        c.skip("S3V_CROSS_ACCOUNT_S3 / S3V_CROSS_ACCOUNT_OWNER not set")
    else:
        run_cli(c, *put_args(NOVA, c, "--image", CROSS_S3, "--src-bucket-owner", CROSS_OWNER, "--async-output-s3-uri", async_out(c)))
        expect_success(c, NOVA, 1)
        c.finish()

    c = new_case("g6-nova-async-cross-account-default-owner", "G6", "probe",
                 "Same object without --src-bucket-owner: the CLI fills in the caller's own account id, which is "
                 "wrong for a cross-account bucket, so Bedrock should refuse to read the source. Proves the field matters.",
                 expect="rejected")
    if not (CROSS_S3 and CROSS_OWNER):
        c.skip("S3V_CROSS_ACCOUNT_S3 / S3V_CROSS_ACCOUNT_OWNER not set")
    else:
        run_cli(c, *put_args(NOVA, c, "--image", CROSS_S3, "--async-output-s3-uri", async_out(c)))
        expect_clean_outcome(c, NOVA, "rejected (wrong bucketOwner)", async_out(c))
        c.finish()


def group_g7(fx: Fixtures) -> None:
    """Video segment structure (needs a provided video with an audio track)."""
    reason = "S3V_VIDEO_S3 not set"

    c = new_case("g7-nova-async-video-default-structure", "G7", "positive",
                 "Async video with the CLI default segmentationConfig {durationSeconds: 5} (sent explicitly, not an "
                 "empty wrapper). Expect contiguous AUDIO_VIDEO_COMBINED segments of exactly 5 s"
                 + (f", count ceil({VIDEO_DURATION}/5)" if VIDEO_DURATION else "") + ".")
    if not VIDEO_S3:
        c.skip(reason)
    else:
        run_cli(c, *put_args(NOVA, c, "--video", VIDEO_S3, "--async-output-s3-uri", async_out(c)))
        vectors = expect_success(c, NOVA)
        c.check("all vectors typed AUDIO_VIDEO_COMBINED", vectors and all(v["metadata"].get(META_TYPE) == "AUDIO_VIDEO_COMBINED" for v in vectors))
        check_segments(c, vectors, start_key=META_START, end_key=META_END, length=5, total=VIDEO_DURATION)
        c.finish()

    c = new_case("g7-nova-async-video-separate-duration-10", "G7", "positive",
                 "Nested override on two sibling fields at once: embeddingMode AUDIO_VIDEO_SEPARATE and "
                 "segmentationConfig.durationSeconds 10. Expect AUDIO and VIDEO runs of equal length, each "
                 "contiguous with 10 s non-final segments.")
    if not VIDEO_S3:
        c.skip(reason)
    else:
        run_cli(c, *put_args(NOVA, c, "--video", VIDEO_S3, "--async-output-s3-uri", async_out(c),
                             "--bedrock-inference-params",
                             params({"segmentedEmbeddingParams": {"video": {"embeddingMode": "AUDIO_VIDEO_SEPARATE",
                                                                             "segmentationConfig": {"durationSeconds": 10}}}})))
        vectors = expect_success(c, NOVA)
        types = sorted({v["metadata"].get(META_TYPE) for v in vectors})
        audio = [v for v in vectors if v["metadata"].get(META_TYPE) == "AUDIO"]
        video = [v for v in vectors if v["metadata"].get(META_TYPE) == "VIDEO"]
        c.check("SEPARATE mode produced VIDEO-typed vectors (not AUDIO_VIDEO_COMBINED): embeddingMode reached Bedrock",
                bool(video) and "AUDIO_VIDEO_COMBINED" not in types, types)
        if audio:
            c.check("equal segment counts per type", len(audio) == len(video), (len(audio), len(video)))
            check_segments(c, audio, start_key=META_START, end_key=META_END, length=10, total=VIDEO_DURATION, label="AUDIO")
        else:
            c.observe("audio", "no AUDIO vectors: the clip has no audio track, so Bedrock wrote a VIDEO manifest entry only")
        check_segments(c, video, start_key=META_START, end_key=META_END, length=10, total=VIDEO_DURATION, label="VIDEO")
        c.finish()

    c = new_case("g7-nova-sync-video-separate", "G7", "positive",
                 "Sync video (<= 30 s) with embeddingMode AUDIO_VIDEO_SEPARATE: expect exactly 2 vectors typed "
                 "AUDIO and VIDEO (a video with an audio track is required for this).")
    if not VIDEO_S3:
        c.skip(reason)
    else:
        run_cli(c, *put_args(NOVA, c, "--video", VIDEO_S3, "--bedrock-inference-params",
                             params({"singleEmbeddingParams": {"video": {"embeddingMode": "AUDIO_VIDEO_SEPARATE"}}})))
        vectors = expect_success(c, NOVA)
        types = sorted(v["metadata"].get(META_TYPE) for v in vectors)
        c.check("SEPARATE mode: types are [AUDIO, VIDEO] for a clip with audio, or [VIDEO] for a silent clip",
                types in (["AUDIO", "VIDEO"], ["VIDEO"]), types)
        c.observe("types", types)
        c.finish()


GROUPS = {"g1": group_g1, "g2": group_g2, "g3": group_g3, "g4": group_g4, "g5": group_g5, "g6": group_g6, "g7": group_g7}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def environment() -> Dict[str, Any]:
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
        except Exception:
            return "n/a"
    try:
        account = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    except Exception:
        account = "n/a"
    return {
        "run": RUN,
        "repo": REPO,
        "git_head": git("rev-parse", "--short", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0],
        "cli_python": PYTHON,
        "account": account,
        "region": REGION,
        "vector_bucket": VB,
        "indexes": {m: f"{i} (dim {DIM[m]})" for m, i in INDEX.items()},
        "input_prefix": f"s3://{INPUT_BUCKET}/{INPUT_PREFIX}/{RUN}/",
        "async_output": f"{ASYNC_OUTPUT_ROOT}/{RUN}/",
        "video": VIDEO_S3 or "not provided (G7 skipped)",
        "video_duration": VIDEO_DURATION,
        "jpeg": "provided" if JPEG_SRC else "generated with Pillow if available",
        "cross_account": "provided" if (CROSS_S3 and CROSS_OWNER) else "not provided (G6 skipped)",
    }


def write_report(out_dir: str, env: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    passed = sum(c.verdict == "PASS" for c in CASES)
    failed = sum(c.verdict == "FAIL" for c in CASES)
    skipped = sum(c.verdict == "SKIP" for c in CASES)

    lines = [f"# Gap validation report, run `{RUN}`", ""]
    lines += [f"Code: `{env['git_branch']} @ {env['git_head']}`" + (" (uncommitted changes present)" if env["git_dirty"] else ""),
              f"Account `{env['account']}`, region `{env['region']}`, vector bucket `{env['vector_bucket']}`.", ""]
    lines += ["| Model | Index |", "|---|---|"] + [f"| `{m}` | `{i}` |" for m, i in env["indexes"].items()] + [""]
    lines += [f"Fixtures: `{env['input_prefix']}`; async output: `{env['async_output']}`.",
              f"Video fixture: {env['video']}" + (f" ({env['video_duration']} s)" if env["video_duration"] else ""),
              f"Cross-account fixture: {env['cross_account']}", "",
              f"**Result: {passed} passed, {failed} failed, {skipped} skipped, of {len(CASES)}.**", ""]

    lines += ["## Pass criteria", "",
              "- positive: exit 0, no traceback, index count equals the CLI-reported count and the expected count, "
              "metadata and segment structure as stated per case (contiguous indexes, start[i+1] == end[i], "
              "non-final segment length equal to the configured value), query top hit identity.",
              "- negative: non-zero exit, no traceback, the expected client-side message, nothing stored, nothing "
              "written to the async output prefix.",
              "- probe: input the CLI cannot validate itself. Passes when the CLI finishes cleanly (no traceback, "
              "no hang) with state consistent with its exit code. The *outcome* and Bedrock's message are recorded; "
              "the expected outcome is noted per case but not enforced, since it is Bedrock behavior, not CLI behavior.", ""]

    lines += ["## Summary", "", "| Case | Group | Kind | Verdict | Exit | Time | Outcome |", "|---|---|---|---|---|---|---|"]
    for c in CASES:
        outcome = next((v for k, v in c.observations if k == "outcome"), "")
        if c.verdict == "SKIP":
            outcome = f"skipped: {c.skip_reason}"
        lines.append(f"| [{c.id}](#{c.id}) | {c.group} | {c.kind} | **{c.verdict}** | {c.exit_code if c.exit_code is not None else ''} "
                     f"| {c.elapsed:.0f}s | {outcome} |")
    lines.append("")

    if failed:
        lines += ["## Failed checks", ""]
        for c in CASES:
            if c.verdict == "FAIL":
                for label, ok, observed in c.checks:
                    if not ok:
                        lines.append(f"- `{c.id}`: {label} — observed `{observed[:300]}`")
        lines.append("")

    group_titles = {"G1": "G1 Bedrock-side failures", "G2": "G2 Media extensions", "G3": "G3 Nova limits and segmentation",
                    "G4": "G4 Batch robustness", "G5": "G5 Query", "G6": "G6 Cross-account", "G7": "G7 Video structure"}
    for g, title in group_titles.items():
        cases = [c for c in CASES if c.group == g]
        if not cases:
            continue
        lines += [f"## {title}", ""]
        for c in cases:
            lines += [f"### {c.id}", "", f"- Kind: {c.kind}", f"- Verdict: **{c.verdict}**"
                      + (f" (exit {c.exit_code}, {c.elapsed:.0f}s)" if c.verdict != "SKIP" else f" ({c.skip_reason})"),
                      f"- Purpose: {c.purpose}"]
            if c.expect:
                lines.append(f"- Expected outcome: {c.expect}")
            if c.checks:
                lines.append("- Checks:")
                for label, ok, observed in c.checks:
                    mark = "PASS" if ok else "FAIL"
                    lines.append(f"    - {mark}: {label}" + (f" — `{observed[:400]}`" if observed and (not ok or 'segment' in label or 'count' in label) else ""))
            if c.observations:
                lines.append("- Observations:")
                for k, v in c.observations:
                    lines.append(f"    - {k}: `{v[:600]}`")
            if c.command:
                lines += ["", "```bash", c.command, "```", "<details><summary>Full CLI output</summary>", "", "```",
                          c.output.strip()[-6000:] or "(no output)", "```", "</details>", ""]
            lines.append("")

    lines += ["## Cleanup", "",
              f"Objects under `{env['input_prefix']}` and `{env['async_output']}` can be deleted. Indexes created by this run: "
              + (", ".join(f"`{i}`" for i in created_indexes) or "none (all pre-existed)") + ".", ""]

    with open(os.path.join(out_dir, "gap-report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "gap-summary.tsv"), "w", encoding="utf-8") as f:
        for c in CASES:
            f.write("\t".join([c.id, c.group, c.kind, c.verdict, str(c.exit_code if c.exit_code is not None else ""),
                               f"{c.elapsed:.0f}s", c.purpose.replace("\t", " ")]) + "\n")
    with open(os.path.join(out_dir, "gap-results.json"), "w", encoding="utf-8") as f:
        json.dump({"environment": env, "cases": [{
            "id": c.id, "group": c.group, "kind": c.kind, "verdict": c.verdict, "skip_reason": c.skip_reason,
            "purpose": c.purpose, "expected": c.expect, "exit_code": c.exit_code, "elapsed_s": round(c.elapsed, 1),
            "checks": [{"label": l, "ok": ok, "observed": o} for l, ok, o in c.checks],
            "observations": dict(c.observations), "command": c.command, "output": c.output,
            "vector_keys": [v["key"] for v in c.vectors],
        } for c in CASES]}, f, indent=2)
    print(f"\nreport: {os.path.join(out_dir, 'gap-report.md')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("Required environment"):])
    ap.add_argument("--out", default=None, help="report directory (default ./gap-run-<run id>)")
    ap.add_argument("--only", default=None, help="comma-separated groups to run, e.g. g1,g3")
    ap.add_argument("--cleanup", action="store_true", help="delete uploaded fixtures, async output, and indexes this run created")
    ap.add_argument("--keep-fixtures", action="store_true", help="keep the local fixture directory")
    args = ap.parse_args()

    for name, value in (("S3V_VECTOR_BUCKET", VB), ("S3V_INPUT_BUCKET", INPUT_BUCKET), ("S3V_ASYNC_OUTPUT", ASYNC_OUTPUT_ROOT)):
        if not value:
            die(f"{name} is required (see --help)")
    if not ASYNC_OUTPUT_ROOT.startswith("s3://"):
        die("S3V_ASYNC_OUTPUT must be an s3:// URI")
    if not os.path.isdir(os.path.join(REPO, "s3vectors")):
        die(f"S3V_REPO={REPO} does not look like the s3vectors-embed-cli checkout")
    if NOVA_DIM not in (256, 384, 1024, 3072):
        die("S3V_NOVA_DIM must be one of 256, 384, 1024, 3072")

    selected = {g.strip().lower() for g in args.only.split(",")} if args.only else set(GROUPS)
    unknown = selected - set(GROUPS)
    if unknown:
        die(f"unknown group(s) {sorted(unknown)}; choose from {sorted(GROUPS)}")

    out_dir = args.out or os.path.join(os.getcwd(), f"gap-run-{RUN}")
    env = environment()
    print(f"run {RUN}: {env['git_branch']} @ {env['git_head']}, account {env['account']}, region {REGION}")
    print(f"groups: {', '.join(sorted(selected))}; report -> {out_dir}\n", flush=True)

    ensure_infra()
    fx = Fixtures(tempfile.mkdtemp(prefix="pr18gap-"))
    try:
        fx.build()
        print("", flush=True)
        for g in GROUPS:
            if g in selected:
                try:
                    GROUPS[g](fx)
                except Exception as exc:  # a harness bug must not lose the rest of the run
                    pending = [c for c in CASES if c.verdict == "PENDING"]
                    for c in pending:
                        c.check("harness completed the case without an internal error", False, f"{type(exc).__name__}: {exc}")
                        c.finish()
                    print(f"harness error in group {g}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        write_report(out_dir, env)
        if args.cleanup:
            cleanup(fx, args.keep_fixtures, delete_infra=True)
        elif not args.keep_fixtures:
            shutil.rmtree(fx.root, ignore_errors=True)

    failed = [c.id for c in CASES if c.verdict == "FAIL"]
    passed = sum(c.verdict == "PASS" for c in CASES)
    skipped = sum(c.verdict == "SKIP" for c in CASES)
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped, of {len(CASES)}")
    for cid in failed:
        print(f"  FAIL {cid}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
