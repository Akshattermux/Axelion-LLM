"""Build a deduplicated, weighted raw corpus for Axelion causal pretraining."""

import argparse
import hashlib
import json
import re
from pathlib import Path


TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".java", ".cpp", ".c", ".h", ".sql", ".html", ".css"}
BINARY_RATIO_LIMIT = 0.02


def clean_document(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_usable(text: str) -> bool:
    if len(text) < 200:
        return False
    control_chars = sum(ord(char) < 32 and char not in "\n\r\t" for char in text)
    return control_chars / max(1, len(text)) <= BINARY_RATIO_LIMIT


def load_documents(source_dir: Path):
    seen = set()
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if any(part in {"node_modules", "vendor", "dist", "build", ".git"} for part in path.parts):
            continue
        try:
            text = clean_document(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        if not is_usable(text):
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            yield text


def build_corpus(manifest_path: str, output_path: str, max_chars: int | None = None):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output = []
    total_chars = 0
    for category, weight in manifest["sampling_weights"].items():
        source_dir = Path(manifest["source_layout"][category])
        documents = list(load_documents(source_dir)) if source_dir.exists() else []
        if not documents:
            print(f"Warning: no usable documents for {category}")
            continue
        category_budget = max_chars * weight if max_chars else None
        category_chars = 0
        for document in documents:
            if category_budget and category_chars >= category_budget:
                break
            output.append(f"\n\n### {category}\n{document}")
            category_chars += len(document)
            total_chars += len(document)
    if not output:
        raise RuntimeError("No usable source documents found")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("".join(output), encoding="utf-8")
    print(f"Wrote {total_chars:,} characters to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a cleaned weighted pretraining corpus")
    parser.add_argument("--manifest", default="data/pretraining/mixture.json")
    parser.add_argument("--output", default="data/pretraining/train.txt")
    parser.add_argument("--max-chars", type=int, default=None)
    args = parser.parse_args()
    build_corpus(args.manifest, args.output, args.max_chars)
