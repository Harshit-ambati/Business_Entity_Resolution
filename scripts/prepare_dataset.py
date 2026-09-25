"""Extract the approved challenge files from the LFS-tracked resource archive."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path


ARCHIVE_SHA256 = "2aefd2f8eb6f132b8933fccc1cbb98fa6356756a7581fed5914a5f26704a8bc5"
RESOURCE_ROOT = "student_resource/"
ALLOWED_FILES = {
    "student_resource/README.md",
    "student_resource/Documentation_template.md",
    "student_resource/utils/validate_submission.py",
    "student_resource/dataset/train/train_source1.tsv",
    "student_resource/dataset/train/train_source2.tsv",
    "student_resource/dataset/train/train_source3.tsv",
    "student_resource/dataset/train/train_ground_truth.tsv",
    "student_resource/dataset/test/test_source1.tsv",
    "student_resource/dataset/test/test_source2.tsv",
    "student_resource/dataset/test/test_source3.tsv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive: Path, destination: Path, *, force: bool = False) -> None:
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"{archive} is not a ZIP archive; run 'git lfs pull' first")
    actual_hash = sha256(archive)
    if actual_hash != ARCHIVE_SHA256:
        raise ValueError(f"archive SHA-256 mismatch: {actual_hash}")

    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        actual_files = {item.filename for item in zipped.infolist() if item.filename in ALLOWED_FILES}
        if actual_files != ALLOWED_FILES:
            missing = sorted(ALLOWED_FILES - actual_files)
            raise ValueError(f"archive missing expected files: {missing}")

        for name in sorted(ALLOWED_FILES):
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"unsafe archive path: {name}")
            expected_size = zipped.getinfo(name).file_size
            if target.exists() and not force:
                if target.stat().st_size != expected_size:
                    raise ValueError(f"incomplete existing file: {target} (use --force to replace)")
                print(f"Already present: {target} (use --force to overwrite)")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with zipped.open(name) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if temporary.stat().st_size != expected_size:
                raise ValueError(f"incomplete extraction: {name}")
            temporary.replace(target)
            print(f"Extracted: {target} ({expected_size} bytes)")


def main() -> int:
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=repository / "data/student_resource.zip")
    parser.add_argument("--destination", type=Path, default=repository)
    parser.add_argument("--force", action="store_true", help="overwrite existing extracted files")
    args = parser.parse_args()
    try:
        extract(args.archive, args.destination, force=args.force)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
