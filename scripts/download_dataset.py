from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import struct
import urllib.request
import zipfile
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbannoise.dataset import load_metadata


DOWNLOAD_URL = "https://api.figshare.com/v2/file/download/58104025"
EXPECTED_SIZE = 2_971_638_528
EXPECTED_MD5 = "b3eb4af4cfdd3147de42379922652eaa"


def download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == EXPECTED_SIZE:
        print(f"Using existing archive: {destination}")
        return

    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "urban-noise-uganda-course-project/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", EXPECTED_SIZE))
            received = 0
            with partial.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    received += len(chunk)
                    print(
                        f"\rDownloaded {received:,} / {total:,} bytes",
                        end="",
                        flush=True,
                    )
        print()
        if received != EXPECTED_SIZE:
            raise RuntimeError(
                f"Expected {EXPECTED_SIZE:,} bytes, received {received:,}."
            )
        partial.replace(destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def verify_md5(archive_path: Path) -> None:
    digest = hashlib.md5()
    with archive_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != EXPECTED_MD5:
        raise RuntimeError(
            f"Dataset archive checksum mismatch: expected {EXPECTED_MD5}, got {actual}."
        )


def read_range(start: int, end: int) -> bytes:
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={
            "Range": f"bytes={start}-{end - 1}",
            "User-Agent": "urban-noise-uganda-course-project/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(
            f"bytes {start}-{end - 1}/"
        ):
            raise RuntimeError(
                "The dataset host did not honor a bounded HTTP range request."
            )
        return response.read()


def extract_metadata_by_range(data_directory: Path) -> None:
    archive_size = EXPECTED_SIZE
    tail_start = max(0, archive_size - 65_557)
    tail = read_range(tail_start, archive_size)
    eocd_position = tail.rfind(b"PK\x05\x06")
    if eocd_position < 0:
        raise RuntimeError("Could not locate ZIP end-of-central-directory record.")
    _, _, _, _, entry_count, central_size, central_offset, _ = struct.unpack_from(
        "<4s4H2LH", tail, eocd_position
    )

    zip64_locator_position = tail.rfind(
        b"PK\x06\x07",
        max(0, eocd_position - 20),
        eocd_position,
    )
    if zip64_locator_position >= 0:
        _, _, zip64_offset, _ = struct.unpack_from(
            "<4sLQL", tail, zip64_locator_position
        )
        zip64_end = read_range(zip64_offset, zip64_offset + 56)
        (
            signature,
            _,
            _,
            _,
            _,
            _,
            _,
            entry_count,
            central_size,
            central_offset,
        ) = struct.unpack_from("<4sQ2H2L4Q", zip64_end)
        if signature != b"PK\x06\x06":
            raise RuntimeError("Invalid ZIP64 end-of-central-directory record.")

    central = read_range(central_offset, central_offset + central_size)
    position = 0
    metadata_member: tuple[int, int, int, int, int] | None = None
    for _ in range(entry_count):
        if central[position : position + 4] != b"PK\x01\x02":
            raise RuntimeError("Invalid ZIP central-directory entry.")
        values = struct.unpack_from("<4s6H3L5H2L", central, position)
        crc, compressed, uncompressed = values[7], values[8], values[9]
        name_length, extra_length, comment_length = values[10:13]
        method = values[4]
        local_offset = values[16]
        name_start = position + 46
        name = central[name_start : name_start + name_length].decode(
            "utf-8", errors="strict"
        )
        extra_start = name_start + name_length
        extra = central[extra_start : extra_start + extra_length]

        if compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
            extra_position = 0
            while extra_position + 4 <= len(extra):
                tag, size = struct.unpack_from("<HH", extra, extra_position)
                value = extra[extra_position + 4 : extra_position + 4 + size]
                if tag == 1:
                    value_position = 0
                    if uncompressed == 0xFFFFFFFF:
                        uncompressed = struct.unpack_from("<Q", value, value_position)[0]
                        value_position += 8
                    if compressed == 0xFFFFFFFF:
                        compressed = struct.unpack_from("<Q", value, value_position)[0]
                        value_position += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack_from("<Q", value, value_position)[0]
                    break
                extra_position += 4 + size

        if name == "urban-noise/metadata.csv":
            metadata_member = (crc, compressed, uncompressed, local_offset, method)
            break
        position += 46 + name_length + extra_length + comment_length

    if metadata_member is None:
        raise RuntimeError("The archive does not contain urban-noise/metadata.csv.")
    crc, compressed, uncompressed, local_offset, method = metadata_member
    local_header = read_range(local_offset, local_offset + 30)
    header = struct.unpack("<4s5H3L2H", local_header)
    if header[0] != b"PK\x03\x04":
        raise RuntimeError("Invalid local ZIP header for metadata.csv.")
    name_length, extra_length = header[9], header[10]
    data_offset = local_offset + 30 + name_length + extra_length
    compressed_data = read_range(data_offset, data_offset + compressed)
    if method == 8:
        content = zlib.decompress(compressed_data, -15)
    elif method == 0:
        content = compressed_data
    else:
        raise RuntimeError(f"Unsupported metadata compression method: {method}.")
    if len(content) != uncompressed or zlib.crc32(content) != crc:
        raise RuntimeError("The extracted metadata failed its ZIP integrity check.")

    data_directory.mkdir(parents=True, exist_ok=True)
    (data_directory / "metadata.csv").write_bytes(content)
    metadata = load_metadata(data_directory / "metadata.csv")
    print(f"Extracted and validated {len(metadata):,} metadata rows by HTTP ranges.")
    if "timestamp" not in metadata.columns:
        print("No timestamp column in the public archive; temporal features are disabled.")


def extract_dataset(archive_path: Path, data_directory: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            item
            for item in archive.infolist()
            if item.filename == "urban-noise/metadata.csv"
            or (
                item.filename.startswith("urban-noise/audio_files/")
                and item.filename.lower().endswith(".ogg")
                and not item.is_dir()
            )
        ]
        if not any(item.filename == "urban-noise/metadata.csv" for item in members):
            raise RuntimeError("The archive does not contain urban-noise/metadata.csv.")

        for index, item in enumerate(members, start=1):
            destination = (
                data_directory / "metadata.csv"
                if item.filename == "urban-noise/metadata.csv"
                else data_directory / "audio_files" / Path(item.filename).name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if index % 500 == 0 or index == len(members):
                print(f"\rExtracted {index:,} / {len(members):,} files", end="")
        print()

    metadata = load_metadata(data_directory / "metadata.csv")
    print(f"Validated {len(metadata):,} metadata rows.")
    if "timestamp" not in metadata.columns:
        print("No timestamp column in the public archive; temporal features are disabled.")
    audio_directory = data_directory / "audio_files"
    expected_names = set(metadata["filename"].astype(str))
    actual_names = {
        path.name for path in audio_directory.glob("*.ogg") if path.is_file()
    }
    missing_names = expected_names.difference(actual_names)
    unexpected_names = actual_names.difference(expected_names)
    if missing_names or unexpected_names:
        raise RuntimeError(
            "Audio/metadata filename mismatch: "
            f"{len(missing_names)} missing, {len(unexpected_names)} unexpected."
        )
    print(f"Verified all {len(actual_names):,} audio filenames against the CSV.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and safely extract the public Uganda dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Destination for metadata.csv and audio_files/.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data" / "raw" / "urban-noise.zip",
        help="Path for the downloaded Figshare archive.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Extract metadata.csv but do not extract audio files.",
    )
    arguments = parser.parse_args()
    if arguments.metadata_only:
        extract_metadata_by_range(arguments.data_dir)
        print(f"Metadata ready in {arguments.data_dir}")
        return
    download_archive(arguments.archive)
    verify_md5(arguments.archive)
    extract_dataset(arguments.archive, arguments.data_dir)
    print(f"Data ready in {arguments.data_dir}")


if __name__ == "__main__":
    main()
