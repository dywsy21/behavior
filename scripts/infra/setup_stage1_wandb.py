"""Provision an explicitly supplied W&B key without echoing or committing it."""
import argparse
import getpass
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credential-file", type=Path, required=True)
    args = parser.parse_args()
    if args.credential_file.exists():
        raise FileExistsError("Credential already exists; do not silently replace another user's setup")
    key = getpass.getpass("W&B API key (hidden): ").strip()
    if not key or "\n" in key:
        raise ValueError("Invalid credential")
    args.credential_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(args.credential_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(key + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("Credential saved outside source; mode 0600. Online verification is a separate gate.")


if __name__ == "__main__":
    main()
