import json
import pwd
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


with Path(__file__).with_name("config.json").open(encoding="utf-8") as stream:
    models = json.load(stream)["models"]

print(f"Downloading {len(models)} models...")

for model in models:
    print(f"Downloading model {model['name']} from {model['repo']}...")
    try:
        snapshot_download(repo_id=model["repo"])
    except PermissionError as error:
        if error.filename:
            path = Path(error.filename)
            try:
                uid = path.stat().st_uid
                try:
                    owner = pwd.getpwuid(uid).pw_name
                except KeyError:
                    owner = "unknown"
                print(f"Owner of {path}: {owner} (UID {uid})", file=sys.stderr)
            except OSError as owner_error:
                print(f"Cannot determine owner of {path}: {owner_error}", file=sys.stderr)
        # raise
