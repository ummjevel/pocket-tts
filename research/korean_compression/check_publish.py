"""Reject machine-local data in this branch's tracked files before publishing."""
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    # Build patterns at runtime so the checker itself does not carry machine labels.
    patterns=[re.compile('/'+'data'+'/users/'),re.compile('/'+'data'+'/datasets/'),
        re.compile('zo'+'ey',re.I),re.compile('H'+'100',re.I),
        re.compile(r'GPU-[0-9a-f]{8}-',re.I)]
    errors=[]
    for name in filter(None,files):
        p=ROOT/name
        if p.is_symlink():
            import os
            target=Path(os.readlink(p))
            if target.is_absolute() or not p.resolve().is_relative_to(ROOT):
                errors.append((name,'external symlink'))
            continue
        if p.suffix in {'.pt','.safetensors','.wav','.mp3','.parquet'}:errors.append((name,'artifact'));continue
        try:text=p.read_text()
        except (UnicodeDecodeError,IsADirectoryError):continue
        if any(x.search(text) for x in patterns):errors.append((name,'environment literal'))
    if errors:
        print(errors);return 1
    print('Tracked-file privacy/artifact check passed');return 0


if __name__=='__main__':sys.exit(main())
