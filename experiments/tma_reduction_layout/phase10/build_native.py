"""Build an exact upstream snapshot over the cached source/build directory."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile


def main():
    root = Path('/opt/triton-src')
    evidence = Path('/opt/phase10-build-evidence')
    evidence.mkdir(exist_ok=True)
    archive = Path('/opt/phase10-upstream.tar')
    (evidence / 'source_archive_sha256.txt').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'\n')
    (evidence / 'ccache_before.txt').write_text(subprocess.check_output(['ccache','--print-stats'],text=True))
    # Preserve the incremental CMake build, remove stale source files, then
    # extract only the immutable tracked upstream snapshot at the same path.
    for path in root.iterdir():
        if path.name == 'build':
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    with tarfile.open(archive) as tar:
        tar.extractall(root,filter='data')
    commands = [[sys.executable,'-m','pip','install','-r','python/requirements.txt'],
                [sys.executable,'-m','pip','install','-e','.','--no-build-isolation','-v'],
                ['make']]
    for index,command in enumerate(commands):
        with (evidence / f'command_{index}.log').open('w') as log:
            result = subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print((evidence / f'command_{index}.log').read_text()[-20000:],flush=True)
        result.check_returncode()
    (evidence / 'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    for name,args in [('ccache_after.txt',['ccache','--print-stats']),('ccache_summary.txt',['ccache','-s'])]:
        data = subprocess.check_output(args,text=True)
        (evidence / name).write_text(data)
        print(data,flush=True)


if __name__ == '__main__':
    main()
