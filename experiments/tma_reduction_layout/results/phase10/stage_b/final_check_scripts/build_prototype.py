"""Incremental compiler prototype build and required native/lit checks."""
from pathlib import Path
import json
import subprocess


def main():
    root=Path('/opt/triton-src')
    out=Path('/opt/phase10-prototype-build')
    out.mkdir(exist_ok=True)
    (out/'ccache_before.txt').write_text(subprocess.check_output(['ccache','--print-stats'],text=True))
    commands=[['make'],['make','triton-opt']]
    for index,command in enumerate(commands):
        with (out/f'build_{index}.log').open('w') as log:
            process=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                log.write(line);print(line,end='',flush=True)
            code=process.wait()
        if code:raise RuntimeError(f'Build failed: {command}, code={code}')
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    (out/'ccache_after.txt').write_text(subprocess.check_output(['ccache','--print-stats'],text=True))


if __name__=='__main__':main()
