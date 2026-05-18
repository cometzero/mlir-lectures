#!/usr/bin/env python3
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[1]
errors=[]
lectures=sorted((ROOT/'lectures').glob('lecture-*'))
if len(lectures)!=20:
    errors.append(f'expected 20 lecture directories, found {len(lectures)}')
for i in range(1,21):
    d=ROOT/'lectures'/f'lecture-{i:02d}'
    if not d.exists():
        errors.append(f'missing {d.relative_to(ROOT)}')
        continue
    for required in ['README.md','LESSON.md']:
        if not (d/required).exists():
            errors.append(f'missing {d.relative_to(ROOT)}/{required}')
    if not any(d.glob('*_Handout.md')):
        errors.append(f'missing handout in {d.relative_to(ROOT)}')
    if not (d/'examples').exists():
        errors.append(f'missing examples dir in {d.relative_to(ROOT)}')
    mf=d/'manifest.json'
    if mf.exists():
        try:
            json.loads(mf.read_text(encoding='utf-8'))
        except Exception as e:
            errors.append(f'invalid manifest {mf.relative_to(ROOT)}: {e}')
if not (ROOT/'docs'/'syllabus.md').exists(): errors.append('missing docs/syllabus.md')
if not (ROOT/'docs'/'lab-environment.md').exists(): errors.append('missing docs/lab-environment.md')
if errors:
    print('Material verification failed:')
    for e in errors: print(' -', e)
    sys.exit(1)
print('Material verification passed.')
print(f'Lectures: {len(lectures)}')
for d in lectures:
    ex=list((d/'examples').glob('*')) if (d/'examples').exists() else []
    print(f'- {d.name}: {len(ex)} example files')
