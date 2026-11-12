"""Operate this run with ZZHKAGGLETOKEN only; never print credentials."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.kaggle_runtime'))
token = os.environ.get('ZZHKAGGLETOKEN')
if not token:
    raise SystemExit('ZZHKAGGLETOKEN is missing; no other credentials will be used.')
os.environ['KAGGLE_API_TOKEN'] = token
for key in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
    os.environ.pop(key, None)
from kaggle import api

owner = api.get_config_value('username')
JOB = ROOT / '.kaggle_job'
JOB.mkdir(exist_ok=True)
state_path = JOB / 'state.json'
if state_path.exists():
    state = json.loads(state_path.read_text())
    assert state['owner'] == owner, 'Run owner differs from authorized account'
else:
    state = {'owner': owner, 'dataset': f'{owner}/recsys-baby-20261113',
             'kernel': f'{owner}/recsys-baby-full-20261113'}
    state_path.write_text(json.dumps(state, indent=2))
action = sys.argv[1]
if action == 'auth':
    print(json.dumps(state))
elif action == 'upload':
    import shutil
    stage = JOB / 'dataset'
    stage.mkdir(exist_ok=True)
    for source in (ROOT / 'data' / 'baby').iterdir():
        if source.is_file() and source.suffix in ('.inter', '.csv', '.npy'):
            target = stage / source.name
            if not target.exists():
                shutil.copy2(source, target)
    (stage / 'dataset-metadata.json').write_text(json.dumps({
        'title': 'RecSys Baby 20261113', 'id': state['dataset'],
        'licenses': [{'name': 'other'}]
    }))
    response = api.dataset_create_new(str(stage), public=False, quiet=True, convert_to_csv=False)
    print(response)
elif action == 'push':
    import shutil
    stage = JOB / 'kernel'
    stage.mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'notebook' / 'recsys_baseline_kaggle.ipynb', stage)
    (stage / 'kernel-metadata.json').write_text(json.dumps({
        'id': state['kernel'], 'title': 'RecSys Baby Full 20261113',
        'code_file': 'recsys_baseline_kaggle.ipynb', 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
        'enable_internet': False, 'dataset_sources': [state['dataset']],
        'competition_sources': [], 'kernel_sources': []
    }))
    print(api.kernels_push(str(stage), timeout='43200', acc='NvidiaTeslaT4'))
elif action == 'status':
    print(api.kernels_status(state['kernel']))
elif action == 'notebook':
    target = ROOT / 'notebook' / 'recsys_baseline_kaggle_results.ipynb'
    target.touch(exist_ok=True)
    api.kernels_pull(state['kernel'] + '/2', path=str(target), metadata=False)
    notebook = json.loads(target.read_text(encoding='utf-8'))
    import pandas as pd
    results = ROOT / 'results' / 'kaggle_20261113' / 'recsys' / 'results'
    for index, name in [(9, 'itemcf'), (13, 'swing'), (17, 'usercf'),
                        (22, 'dualtower'), (26, 'threetower')]:
        frame = pd.read_csv(results / f'{name}_all_metrics.csv')
        notebook['cells'][index]['outputs'] = [{
            'output_type': 'display_data', 'metadata': {},
            'data': {'text/plain': [frame.to_string(index=False)],
                     'text/html': [frame.to_html(index=False)]}
        }]
    notebook['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# Kaggle version 2 — downloaded code with actual result tables\n',
        'The five evaluation tables below are embedded from the verified CSV outputs of the completed Kaggle run.\n',
        'These are actual results, not a local rerun. Original per-cell execution outputs are not provided by the pull API.\n',
        'Source: https://www.kaggle.com/code/callingzeng/recsys-baby-full-20261113 (version 2)\n'
    ]})
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('Saved:', target)
    print('Cells with outputs:', sum(bool(c.get('outputs')) for c in notebook['cells']))
elif action == 'logs':
    print(api.kernels_logs(state['kernel']))
elif action == 'output':
    target = ROOT / 'results' / 'kaggle_20261113'
    target.mkdir(parents=True, exist_ok=True)
    print(api.kernels_output(state['kernel'], path=str(target)))
else:
    raise SystemExit('Unknown action')
