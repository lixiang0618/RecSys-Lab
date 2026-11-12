"""Generate the Kaggle notebook without changing experiment definitions."""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
nb = json.loads((HERE / 'recsys_baseline.ipynb').read_text(encoding='utf-8'))
nb = copy.deepcopy(nb)

setup = '''from pathlib import Path
import shutil
import gc
import pickle
import tempfile
from collections.abc import MutableMapping

class DiskResults(MutableMapping):
    # Keep the dictionary interface and values, but load one experiment at a time.
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='recsys-')
        self.paths = {}
    def __getitem__(self, key):
        with self.paths[key].open('rb') as stream:
            return pickle.load(stream)
    def __setitem__(self, key, value):
        path = Path(self.directory.name) / (str(key) + '.pickle')
        with path.open('wb') as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.paths[key] = path
    def __delitem__(self, key):
        self.paths.pop(key).unlink()
    def __iter__(self):
        return iter(self.paths)
    def __len__(self):
        return len(self.paths)

# Set this explicitly only if multiple attached datasets contain baby.inter.
DATASET_DIR = None
WORK = Path('/kaggle/working/recsys')
RESULTS_ROOT = WORK / 'results'
RECALL_ROOT = WORK / 'recall'
for directory in (WORK, RESULTS_ROOT, RECALL_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
if DATASET_DIR is None:
    matches = sorted(Path('/kaggle/input').rglob('baby.inter'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one baby.inter; found {matches}. Set DATASET_DIR explicitly.')
    INPUT_DATA_DIR = matches[0].parent
else:
    INPUT_DATA_DIR = Path(DATASET_DIR)
required = ['baby.inter', 'u_id_mapping.csv', 'i_id_mapping.csv',
            'image_feat.npy', 'text_feat.npy', 'user_graph_dict.npy']
missing = [name for name in required if not (INPUT_DATA_DIR / name).is_file()]
assert not missing, f'Missing input files: {missing}'
# Optional existing recall artifacts may be attached as a separate dataset.
recall_inputs = sorted(p for p in Path('/kaggle/input').rglob('*.parquet')
                       if p.parent.name == 'recall')
sources = {}
for source in recall_inputs:
    if source.name in sources:
        raise RuntimeError(f'Duplicate recall filename: {source.name}')
    sources[source.name] = source
    shutil.copy2(source, RECALL_ROOT / source.name)
print('Input:', INPUT_DATA_DIR, '| Output:', WORK)
print('Imported recall files:', len(sources))
'''

for index, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    s = ''.join(cell['source'])
    if index == 18:
        s = s.replace('torch.arange(N_ITEMS).split(512)',
                      'torch.arange(N_ITEMS, device=DEVICE).split(512)')
    if index == 1:
        start = s.index('# ')
        end = s.index('# ', s.index('data_dir = matches[0].parent'))
        s = s.replace('from google.colab import drive\n', '').replace('drive.mount("/content/drive")\n', '')
        start = s.index('# ')
        end = s.index('# ', s.index('data_dir = matches[0].parent'))
        s = s[:start] + 'data_dir = INPUT_DATA_DIR\n\n' + s[end:]
    if index == 23:
        s = '''# Previous sections have persisted all recall artifacts and metrics.
for _name in ('online_results', 'scored_results', 'offline_models', 'rows',
              'records', 'recall_df', 'recs', 'scored_recs', 'model',
              'R', 'S', 'S_full', 'R_usercf', 'R_binary', 'TEXT', 'IMAGE'):
    globals().pop(_name, None)
gc.collect()
''' + s
        start = s.index("root = Path(")
        end = s.index("inter = pd.read_csv", start)
        s = s[:start] + 'data_dir = INPUT_DATA_DIR\n' + s[end:]
        s = s.replace("OUT = data_dir/'prerank'", "OUT = WORK/'prerank'")
        s = s.replace('    np.add.at(uh_t,uu,TEXT[jj]); np.add.at(uh_i,uu,IMAGE[jj]); np.add.at(ct,uu,1)',
'''    for start in range(0, len(uu), 2048):
        ublock, iblock = uu[start:start+2048], jj[start:start+2048]
        np.add.at(uh_t, ublock, TEXT[iblock])
        np.add.at(uh_i, ublock, IMAGE[iblock])
        np.add.at(ct, ublock, 1)''')
        s = s.replace('def cross_numpy(users,items,ff):', 'def _cross_numpy_block(users,items,ff):')
        pos = s.index('# ', s.index('return np.column_stack'))
        s = s[:pos] + '''def cross_numpy(users, items, ff):
    # Keep the original nine features; bound temporary high-dimensional arrays.
    result = np.empty((len(users), 9), dtype=np.float32)
    for start in range(0, len(users), 2048):
        end = min(start + 2048, len(users))
        result[start:end] = _cross_numpy_block(users[start:end], items[start:end], ff)
    return result

''' + s[pos:]
        s = s.replace('    model.eval(); MODELS[kind]=model', '''    model.eval()
    checkpoint_path = RESULTS / f'threetower-{kind}.pt'
    torch.save(model.state_dict(), checkpoint_path)
    MODELS[kind] = checkpoint_path''')
        s += '''\n    del model, opt, loss
    gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
# Release training-only tensors before candidate evaluation.
del FT, loader
gc.collect()
if DEVICE.type == 'cuda': torch.cuda.empty_cache()
'''
    if index == 24:
        s = s.replace("(data_dir/'recall')", 'RECALL_ROOT')
        s = s.replace("{data_dir/\"recall\"}", '{RECALL_ROOT}')
        s = s.replace("t=pd.read_parquet(f)", "t=pd.read_parquet(f, columns=['userID', 'itemID'])")
        s = s.replace("candidates=pd.concat(frames,ignore_index=True).drop_duplicates(['userID','itemID'])", "recall_frame_count = len(frames)\ncandidates=pd.concat(frames,ignore_index=True).drop_duplicates(['userID','itemID'])\ndel frames, t\ngc.collect()")
        s = s.replace("print('Recall files:',len(frames)", "print('Recall files:',recall_frame_count")
        s = s.replace('    model=MODELS[kind]', "    model=ThreeTower(kind).to(DEVICE)\n    model.load_state_dict(torch.load(MODELS[kind], map_location=DEVICE, weights_only=True))")
        s = s.replace('all_metrics=[]; scored_tables={};', 'all_metrics=[];')
        s = s.replace('    scored_tables[kind]=scored\n', '')
        s = s.replace("metrics=pd.DataFrame(all_metrics)", "    del model, scored, logits, ranked\n    gc.collect()\n    if DEVICE.type == 'cuda': torch.cuda.empty_cache()\n\nmetrics=pd.DataFrame(all_metrics)")
    s = s.replace('Path("/content/drive/MyDrive/RecSys-Lab/results")', 'RESULTS_ROOT')
    s = s.replace("Path('/content/drive/MyDrive/RecSys-Lab/results')", 'RESULTS_ROOT')
    s = s.replace("Path('/content/drive/MyDrive/RecSys-Lab/results/dualtower_models')", "RESULTS_ROOT / 'dualtower_models'")
    s = s.replace('data_dir / "recall"', 'RECALL_ROOT').replace("Path(data_dir) / 'recall'", 'RECALL_ROOT')
    if index in (5, 10, 14, 19):
        s = s.replace('online_results = {}', 'online_results = DiskResults()')
        s = s.replace('scored_results = {}', 'scored_results = DiskResults()')
        s = s.replace('online_results, online_rows = {}, []', 'online_results, online_rows = DiskResults(), []')
    cell['source'] = s.splitlines(keepends=True)
    cell['outputs'] = []
    cell['execution_count'] = None

nb['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [
    '# Kaggle version\n',
    'Attach the original Baby dataset under `/kaggle/input`. Enable a GPU in Kaggle settings.\n',
    'Run all cells for the full experiment. To run only pre-ranking, run setup and section 2; attach the complete original `recall/*.parquet` set as input.\n',
    'Experiment definitions and training settings are unchanged. Outputs are under `/kaggle/working/recsys`.\n',
    'Memory changes: chunk history accumulation and cross features; save/load three-tower models sequentially; release intermediate tables. No API calls are made by setup.\n']})
nb['cells'].insert(1, {'cell_type': 'code', 'metadata': {}, 'source': setup.splitlines(keepends=True), 'outputs': [], 'execution_count': None})
nb['cells'].append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': '''# Verify every requested output before marking the run complete.
import hashlib
import json
expected_recall = (
    [f'itemcf-{mode}-nb{nb}.parquet' for mode in ['binary','rating'] for nb in [10,50,100,'all']]
    + [f'swing-{mode}-a1-nb{nb}.parquet' for mode in ['binary','rating'] for nb in [10,50,100]]
    + [f'usercf-{method}-nb{nb}.parquet' for method in ['cosine','pop-penalty'] for nb in [10,50,100]]
    + [cfg['name'] + '.parquet' for cfg in EXPERIMENTS]
)
expected_prerank = [f'threetower-{kind}.parquet' for kind in KINDS]
for directory, names in [(RECALL_ROOT, expected_recall), (OUT, expected_prerank)]:
    assert all((directory / name).is_file() for name in names), (directory, names)
    for name in names:
        frame = pd.read_parquet(directory / name)
        assert {'userID','itemID','score','rank','channel'}.issubset(frame.columns)
        assert len(frame) > 0 and frame['rank'].between(1,100).all(), name
files = []
for path in sorted(WORK.rglob('*')):
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        files.append({'path': str(path.relative_to(WORK)), 'bytes': path.stat().st_size,
                      'sha256': digest.hexdigest()})
(WORK / 'run_summary.json').write_text(json.dumps({
    'status': 'complete', 'recall_experiments': len(expected_recall),
    'prerank_experiments': len(expected_prerank), 'files': files
}, indent=2))
print('COMPLETE:', len(expected_recall), 'recall experiments;', len(expected_prerank), 'pre-ranking experiments')
'''.splitlines(keepends=True)})
nb['metadata']['kaggle'] = {'isInternetEnabled': False, 'isGpuEnabled': True}
target = HERE / 'recsys_baseline_kaggle.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(target)
