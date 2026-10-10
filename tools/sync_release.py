"""Copy approved research products from a local AC-Twin workspace.

No collection, model loading, training, inference or statistical adjudication runs.
The research workspace is read only. The destination is this Git checkout.
"""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, json, re, shutil, subprocess, time

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {'__pycache__', '.git', '.pytest_cache', 'node_modules', '.venv',
              'revisions', 'before', 'rendered', 'archive_draft_v1',
              'blueprint_drafts_v0.1', 'legacy_v0.3', '_preview_main_tables'}
SUFFIXES = {'.py', '.ps1', '.sh', '.js', '.mjs', '.json', '.jsonl', '.csv', '.tsv',
            '.txt', '.md', '.log', '.lock', '.sha256', '.yaml', '.yml', '.npz',
            '.npy', '.pt', '.pth', '.ckpt', '.png', '.pdf', '.svg', '.xlsx', '.tiff', '.ini'}
RAW = {'.npz', '.npy', '.pkl', '.pickle', '.tfrecord', '.tfrecords', '.record', '.bin'}
META = ('stage4d_external', 'stage4d_d2e', 'stage4d_d2f', 'stage4d_d2r')
EXTERNAL_THRESHOLD = 10 * 1024 * 1024

def excluded_reason(rel: Path):
    name=rel.as_posix(); parts=rel.parts
    if any(p in SKIP_PARTS or p.startswith('_dup_repair_backup') for p in parts):
        return 'cache_or_duplicate_backup'
    if any(p.startswith('~$') for p in parts):return 'temporary_file'
    external=any(p in META for p in parts)
    if external and rel.suffix.lower() in {'.pkl','.pickle','.tfrecord','.tfrecords','.record'}:
        return 'third_party_scenario_file'
    if external and rel.suffix.lower() in RAW and (
        'train' in parts or re.search(r'(^|[_-])train([_.-]|$)',rel.name,re.I)):
        return 'metadrive_training_data_or_prediction_cache'
    if rel.suffix.lower() not in SUFFIXES:return 'outside_release_formats'
    if rel.suffix=='.md' and (re.match(r'^20\d\d-',rel.name) or any(s in rel.name for s in ['实施计划','研究计划','评审','讨论'])):
        return 'internal_planning_document'
    if 'DESKTOP-' in rel.name:return 'cloud_conflict_copy'
    return None

def digest(p):
    h=hashlib.sha256()
    started=time.monotonic();last=started;done=0;total=p.stat().st_size
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b);done+=len(b);now=time.monotonic()
            if now-last>=20:
                elapsed=now-started;eta=elapsed*(total-done)/max(done,1)
                print(f'HASH phase: {p.name}, {done}/{total} bytes, {100*done/max(total,1):.1f}%, elapsed={elapsed:.0f}s ETA={eta:.0f}s',flush=True);last=now
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    args=ap.parse_args(); source=args.source.resolve()
    assert source!=ROOT and (source/'00_READ_FIRST_AC_TWIN.md').is_file()
    selected={};excluded=[]
    for base in ['code','config','tests','data','experiments','paper/figures','paper/tables']:
        for p in (source/base).rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(source); reason=excluded_reason(rel)
            if reason:
                excluded.append({'path':rel.as_posix(),'reason':reason,'bytes':p.stat().st_size});continue
            selected[rel.as_posix()]=p
    # Include source helpers already distributed at the repository root and legacy.
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in tracked:
        p=source/name
        if name and p.is_file() and (name.startswith('legacy/') or '/' not in name) and p.suffix in {'.py','.txt','.ini'}:
            selected[name]=p
        elif name and not p.exists() and name.startswith(('code/','tests/','paper/')) and (ROOT/name).suffix in {'.py','.ps1','.js','.mjs'}:
            selected[name]=ROOT/name  # Preserve previously released historical source.
    for p in (source/'paper/figures/fig2/icon').glob('*'):
        if p.is_file() and p.suffix in {'.png','.svg'}:selected[p.relative_to(source).as_posix()]=p
    total=sum(p.stat().st_size for p in selected.values());done=0;start=time.monotonic();rows=[];transformations=[]
    print(f'COPY phase: 0/{len(selected)} files, 0.0%, 0/{total} bytes, elapsed=0s ETA=calculating',flush=True)
    last=start
    for i,(name,p) in enumerate(sorted(selected.items()),1):
        target=ROOT/name; target.parent.mkdir(parents=True,exist_ok=True)
        before=p.stat();sha=digest(p)
        if name=='code/drive_stage4d_d2f.py':
            raw=p.read_bytes();portable=raw.replace(b'95%-CI',b'95%%-CI').replace(b'(95% CI)',b'(95%% CI)')
            source_sha=sha;sha=hashlib.sha256(portable).hexdigest()
            if not target.exists() or digest(target)!=sha:target.write_bytes(portable)
            transformations.append({'path':name,'source_sha256':source_sha,'release_sha256':sha,'reason':'Preserve argparse help escaping; experiment logic unchanged.'})
        elif before.st_size>=EXTERNAL_THRESHOLD:
            pass  # Keep large data at its source; publish only a checksum/index.
        elif not target.exists() or target.stat().st_size!=before.st_size or digest(target)!=sha:
            shutil.copy2(p,target)
        if before.st_size<EXTERNAL_THRESHOLD:assert digest(target)==sha,name
        after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),name
        size=len(portable) if name=='code/drive_stage4d_d2f.py' else before.st_size;done+=before.st_size
        category='weights' if p.suffix in {'.pt','.pth','.ckpt'} else 'data' if name.startswith('data/') else 'source' if p.suffix in {'.py','.ps1','.js','.mjs','.sh'} or name.startswith('config/') else 'experiment_output'
        rows.append({'path':name,'bytes':size,'sha256':sha,'category':category,'storage':'external-download' if size>=EXTERNAL_THRESHOLD else 'git'})
        now=time.monotonic()
        if now-last>=10 or i==len(selected):
            elapsed=now-start;eta=elapsed*(total-done)/max(done,1)
            print(f'COPY phase: {i}/{len(selected)} files, {100*i/len(selected):.1f}%, {done}/{total} bytes, elapsed={elapsed:.0f}s ETA={eta:.0f}s',flush=True);last=now
    release=ROOT/'release';release.mkdir(exist_ok=True)
    with (release/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256','category','storage']);w.writeheader();w.writerows(rows)
    (release/'excluded_files.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (release/'source_transformations.json').write_text(json.dumps(transformations,indent=2)+'\n',encoding='utf-8')
    extras=['.gitignore','.gitattributes','README.md','DATA_AVAILABILITY.md','REPRODUCIBILITY.md',
            'release/manifest.csv','release/excluded_files.json','release/summary.json','release/large_data_inventory.json','release/source_transformations.json',
            'tools/check_release.py','tools/sync_release.py']
    allowed={r['path'] for r in rows if r['storage']=='git'}|set(extras)|{n for n in tracked if n and n.startswith('tools/')}
    # Preserve existing source-only files that are absent from the current source tree.
    large_names={r['path'] for r in rows if r['storage']=='external-download'}
    allowed.update(n for n in tracked if n and (ROOT/n).is_file() and n not in large_names)
    (ROOT/'.gitignore').write_text('# Approved research release: deny by default. Regenerate with tools/sync_release.py.\n*\n!*/\n'+''.join('!/'+n.replace('[','\\[').replace(']','\\]')+'\n' for n in sorted(allowed)),encoding='utf-8')
    external=[r for r in rows if r['storage']=='external-download']
    (ROOT/'.gitattributes').write_text('# Preserve frozen artifact bytes and checksums on every platform.\n* -text\n# Large data are obtained separately; no LFS payload in this release.\n',encoding='utf-8')
    # Large data remain with the authors; do not publish hosting addresses.
    inventory=[dict(r, availability='contact_authors',
                    access_note='Large data are not distributed through GitHub. Contact the authors to discuss access.')
               for r in external]
    (release/'large_data_inventory.json').write_text(json.dumps(
        {'access_policy':'CONTACT_AUTHORS_NO_HOSTED_LINK','files':inventory},
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'date':'2026-10-10','files':len(rows),'bytes':sum(r['bytes'] for r in rows),'categories':dict(Counter(r['category'] for r in rows)),
             'external_files':len(external),'external_bytes':sum(r['bytes'] for r in external),'excluded_files':len(excluded),
             'git_product_files':len(rows)-len(external),'git_product_bytes':sum(r['bytes'] for r in rows if r['storage']=='git'),
             'download_status':'CONTACT_AUTHORS_NO_HOSTED_LINK',
             'excluded_metadrive_training_files':sum(r['reason']=='metadrive_training_data_or_prediction_cache' for r in excluded),
             'largest_file_bytes':max(r['bytes'] for r in rows),'source_read_only':True}
    (release/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('COPY SUCCESS: '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
