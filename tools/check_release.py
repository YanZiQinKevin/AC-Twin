"""Verify release products, exclusions, hashes, syntax and Git file sizes."""
from pathlib import Path
import ast,csv,hashlib,json,re,subprocess,time
from sync_release import excluded_reason
ROOT=Path(__file__).resolve().parents[1]
SECRET=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
TEXT={'.py','.ps1','.sh','.js','.mjs','.json','.jsonl','.csv','.tsv','.txt','.md','.log','.lock','.yaml','.yml','.ini','.svg','.sha256'}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic();last=start;problems=[]
 names=set(filter(None,subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split(chr(0))))
 rows=list(csv.DictReader((ROOT/'release/manifest.csv').open(encoding='utf-8')))
 inventory={r['path']:r for r in rows}
 if len(inventory)!=len(rows):problems.append('duplicate manifest paths')
 large=json.loads((ROOT/'release/large_data_inventory.json').read_text(encoding='utf-8'))
 external={r['path']:r for r in rows if r['storage']=='external-download'}
 if large.get('access_policy')!='CONTACT_AUTHORS_NO_HOSTED_LINK':problems.append('incorrect large-data access policy')
 if {r['path'] for r in large['files']}!=set(external):problems.append('large-data inventory differs from manifest')
 for r in large['files']:
  if 'url' in r or r.get('availability')!='contact_authors':problems.append(r['path']+': hosted link or incorrect access policy')
  ref=external.get(r['path'],{})
  if r['sha256']!=ref.get('sha256') or int(r['bytes'])!=int(ref.get('bytes',-1)):problems.append(r['path']+': inventory metadata mismatch')
 total=0
 for i,(name,row) in enumerate(inventory.items(),1):
  p=ROOT/name
  if excluded_reason(Path(name)):problems.append(name+': excluded product in inventory')
  if row['storage']=='external-download':
   if name in names:problems.append(name+': external data selected by Git')
   continue
  if name not in names:problems.append(name+': released product ignored by Git')
  if not p.is_file() or p.stat().st_size!=int(row['bytes']) or digest(p)!=row['sha256']:problems.append(name+': size/hash mismatch')
  total+=int(row['bytes']);now=time.monotonic()
  if now-last>=10:
   elapsed=now-start;eta=elapsed*(len(rows)-i)/max(i,1)
   print(f'VERIFY phase: {i}/{len(rows)}, {100*i/len(rows):.1f}%, elapsed={elapsed:.0f}s ETA={eta:.0f}s',flush=True);last=now
 metadata={'.gitignore','.gitattributes','README.md','DATA_AVAILABILITY.md','REPRODUCIBILITY.md','pytest.ini','conftest.py'}
 for name in sorted(names):
  p=ROOT/name
  if not p.is_file():problems.append(name+': missing file');continue
  if name not in inventory and name not in metadata and not name.startswith(('tools/','release/')) and not (len(p.relative_to(ROOT).parts)==1 and (p.suffix=='.py' or p.name.startswith('requirements'))):problems.append(name+': unregistered file')
  if p.stat().st_size>=100*1024*1024:problems.append(name+': exceeds ordinary Git size limit')
  if p.suffix in TEXT or name in {'.gitignore','.gitattributes'}:
   try:
    raw=p.read_bytes()
    if raw.startswith((b'\xff\xfe',b'\xfe\xff')):s=raw.decode('utf-16')
    else:
     try:s=raw.decode('utf-8-sig')
     except UnicodeDecodeError:
      if p.suffix!='.log':raise
      # Historical terminal logs mix encodings; ASCII credentials still scan.
      # Preserve the original bytes and verify their SHA-256 above.
      s=raw.decode('utf-8',errors='replace')
    if SECRET.search(s):problems.append(name+': credential pattern')
    if p.suffix=='.py':ast.parse(s,filename=name)
    if p.suffix=='.json':json.loads(s)
    if p.suffix=='.jsonl':
     for line in s.splitlines():
      if line.strip():json.loads(line)
   except (ValueError,SyntaxError,UnicodeError) as e:problems.append(name+': parse failure '+type(e).__name__)
 result={'status':'FAIL' if problems else 'PASS','git_files':len(names),'manifest_files':len(rows),'git_product_bytes':total,'external_files':sum(r['storage']=='external-download' for r in rows),'problems':problems}
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
 return bool(problems)
if __name__=='__main__':raise SystemExit(main())
