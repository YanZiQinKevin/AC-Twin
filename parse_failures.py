import re

# summary: ordered (status, nodeid)
summary = []
for ln in open('regression_failures.txt', encoding='utf-8', errors='replace'):
    m = re.match(r'^(FAILED|ERROR) (\S+)', ln.strip())
    if m:
        summary.append((m.group(1), m.group(2)))
failed_nodes = [n for s, n in summary if s == 'FAILED']
error_nodes = [n for s, n in summary if s == 'ERROR']

raw = open('regression_causes.txt', encoding='utf-8', errors='replace').read()
fail_sec = raw.split('=' * 10 + ' FAILURES ' + '=' * 10)[1]
parts = re.split(r'=+ ERRORS =+', fail_sec)
fail_blocks_raw, err_sec = parts[0], parts[1] if len(parts) > 1 else ''

def final_cause(block):
    # last `path:line: Exception...` line, else last E line
    loc = re.findall(r'^(\S+\.py):(\d+): (.+)$', block, flags=re.M)
    if loc:
        f, line, msg = loc[-1]
        return f'{f.split(chr(92))[-1]}:{line}: {msg}'[:150]
    es = [l[1:].strip() for l in block.splitlines() if re.match(r'^E\s', l)]
    return (es[-1][:150] if es else 'NO-CAUSE')

# FAILED blocks: no headers; split on blank-line-free? Instead split by the locating line end.
# Each block ends with a line `path:NNN: Exception...`. Split keeping ends.
blocks = re.split(r'(?m)^(?=\S)', fail_blocks_raw)  # blocks start at column 0
blocks = [b for b in re.split(r'\n(?=\S)', fail_blocks_raw) if b.strip()]
# The above is fragile; instead accumulate lines until a locating line closes the block.
def split_blocks(text, headed):
    out, cur = [], []
    for ln in text.splitlines():
        if headed:
            h = re.match(r'^_{2,}\s*ERROR at setup of (\S+)\s*_{2,}\s*$', ln)
            if h:
                if cur:
                    out.append('\n'.join(cur))
                cur = [ln]
                continue
        if re.match(r'^\S+\.py:\d+: ', ln):
            cur.append(ln)
            out.append('\n'.join(cur))
            cur = []
            continue
        if cur or ln.strip():
            cur.append(ln)
    if cur:
        out.append('\n'.join(cur))
    return [b for b in out if re.search(r'^E\s', b, flags=re.M)]

fb = split_blocks(fail_blocks_raw, headed=False)
eb = split_blocks(err_sec, headed=True)

print(f'FAILED summary={len(failed_nodes)} blocks={len(fb)} | ERROR summary={len(error_nodes)} blocks={len(eb)}')
print()
for node, block in zip(failed_nodes, fb):
    print(f'{node}\n    -> {final_cause(block)}')
print()
for node, block in zip(error_nodes, eb):
    print(f'{node}\n    -> {final_cause(block)}')
