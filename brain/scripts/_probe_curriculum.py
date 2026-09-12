import json, collections, pathlib
root = pathlib.Path('D:/Projects/flylingo')
p = root / 'brain' / 'brain' / 'curriculum' / 'es-en.json'
d = json.loads(p.read_text(encoding='utf-8'))
chs = []
for unit in d.get('units', []):
    for lesson in unit.get('lessons', []):
        for ch in lesson.get('challenges', []):
            chs.append(ch)
print('n challenges', len(chs))
print('keys', list(chs[0].keys()))
print('opt counts', collections.Counter(len(c['options']) for c in chs))
print('types', collections.Counter(c.get('type') for c in chs))
print('difficulty', collections.Counter(c.get('difficulty') for c in chs))
print('correctIndex dist', collections.Counter(int(c['correctIndex']) for c in chs))
print('units', len(d['units']), 'lessons', sum(len(u.get('lessons', [])) for u in d['units']))
print(json.dumps(chs[0], ensure_ascii=False)[:600])
print('answered options sample', chs[0]['options'], chs[0]['answer'], chs[0]['correctIndex'])
# curriculum ordering / drop rules as train.py does
bad = [c for c in chs if len(c['options']) < 4 or int(c['correctIndex']) >= 4]
print('dropped by train.py rules', len(bad))
use = [c for c in chs if len(c['options']) >= 4 and int(c['correctIndex']) < 4]
use = sorted(use, key=lambda c: (int(c.get('difficulty', 1)), str(c['id'])))
print('usable', len(use))
runs = root / 'brain' / 'runs'
print('runs dir', [str(x.name) for x in runs.iterdir()] if runs.exists() else 'MISSING')
rc = runs / 'curriculum' / 'results.json'
if rc.exists():
    r = json.loads(rc.read_text())
    print('prev task', json.dumps(r['task']))
    print('prev config', json.dumps(r['config']))
    print('prev comparison acc', json.dumps({k: v for k, v in r['comparison'].items()}))
    print('prev encoder cfg', json.dumps({k: v for k, v in r.get('encoder', {}).items() if k != 'hash'}))
