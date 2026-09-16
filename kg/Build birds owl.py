# -*- coding: utf-8 -*-
from collections import defaultdict
from xml.sax.saxutils import escape

BASE    = 'http://www.qingzanggaoyuan.com/data'
BASE_NS = 'http://www.qingzanggaoyuan.com#'

INPUT_FILE  = 'all_birds_triples.txt'
OUTPUT_FILE = 'Birds_Data.owl'

REL_MAP = {
    '学名为':         ('data', '拉丁名',        None),
    '英文名为':       ('data', '英文名',         None),
    '又称':           ('data', '别名',           None),
    '属于科':         ('obj',  '属于科',         '科'),
    '属于目':         ('obj',  '属于目',         '目'),
    '形态特征':       ('obj',  '形态特征',       '特征'),
    '叫声为':         ('obj',  '叫声为',         '叫声'),
    '体长范围':       ('obj',  '体长范围',       '体长'),
    '栖息于':         ('obj',  '栖息于',         '栖息地'),
    '分布于':         ('obj',  '分布于',         '分布地'),
    '食物有':         ('obj',  '食物有',         '食物'),
    '迁徙类型':       ('obj',  '迁徙类型',       '迁徙情况'),
    '生活海拔':       ('obj',  '生活海拔',       '海拔'),
    'IUCN评级为':     ('obj',  'IUCN评级为',     'IUCN'),
    '国家保护级别为': ('obj',  '国家保护级别为', '中国保护等级'),
}

TARGET_CLASSES = [
    'IUCN', '中国保护等级', '体长', '分布地', '叫声',
    '栖息地', '海拔', '特征', '目', '科', '迁徙情况', '食物'
]

def _safe(s):
    out = []
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
            out.append(ch)
        elif ch.isalnum() or ch in ('-', '.', '_'):
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out)

def uri(val):
    return f'{BASE_NS}{_safe(val)}'

def lit(s):
    return escape(str(s))

def read_triples(path):
    triples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == 3:
                h, r, t = [x.strip() for x in parts]
                if h and r and t:
                    triples.append((h, r, t))
    return triples

def build(triples):
    bird_data  = defaultdict(lambda: defaultdict(list))
    class_inds = defaultdict(set)

    for h, r, t in triples:
        if r in REL_MAP:
            bird_data[h][r].append(t)

    birds = sorted(bird_data.keys())

    for bird in birds:
        for rel, values in bird_data[bird].items():
            if rel not in REL_MAP: continue
            rtype, prop, target_cls = REL_MAP[rel]
            if rtype == 'obj' and target_cls:
                for v in values:
                    class_inds[target_cls].add(v)

    L = []; W = L.append

    # ── File header ──
    W('<?xml version="1.0" encoding="UTF-8"?>')
    W(f'<rdf:RDF xmlns="{BASE_NS}"')
    W(f'     xml:base="{BASE}"')
    W(f'     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"')
    W(f'     xmlns:owl="http://www.w3.org/2002/07/owl#"')
    W(f'     xmlns:xml="http://www.w3.org/XML/1998/namespace"')
    W(f'     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"')
    W(f'     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">')
    W('')
    W(f'    <owl:Ontology rdf:about="{BASE}">')
    W(f'        <owl:imports rdf:resource="http://www.qingzanggaoyuan.com"/>')
    W(f'    </owl:Ontology>')
    W('')

    # ── Non-avian individual ──
    W('    <!-- ===== Non-Bird Individuals ===== -->')
    for cls in TARGET_CLASSES:
        if not class_inds[cls]: continue
        W(f'    <!-- {cls} -->')
        for val in sorted(class_inds[cls]):
            W(f'    <owl:NamedIndividual rdf:about="{uri(val)}">')
            W(f'        <rdf:type rdf:resource="{BASE_NS}{cls}"/>')
            W(f'        <rdfs:label xml:lang="zh">{lit(val)}</rdfs:label>')
            W(f'    </owl:NamedIndividual>')
            W('')

    # ── Individual birds ──
    W('    <!-- ===== 中文名 Individuals ===== -->')
    for bird in birds:
        W(f'    <owl:NamedIndividual rdf:about="{uri(bird)}">')
        W(f'        <rdf:type rdf:resource="{BASE_NS}中文名"/>')
        W(f'        <rdfs:label xml:lang="zh">{lit(bird)}</rdfs:label>')
        for rel, values in bird_data[bird].items():
            if rel not in REL_MAP: continue
            rtype, prop, target_cls = REL_MAP[rel]
            for v in values:
                if rtype == 'data':
                    W(f'        <{prop} rdf:datatype="http://www.w3.org/2001/XMLSchema#string">{lit(v)}</{prop}>')
                else:
                    W(f'        <{prop} rdf:resource="{uri(v)}"/>')
        W(f'    </owl:NamedIndividual>')
        W('')

    W('</rdf:RDF>')
    return '\n'.join(L)

if __name__ == '__main__':
    import os, sys
    if not os.path.exists(INPUT_FILE):
        print(f'❌ 找不到 {INPUT_FILE}'); sys.exit(1)
    print(f'📖 读取: {INPUT_FILE}')
    triples = read_triples(INPUT_FILE)
    print(f'   三元组: {len(triples)} 条')
    print('⚙️  生成 birds1.owl（纯个体，类/属性由1.owl定义）...')
    content = build(triples)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f'✅ 完成: {OUTPUT_FILE}  ({kb:.1f} KB)')
