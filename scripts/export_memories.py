# -*- coding: utf-8 -*-
"""Export 700 test memory metadata to markdown."""
import sys, os
sys.path.insert(0, '.')
code = open('tests/retrieval_bge_eval.py', encoding='utf-8').read()
# Split before def run() so we get all constants/data but not function bodies
exec(code[:code.index('def run():')])

lines = ['# 700条测试记忆元数据\n\n']
lines.append('> 包含4种搜索类型：精确搜索(175)、模糊搜索(175)、相关搜索(175)、混合搜索(175)\n\n')

for stype, label in [('exact','精确搜索'), ('fuzzy','模糊搜索'), ('related','相关搜索'), ('hybrid','混合搜索')]:
    lines.append(f'## {label} (175条)\n\n')
    for m in MEMORIES_AND_QUERIES:
        if m['search_type'] == stype:
            lines.append(f"### {m['id']}  [{m['kind']}] [{m['scope']}]\n")
            lines.append(f"- **slot**: `{m['slot']}`\n")
            lines.append(f"- **summary**: {m['summary']}\n")
            lines.append(f"- **content**: {m['content']}\n")
            lines.append(f"- **tags**: {m['tags']}\n")
            lines.append(f"- **keywords**: {m['keywords']}\n")
            lines.append(f"- **priority**: {m['priority']}\n")
            lines.append(f"- **查询**: {m['query']}\n")
            lines.append('\n')

out = ''.join(lines)
os.makedirs('面试准备', exist_ok=True)
with open('面试准备/700测试记忆.md', 'w', encoding='utf-8') as f:
    f.write(out)
print(f'写入完成，共 {len(MEMORIES_AND_QUERIES)} 条')
