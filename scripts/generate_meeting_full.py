#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate complete meeting minutes image."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor='#f8fafc')
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
title_bar = FancyBboxPatch((0.3, 11.0), 17.4, 0.8,
                            boxstyle="round,pad=0.03,rounding_size=0.15",
                            facecolor='#1e40af', edgecolor='none', zorder=3)
ax.add_patch(title_bar)
ax.text(9, 11.4, 'AI产品经理模拟面试 - 会议纪要', fontsize=22, fontweight='bold',
        ha='center', va='center', color='white', zorder=4)
ax.text(9, 11.1, '2026年5月10日  |  面试官 vs 许晓冬', fontsize=11,
        ha='center', va='center', color='#bfdbfe', zorder=4)

# Section 1: Core competency requirements
s1 = FancyBboxPatch((0.3, 8.5), 5.8, 2.3,
                     boxstyle="round,pad=0.03,rounding_size=0.12",
                     facecolor='white', edgecolor='#10b981', linewidth=2)
ax.add_patch(s1)
s1_h = FancyBboxPatch((0.3, 10.55), 5.8, 0.3,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor='#10b981', edgecolor='none')
ax.add_patch(s1_h)
ax.text(3.2, 10.7, '核心能力要求', fontsize=12, fontweight='bold', ha='center', va='center', color='white')

items1 = [
    ('1', '用户需求分析', '明确背景、目标、利益'),
    ('2', '产品方案设计', 'AI能力边界、验收指标'),
    ('3', '跨部门协作', '协调前端后端测试设计'),
    ('4', '工程落地能力', '迭代优化、数据反馈'),
]
y = 10.2
for num, title, desc in items1:
    circle = plt.Circle((0.7, y), 0.12, color='#10b981', zorder=3)
    ax.add_patch(circle)
    ax.text(0.7, y, num, fontsize=9, ha='center', va='center', color='white', fontweight='bold', zorder=4)
    ax.text(1.0, y + 0.05, title, fontsize=10, ha='left', va='center', fontweight='bold', color='#1f2937')
    ax.text(1.0, y - 0.12, desc, fontsize=8, ha='left', va='center', color='#6b7280')
    y -= 0.4

# Section 2: AI vs Traditional
s2 = FancyBboxPatch((6.3, 8.5), 5.8, 2.3,
                     boxstyle="round,pad=0.03,rounding_size=0.12",
                     facecolor='white', edgecolor='#8b5cf6', linewidth=2)
ax.add_patch(s2)
s2_h = FancyBboxPatch((6.3, 10.55), 5.8, 0.3,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor='#8b5cf6', edgecolor='none')
ax.add_patch(s2_h)
ax.text(9.2, 10.7, 'AI vs 传统产品区别', fontsize=12, fontweight='bold', ha='center', va='center', color='white')

items2 = [
    ('输入', '不稳定（对话/多轮）', '固定'),
    ('输出', '需保证准确率', '固定结果'),
    ('评估', '99%准确率', '90%即可'),
]
y = 10.15
for item, ai, trad in items2:
    ax.text(6.5, y, item, fontsize=9, ha='left', va='center', fontweight='bold', color='#1f2937')
    ax.text(7.2, y, ai, fontsize=9, ha='left', va='center', color='#8b5cf6')
    ax.text(8.8, y, '->', fontsize=9, ha='center', va='center', color='#9ca3af')
    ax.text(9.3, y, trad, fontsize=9, ha='left', va='center', color='#6b7280')
    y -= 0.4

# Section 3: Suggestions
s3 = FancyBboxPatch((12.3, 8.5), 5.4, 2.3,
                     boxstyle="round,pad=0.03,rounding_size=0.12",
                     facecolor='white', edgecolor='#f59e0b', linewidth=2)
ax.add_patch(s3)
s3_h = FancyBboxPatch((12.3, 10.55), 5.4, 0.3,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor='#f59e0b', edgecolor='none')
ax.add_patch(s3_h)
ax.text(15.0, 10.7, '面试官建议', fontsize=12, fontweight='bold', ha='center', va='center', color='white')

sugs = ['1. 补齐产品知识体系', '2. 理解AI特殊性', '3. 技术优势要发挥', '4. 项目要有数据支撑']
y = 10.1
for s in sugs:
    ax.text(12.5, y, s, fontsize=9, ha='left', va='center', color='#1f2937')
    y -= 0.38

# Section 4: Key Q&A
qa_box = FancyBboxPatch((0.3, 5.8), 17.4, 2.5,
                         boxstyle="round,pad=0.03,rounding_size=0.12",
                         facecolor='white', edgecolor='#3b82f6', linewidth=2)
ax.add_patch(qa_box)
qa_h = FancyBboxPatch((0.3, 8.05), 17.4, 0.3,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor='#3b82f6', edgecolor='none')
ax.add_patch(qa_h)
ax.text(9.0, 8.2, '关键问答', fontsize=12, fontweight='bold', ha='center', va='center', color='white')

qas = [
    ('Q1', 'AI产品经理与传统PM最大的区别？', '输入不稳定、输出需保证准确率、评估标准更高'),
    ('Q2', '为什么想做AI产品而不是技术？', '想更贴近用户需求，平时喜欢观察，职业规划更想跟人打交道'),
    ('Q3', '如何排序产品需求？', '基础对话 > 多模态 > 长期记忆 > 成本优化'),
]
y = 7.7
for q, q_text, a in qas:
    ax.text(0.5, y, q + ':', fontsize=9, ha='left', va='center', fontweight='bold', color='#1e40af')
    ax.text(0.9, y, q_text, fontsize=9, ha='left', va='center', color='#1f2937')
    ax.text(0.9, y - 0.28, '-> ' + a, fontsize=8, ha='left', va='center', color='#6b7280')
    y -= 0.65

# Section 5: Product Workflow
wf_box = FancyBboxPatch((0.3, 3.3), 8.5, 2.3,
                          boxstyle="round,pad=0.03,rounding_size=0.12",
                          facecolor='white', edgecolor='#ec4899', linewidth=2)
ax.add_patch(wf_box)
wf_h = FancyBboxPatch((0.3, 5.35), 8.5, 0.3,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor='#ec4899', edgecolor='none')
ax.add_patch(wf_h)
ax.text(4.55, 5.5, '产品工作流程', fontsize=12, fontweight='bold', ha='center', va='center', color='white')

wfs = [
    '1. 懂传统训练前面的工作流程',
    '2. 研究后续的输出',
    '3. 设计让输出更稳定',
    '4. 判断产品是否搭配错误',
]
y = 5.1
for w in wfs:
    ax.text(0.5, y, w, fontsize=9, ha='left', va='center', color='#1f2937')
    y -= 0.38

# Section 6: Evaluation Metrics
ev_box = FancyBboxPatch((9.0, 3.3), 8.7, 2.3,
                          boxstyle="round,pad=0.03,rounding_size=0.12",
                          facecolor='white', edgecolor='#14b8a6', linewidth=2)
ax.add_patch(ev_box)
ev_h = FancyBboxPatch((9.0, 5.35), 8.7, 0.3,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor='#14b8a6', edgecolor='none')
ax.add_patch(ev_h)
ax.text(13.35, 5.5, '评估标准', fontsize=12, fontweight='bold', ha='center', va='center', color='white')

evs = [
    '断点率：每次工具的准确率',
    '融合率：不同解锁方案对比（BM25等）',
    '融合率目标：81%-91%',
]
y = 5.1
for e in evs:
    ax.text(9.2, y, e, fontsize=9, ha='left', va='center', color='#1f2937')
    y -= 0.45

# Footer
ax.text(9, 2.5, '--------------------------------------------------------------------',
        fontsize=10, ha='center', va='center', color='#cbd5e1')
ax.text(9, 2.1, '面试准备 - 模拟面试记录', fontsize=10, ha='center', va='center',
        color='#64748b', style='italic')
ax.text(9, 1.7, '完整版会议纪要 - 包含所有问题与回答', fontsize=9, ha='center', va='center', color='#94a3b8')

# Decorative
ax.text(0.2, 11.5, '*', fontsize=14, color='#3b82f6', alpha=0.2)
ax.text(17.6, 11.5, '*', fontsize=14, color='#3b82f6', alpha=0.2)
ax.text(0.2, 1.0, '◆', fontsize=12, color='#10b981', alpha=0.2)
ax.text(17.6, 1.0, '◆', fontsize=12, color='#8b5cf6', alpha=0.2)

plt.tight_layout(pad=0.5)
plt.savefig(r'C:\Users\xlab_w\.nanobot\workspace\interview_prep\meeting_minutes_full.png',
            dpi=150, bbox_inches='tight', facecolor='#f8fafc')
print('Complete image saved!')
