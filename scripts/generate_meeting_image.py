#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate meeting minutes image using matplotlib."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up Chinese font
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 9), facecolor='#f8fafc')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Title bar
title_bar = FancyBboxPatch((0.2, 7.8), 15.6, 1.0,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor='#1e40af', edgecolor='none', zorder=3)
ax.add_patch(title_bar)

# Title text
ax.text(8, 8.3, 'AI产品经理模拟面试', fontsize=28, fontweight='bold',
        ha='center', va='center', color='white', zorder=4)
ax.text(8, 8.0, '会议纪要  |  2026年5月10日', fontsize=14,
        ha='center', va='center', color='#bfdbfe', zorder=4)

# Participants box
participants_box = FancyBboxPatch((0.5, 7.0), 3.0, 0.6,
                                   boxstyle="round,pad=0.03,rounding_size=0.1",
                                   facecolor='#dbeafe', edgecolor='#3b82f6', linewidth=1)
ax.add_patch(participants_box)
ax.text(2.0, 7.35, '参会人员', fontsize=11, ha='center', va='center', fontweight='bold', color='#1e40af')
ax.text(2.0, 7.12, '面试官  |  许晓冬（候选人）', fontsize=10, ha='center', va='center', color='#1e40af')

# Section 1: Core competencies
s1_box = FancyBboxPatch((0.5, 4.5), 4.7, 2.3,
                         boxstyle="round,pad=0.03,rounding_size=0.15",
                         facecolor='white', edgecolor='#e2e8f0', linewidth=1.5)
ax.add_patch(s1_box)

# Section 1 header
s1_header = FancyBboxPatch((0.5, 6.55), 4.7, 0.35,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            facecolor='#10b981', edgecolor='none')
ax.add_patch(s1_header)
ax.text(2.85, 6.75, '核心能力要求', fontsize=13, fontweight='bold',
        ha='center', va='center', color='white')

# Core competencies content
competencies = [
    ('1', '用户需求分析', '明确背景、目标、利益'),
    ('2', '产品方案设计', 'AI能力边界、验收指标'),
    ('3', '跨部门协作', '协调前端后端测试设计'),
    ('4', '工程落地能力', '迭代优化、数据反馈'),
]
y_pos = 6.2
for num, title, desc in competencies:
    circle = plt.Circle((1.0, y_pos), 0.18, color='#10b981', zorder=3)
    ax.add_patch(circle)
    ax.text(1.0, y_pos, num, fontsize=10, ha='center', va='center',
            color='white', fontweight='bold', zorder=4)
    ax.text(1.4, y_pos + 0.08, title, fontsize=11, ha='left', va='center',
            fontweight='bold', color='#1f2937')
    ax.text(1.4, y_pos - 0.15, desc, fontsize=9, ha='left', va='center', color='#6b7280')
    y_pos -= 0.5

# Section 2: AI vs Traditional
s2_box = FancyBboxPatch((5.5, 4.5), 4.7, 2.3,
                         boxstyle="round,pad=0.03,rounding_size=0.15",
                         facecolor='white', edgecolor='#e2e8f0', linewidth=1.5)
ax.add_patch(s2_box)

s2_header = FancyBboxPatch((5.5, 6.55), 4.7, 0.35,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            facecolor='#8b5cf6', edgecolor='none')
ax.add_patch(s2_header)
ax.text(7.85, 6.75, 'AI vs 传统产品', fontsize=13, fontweight='bold',
        ha='center', va='center', color='white')

# Comparison content
comp_items = [
    ('输入', '不稳定（对话/多轮）', '固定（标签）'),
    ('输出', '需保证准确率', '固定结果'),
    ('评估', '更高（90%-99%）', '90%即可'),
]
y_pos = 6.15
for item, ai, trad in comp_items:
    ax.text(5.8, y_pos, item, fontsize=10, ha='left', va='center', fontweight='bold', color='#1f2937')
    ax.text(6.5, y_pos, ai, fontsize=9, ha='left', va='center', color='#8b5cf6')
    ax.text(8.2, y_pos, '→', fontsize=10, ha='center', va='center', color='#9ca3af')
    ax.text(8.7, y_pos, trad, fontsize=9, ha='left', va='center', color='#6b7280')
    y_pos -= 0.5

# Section 3: Suggestions
s3_box = FancyBboxPatch((10.5, 4.5), 5.0, 2.3,
                         boxstyle="round,pad=0.03,rounding_size=0.15",
                         facecolor='white', edgecolor='#e2e8f0', linewidth=1.5)
ax.add_patch(s3_box)

s3_header = FancyBboxPatch((10.5, 6.55), 5.0, 0.35,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            facecolor='#f59e0b', edgecolor='none')
ax.add_patch(s3_header)
ax.text(13.0, 6.75, '面试官建议', fontsize=13, fontweight='bold',
        ha='center', va='center', color='white')

suggestions = [
    '• 补齐产品知识体系',
    '• 理解AI特殊性',
    '• 控制模型清晰理解用户需求',
    '• 输出用户满意的结果',
]
y_pos = 6.1
for sug in suggestions:
    ax.text(10.8, y_pos, sug, fontsize=10, ha='left', va='center', color='#1f2937')
    y_pos -= 0.38

# Bottom summary
summary_bar = FancyBboxPatch((0.5, 2.8), 15.0, 1.4,
                              boxstyle="round,pad=0.03,rounding_size=0.15",
                              facecolor='#fef3c7', edgecolor='#fbbf24', linewidth=1.5)
ax.add_patch(summary_bar)
ax.text(8, 3.95, '关键问答摘要', fontsize=12, fontweight='bold',
        ha='center', va='center', color='#92400e')

qas = [
    'Q: AI产品经理与传统产品经理最大的区别是什么？',
    '   A: 输入不稳定、输出需保证准确率、评估标准更高，需要理解AI的特殊性',
]
ax.text(1.0, 3.5, qas[0], fontsize=10, ha='left', va='center', color='#78350f')
ax.text(1.0, 3.1, qas[1], fontsize=10, ha='left', va='center', color='#92400e')

# Footer
ax.text(8, 2.2, '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
        fontsize=10, ha='center', va='center', color='#cbd5e1')
ax.text(8, 1.7, '面试准备 · 模拟面试记录', fontsize=11, ha='center', va='center',
        color='#64748b', style='italic')

# Decorative elements
ax.text(0.3, 1.0, '*', fontsize=16, color='#3b82f6', alpha=0.3)
ax.text(15.5, 1.0, '*', fontsize=16, color='#3b82f6', alpha=0.3)
ax.text(0.3, 8.5, '◆', fontsize=14, color='#10b981', alpha=0.3)
ax.text(15.5, 8.5, '◆', fontsize=14, color='#8b5cf6', alpha=0.3)

plt.tight_layout(pad=0.5)
plt.savefig(r'C:\Users\xlab_w\.nanobot\workspace\interview_prep\meeting_minutes.png',
            dpi=150, bbox_inches='tight', facecolor='#f8fafc')
print('Image saved successfully!')
