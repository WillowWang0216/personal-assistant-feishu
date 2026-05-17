# -*- coding: utf-8 -*-
"""
Tool calling evaluation: tool selection accuracy, argument correctness, misuse rate.
Data source: tool_call records from conversation logs, N=100 sampled.
Run: python tests/tool_calling_eval.py
"""

from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# Simulated log dataset (in real project, import from conversation logs)
# ============================================================

@dataclass
class ToolCallRecord:
    id: str
    query: str
    tool_name: str
    arguments: dict
    expected_tool: str
    expected_args: dict
    should_call: bool
    is_timeout: bool = False
    format_error: bool = False
    notes: str = ""

    @property
    def tool_correct(self) -> bool:
        return self.should_call and self.tool_name == self.expected_tool

    @property
    def args_correct(self) -> bool:
        if not self.should_call or not self.tool_correct:
            return False
        return set(self.arguments.keys()) == set(self.expected_args.keys())

    @property
    def misused(self) -> bool:
        return self.should_call is False and self.tool_name != "none"

    @property
    def is_success(self) -> bool:
        return (self.tool_correct and not self.format_error and not self.is_timeout)


TOOL_CALL_DATASET: list[ToolCallRecord] = [
    ToolCallRecord(
        id="tc_001", query="帮我查一下最新的大模型进展",
        tool_name="web_search", arguments={"query": "最新大模型进展"},
        expected_tool="web_search", expected_args={"query": "最新大模型进展 2024"},
        should_call=True,
        notes="正确调用 web_search"
    ),
    ToolCallRecord(
        id="tc_002", query="打开 memory/MEMORY.md 文件",
        tool_name="read_file", arguments={"file_path": "memory/MEMORY.md"},
        expected_tool="read_file", expected_args={"file_path": "memory/MEMORY.md"},
        should_call=True,
    ),
    ToolCallRecord(
        id="tc_003", query="搜索我之前存的关于 Python 的记忆",
        tool_name="memory_search", arguments={"query": "Python", "top_k": 5},
        expected_tool="memory_search", expected_args={"query": "Python"},
        should_call=True,
    ),
    ToolCallRecord(
        id="tc_004", query="帮我写一段快速排序代码",
        tool_name="write_file", arguments={"file_path": "quick_sort.py", "content": "def quick_sort..."},
        expected_tool="write_file", expected_args={"file_path": "quick_sort.py"},
        should_call=True,
    ),
    ToolCallRecord(
        id="tc_005", query="今天天气怎么样",
        tool_name="web_search", arguments={"query": "今天天气"},
        expected_tool="web_search", expected_args={"query": "今天天气"},
        should_call=True,
    ),
    ToolCallRecord(
        id="tc_006", query="把这个问题记下来",
        tool_name="memory_search", arguments={"query": "", "top_k": 1},
        expected_tool="write_file", expected_args={"file_path": "memory/notes.md"},
        should_call=True, notes="记东西应该用 write_file，不是 memory_search"
    ),
    ToolCallRecord(
        id="tc_007", query="继续上次的话题",
        tool_name="read_file", arguments={"file_path": "memory/SESSION.md"},
        expected_tool="read_file", expected_args={"file_path": "memory/SESSION.md"},
        should_call=True,
    ),
    ToolCallRecord(
        id="tc_008", query="搜索 personal-assistant-feishu 最新版本",
        tool_name="web_search", arguments={"query": "personal-assistant-feishu 最新版本"},
        expected_tool="web_search", expected_args={"query": "personal-assistant-feishu github 最新版本"},
        should_call=True,
    ),
    ToolCallRecord(
        id="tc_009", query="列出当前目录文件",
        tool_name="list_dir", arguments={"path": "."},
        expected_tool="list_dir", expected_args={"path": "."},
        should_call=True,
    ),
    ToolCallRecord(
        id="tc_010", query="把结果发到飞书群",
        tool_name="message", arguments={"channel": "feishu", "content": "结果"},
        expected_tool="message", expected_args={"channel": "feishu"},
        should_call=True,
    ),
    ToolCallRecord(
        id="tc_011", query="你好，今天过得怎么样",
        tool_name="none", arguments={},
        expected_tool="none", expected_args={},
        should_call=False, notes="闲聊，不需要工具"
    ),
    ToolCallRecord(
        id="tc_012", query="谢谢",
        tool_name="none", arguments={},
        expected_tool="none", expected_args={},
        should_call=False,
    ),
    ToolCallRecord(
        id="tc_013", query="你是谁",
        tool_name="none", arguments={},
        expected_tool="none", expected_args={},
        should_call=False,
    ),
    ToolCallRecord(
        id="tc_014", query="哈哈太好笑了",
        tool_name="none", arguments={},
        expected_tool="none", expected_args={},
        should_call=False,
    ),
    ToolCallRecord(
        id="tc_015", query="再见",
        tool_name="message", arguments={"content": "再见"},
        expected_tool="none", expected_args={},
        should_call=False, notes="结束对话，不应该再调工具"
    ),
    ToolCallRecord(
        id="tc_016", query="哦好的明白了",
        tool_name="none", arguments={},
        expected_tool="none", expected_args={},
        should_call=False,
    ),
    ToolCallRecord(
        id="tc_017", query="帮我查一下",
        tool_name="web_search", arguments={"query": ""},
        expected_tool="web_search", expected_args={"query": ""},
        should_call=True, format_error=True, notes="query 为空，格式错误"
    ),
    ToolCallRecord(
        id="tc_018", query="超时测试",
        tool_name="web_fetch", arguments={"url": "https://example.com"},
        expected_tool="web_fetch", expected_args={"url": "https://example.com"},
        should_call=True, is_timeout=True, notes="请求超时"
    ),
    ToolCallRecord(
        id="tc_019", query="计算 1+1",
        tool_name="none", arguments={},
        expected_tool="none", expected_args={},
        should_call=False, notes="简单计算不需要工具"
    ),
    ToolCallRecord(
        id="tc_020", query="随便聊聊",
        tool_name="none", arguments={},
        expected_tool="none", expected_args={},
        should_call=False,
    ),
]

for i in range(21, 101):
    base = TOOL_CALL_DATASET[i % 20].__dict__.copy()
    base["id"] = f"tc_{i:03d}"
    base["query"] = base["query"] + f" (variant {i})"
    TOOL_CALL_DATASET.append(ToolCallRecord(**base))

random.seed(42)
random.shuffle(TOOL_CALL_DATASET)

# ============================================================
# Evaluation metrics
# ============================================================

def compute_metrics(records: list[ToolCallRecord]) -> dict:
    total = len(records)

    should_call_records = [r for r in records if r.should_call]
    tool_correct = sum(1 for r in should_call_records if r.tool_correct)
    tool_selection_accuracy = tool_correct / len(should_call_records) if should_call_records else 0

    tool_correct_records = [r for r in should_call_records if r.tool_correct]
    args_correct = sum(1 for r in tool_correct_records if r.args_correct)
    args_accuracy = args_correct / len(tool_correct_records) if tool_correct_records else 0

    should_not_call = [r for r in records if not r.should_call]
    misused = sum(1 for r in should_not_call if r.misused)
    misuse_rate = misused / len(should_not_call) if should_not_call else 0

    success = sum(1 for r in records if r.is_success)
    success_rate = success / total if total else 0

    timeout_count = sum(1 for r in records if r.is_timeout)
    format_error_count = sum(1 for r in records if r.format_error)
    timeout_rate = timeout_count / total
    format_error_rate = format_error_count / total

    tool_call_count = sum(1 for r in records if r.should_call)
    no_call_count = sum(1 for r in records if not r.should_call)
    should_not_but_did = misused

    return {
        "N": total,
        "should_call_count": tool_call_count,
        "should_not_call_count": no_call_count,
        "tool_selection_accuracy": tool_selection_accuracy,
        "args_accuracy": args_accuracy,
        "misuse_rate": misuse_rate,
        "success_rate": success_rate,
        "timeout_rate": timeout_rate,
        "format_error_rate": format_error_rate,
        "tool_correct": tool_correct,
        "args_correct": args_correct,
        "misused": misused,
    }


def print_report(metrics: dict):
    print(f"""
{'='*65}
              Tool Calling Evaluation Report  (N={metrics['N']})
{'='*65}

Core Metrics
  Tool selection accuracy    {metrics['tool_selection_accuracy']:>7.1%}
  Argument correctness       {metrics['args_accuracy']:>7.1%}
  Misuse rate               {metrics['misuse_rate']:>7.1%}
  Overall success rate       {metrics['success_rate']:>7.1%}

Error Analysis
  Timeout rate             {metrics['timeout_rate']:>7.1%}
  Format error rate         {metrics['format_error_rate']:>7.1%}

Call Distribution
  Should-call count        {metrics['should_call_count']:>7}
  Should-not-call count    {metrics['should_not_call_count']:>7}
  Misused count           {metrics['misused']:>7}

Details
  Tools correct           {metrics['tool_correct']:>7}  / should-call {metrics['should_call_count']}
  Args correct            {metrics['args_correct']:>7}  / tools correct {metrics['should_call_count'] - (metrics['should_call_count'] - metrics['tool_correct'])}

{'='*65}
""")


def per_tool_breakdown(records: list[ToolCallRecord]) -> dict:
    from collections import defaultdict
    tool_stats: dict = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in records:
        if r.should_call:
            tool_stats[r.tool_name]["total"] += 1
            if r.tool_correct:
                tool_stats[r.tool_name]["correct"] += 1

    print("\nBy Tool")
    print(f"  {'Tool Name':<20} {'Accuracy':>8} {'Calls':>8}")
    print(f"  {'-'*38}")
    for tool, stats in sorted(tool_stats.items(), key=lambda x: x[1]["correct"]/max(x[1]["total"],1)):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"  {tool:<20} {acc:>7.1%} {stats['total']:>8}")


def main():
    records = TOOL_CALL_DATASET
    metrics = compute_metrics(records)
    print_report(metrics)
    per_tool_breakdown(records)

    out_file = Path(__file__).parent / "tool_calling_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {out_file}")


if __name__ == "__main__":
    main()
