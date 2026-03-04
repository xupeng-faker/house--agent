#!/usr/bin/env python3
"""
从房源 JSONL 汇总所有 tag，导出 tag -> 房源 ID 列表 到 JSON 文件
"""

import argparse
import json
from pathlib import Path

DEFAULT_INPUT = Path(__file__).parent / "houses_export.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "tags_export.json"


def main():
    parser = argparse.ArgumentParser(description="汇总房源 tag 并导出")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT, type=Path, help="房源 JSONL 文件路径")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, type=Path, help="输出 JSON 文件路径")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入文件不存在 {args.input}")
        return 1

    tag_to_houses: dict[str, list[str]] = {}
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            house = json.loads(line)
            house_id = house.get("house_id")
            if not house_id:
                continue
            for tag in house.get("tags", []):
                tag_to_houses.setdefault(tag, []).append(house_id)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tag_to_houses, f, ensure_ascii=False, indent=2)

    print(f"完成！共 {len(tag_to_houses)} 个 tag，已导出到: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
