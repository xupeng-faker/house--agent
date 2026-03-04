#!/usr/bin/env python3
"""
调用房源 API 获取房源信息，汇总所有 tag（tag -> 房源 ID 列表）并导出
从 HF_1 开始递增请求，支持并发
"""

import argparse
import asyncio
import json
import httpx
from pathlib import Path

BASE_URL = "http://7.225.29.223:8080/api/houses"
DEFAULT_OUTPUT = Path(__file__).parent / "tags_export.json"
DEFAULT_LIST_OUTPUT = Path(__file__).parent / "tags_list.json"
DEFAULT_LIMIT = 2000
CONCURRENCY = 50  # 并发数


async def fetch_house(client: httpx.AsyncClient, house_id: str) -> dict | None:
    """获取单个房源信息，成功返回 data，失败返回 None"""
    url = f"{BASE_URL}/{house_id}"
    try:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 0 and "data" in data:
            return data["data"]
        return None
    except (httpx.HTTPError, json.JSONDecodeError, KeyError):
        return None


async def main(limit: int, output: Path, list_output: Path):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tag_to_houses: dict[str, list[str]] = {}
    completed = 0

    async def fetch_one(client: httpx.AsyncClient, idx: int):
        nonlocal completed
        house_id = f"HF_{idx}"
        async with semaphore:
            house_data = await fetch_house(client, house_id)
        if house_data is not None:
            for tag in house_data.get("tags", []):
                tag_to_houses.setdefault(tag, []).append(house_id)
        completed += 1
        if completed % 100 == 0 or completed == limit:
            print(f"  进度: {completed}/{limit}")

    print(f"开始并发获取 {limit} 条房源并汇总 tag（并发数: {CONCURRENCY}）...")
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[fetch_one(client, i) for i in range(1, limit + 1)])

    # 导出 tag -> 房源 ID 映射（压缩无换行）
    with open(output, "w", encoding="utf-8") as f:
        json.dump(tag_to_houses, f, ensure_ascii=False, separators=(",", ":"))

    # 导出纯 tag 列表（压缩无换行）
    tag_list = sorted(tag_to_houses.keys())
    with open(list_output, "w", encoding="utf-8") as f:
        json.dump(tag_list, f, ensure_ascii=False, separators=(",", ":"))

    print(f"\n完成！共 {len(tag_to_houses)} 个 tag")
    print(f"  - tag 映射: {output}")
    print(f"  - 纯 tag 列表: {list_output}")


def run():
    parser = argparse.ArgumentParser(description="调用 API 汇总房源 tag 并导出")
    parser.add_argument("-l", "--limit", type=int, default=DEFAULT_LIMIT, help=f"获取房源数量（默认 {DEFAULT_LIMIT}）")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="tag 映射输出路径")
    parser.add_argument("--list-output", type=Path, default=DEFAULT_LIST_OUTPUT, help="纯 tag 列表输出路径")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.output, args.list_output))


if __name__ == "__main__":
    run()
