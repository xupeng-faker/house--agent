#!/usr/bin/env python3
"""
并发获取房源信息并导出到文件
从 HF_1 开始，导出指定数量，支持并发请求
"""

import asyncio
import json
import httpx
from pathlib import Path

BASE_URL = "http://7.225.29.223:8080/api/houses"
OUTPUT_FILE = Path(__file__).parent / "houses_export.jsonl"
TOTAL_LIMIT = 2000
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


async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results: dict[int, dict] = {}  # index -> house_data，用于按序写入
    completed = 0

    async def fetch_one(client: httpx.AsyncClient, idx: int):
        nonlocal completed
        house_id = f"HF_{idx}"
        async with semaphore:
            house_data = await fetch_house(client, house_id)
        if house_data is not None:
            results[idx] = house_data
        completed += 1
        if completed % 100 == 0 or completed == TOTAL_LIMIT:
            print(f"  进度: {completed}/{TOTAL_LIMIT}")

    print(f"开始并发获取 {TOTAL_LIMIT} 条房源（并发数: {CONCURRENCY}）...")
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[fetch_one(client, i) for i in range(1, TOTAL_LIMIT + 1)])

    # 按 HF_1, HF_2, ... 顺序写入
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx in range(1, TOTAL_LIMIT + 1):
            if idx in results:
                f.write(json.dumps(results[idx], ensure_ascii=False) + "\n")

    success_count = len(results)
    print(f"\n完成！成功获取 {success_count} 条房源，已导出到: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
