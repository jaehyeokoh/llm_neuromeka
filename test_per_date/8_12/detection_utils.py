
import asyncio, json
from AI_messages import find_object_3d_properties_message  # 필수

async def detect_objects_with_retries(tool_node, names: list[str], min_area: float = 0.001, max_retries: int = 3) -> list[dict]:
    results = []
    for name in names:
        attempt = 0
        while attempt < max_retries:
            res = await tool_node.ainvoke([find_object_3d_properties_message([name])])
            raw = res[0].content or ""
            try:
                parsed = json.loads(raw)
                items = parsed if isinstance(parsed, list) else [parsed]
                cleaned = []
                for it in items:
                    if isinstance(it, str):
                        it = json.loads(it)
                    if "error" in it:
                        continue
                    if it.get("area", 0) <= min_area:
                        continue
                    cleaned.append(it)
                if cleaned:
                    results.extend(cleaned)
                    break
            except Exception:
                pass
            attempt += 1
            if attempt < max_retries:
                await asyncio.sleep(0.1)
    return results