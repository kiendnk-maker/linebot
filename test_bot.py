import asyncio
import os

# Set env vars giả để import main không lỗi
os.environ.setdefault("LINE_CHANNEL_SECRET", "test")
os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "test")
os.environ.setdefault("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

from main import (
    classify_query,
    parse_reminder_nlp,
    clean_transcript,
    _needs_realtime,
    init_db,
    save_message,
    get_history_with_summary,
    resolve_model,
    MODEL_REGISTRY,
)

TEST_USER = "test_user_001"

async def run_all():
    print("\n" + "="*50)
    print("BOT FUNCTION TEST")
    print("="*50)

    await init_db()

    # ── Test 1: _needs_realtime ───────────────────────────────
    print("\n[1] _needs_realtime()")
    cases = [
        ("giá bitcoin hôm nay", True),
        ("thời tiết Hà Nội", True),
        ("giải thích machine learning", False),
        ("xin chào", False),
        ("最新消息", True),
    ]
    for text, expected in cases:
        result = _needs_realtime(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' → {result} (expected {expected})")

    # ── Test 2: classify_query ────────────────────────────────
    print("\n[2] classify_query()")
    cases = [
        "xin chào",
        "2 + 2 bằng mấy",
        "viết email xin việc bằng tiếng Trung",
        "phân tích ưu nhược điểm React vs Vue",
        "giải thích quantum entanglement chi tiết",
    ]
    for text in cases:
        result = await classify_query(text)
        model = MODEL_REGISTRY[result]["display"]
        print(f"  '{text}'\n    → {result} [{model}]")

    # ── Test 3: clean_transcript ──────────────────────────────
    print("\n[3] clean_transcript()")
    cases = [
        "thức trình ngày mai",
        "cùng một cuộc họp lúc 14h",
        "báo thức lúc 6h30 sáng mai",
        "hôm nay trời đẹp quá",
    ]
    for text in cases:
        result = await clean_transcript(text)
        status = "✓" if result != text else "~"
        print(f"  {status} '{text}'\n    → '{result}'")

    # ── Test 4: parse_reminder_nlp ────────────────────────────
    print("\n[4] parse_reminder_nlp()")
    cases = [
        "nhắc tôi uống thuốc lúc 8h tối nay",
        "ngày mai 14h có cuộc họp",
        "mỗi ngày 7h sáng nhắc tập thể dục",
        "hôm nay trời đẹp",
        "tôi có bài thuyết trình lúc 10h sáng mai",
    ]
    for text in cases:
        result = await parse_reminder_nlp(TEST_USER, text)
        status = "✓" if result else "~"
        print(f"  {status} '{text}'\n    → {result or 'None (not reminder)'}")

    # ── Test 5: resolve_model ─────────────────────────────────
    print("\n[5] resolve_model()")
    cases = [
        "xin chào",
        "giá vàng hôm nay",
        "viết thư cảm ơn bằng tiếng Trung",
        "tính tích phân của x^2",
        "giải thích lý thuyết tương đối rộng",
    ]
    for text in cases:
        key, model_id = await resolve_model(TEST_USER, text)
        print(f"  '{text}'\n    → {key} [{MODEL_REGISTRY[key]['display']}]")

    # ── Test 6: history ───────────────────────────────────────
    print("\n[6] history flow")
    await save_message(TEST_USER, "user", "tôi tên Minh")
    await save_message(TEST_USER, "assistant", "Xin chào Minh!")
    await save_message(TEST_USER, "user", "tôi học tiếng Trung")
    history = await get_history_with_summary(TEST_USER)
    print(f"  history length: {len(history)} messages")
    for m in history:
        print(f"  [{m['role']}] {m['content'][:50]}")

    print("\n" + "="*50)
    print("TEST COMPLETE")
    print("="*50)

asyncio.run(run_all())
