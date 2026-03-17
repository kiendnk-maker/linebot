import re

content = open("main.py").read()

# ── Fix 2: reasoning_effort=low cho gpt120b ──────────────────────────────────
old2 = (
    "    # Compound models require last message role to be \"user\"\n"
    "    clean_history = list(history)\n"
    "    while clean_history and clean_history[-1][\"role\"] == \"assistant\":\n"
    "        clean_history.pop()\n"
    "\n"
    "    async with httpx.AsyncClient() as http:\n"
    "        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)\n"
    "        try:\n"
    "            resp = await client.chat.completions.create(\n"
    "                model=model_id,\n"
    "                messages=[{\"role\": \"system\", \"content\": system}] + clean_history,\n"
    "                temperature=0.6,\n"
    "                max_tokens=800,\n"
    "            )"
)
new2 = (
    "    # Compound models require last message role to be \"user\"\n"
    "    clean_history = list(history)\n"
    "    while clean_history and clean_history[-1][\"role\"] == \"assistant\":\n"
    "        clean_history.pop()\n"
    "\n"
    "    # reasoning_effort=low cho gpt120b — tiet kiem token, du cho chat\n"
    "    extra: dict = {}\n"
    "    if model_id == MODEL_REGISTRY[\"gpt120b\"][\"model_id\"]:\n"
    "        extra[\"reasoning_effort\"] = \"low\"\n"
    "\n"
    "    async with httpx.AsyncClient() as http:\n"
    "        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)\n"
    "        try:\n"
    "            resp = await client.chat.completions.create(\n"
    "                model=model_id,\n"
    "                messages=[{\"role\": \"system\", \"content\": system}] + clean_history,\n"
    "                temperature=0.6,\n"
    "                max_tokens=800,\n"
    "                **extra,\n"
    "            )"
)
if old2 in content:
    content = content.replace(old2, new2)
    print("OK 2 - reasoning_effort added")
else:
    print("ERROR 2 - pattern not found")

# ── Fix 3a: Them bang user_profile vao init_db ───────────────────────────────
old3a = (
    "        await db.execute(\n"
    "            \"CREATE TABLE IF NOT EXISTS reminders \""
)
new3a = (
    "        await db.execute(\n"
    "            \"CREATE TABLE IF NOT EXISTS user_profile \"\n"
    "            \"(user_id TEXT PRIMARY KEY, \"\n"
    "            \" name TEXT, occupation TEXT, learning TEXT, notes TEXT)\"\n"
    "        )\n"
    "        await db.execute(\n"
    "            \"CREATE TABLE IF NOT EXISTS reminders \""
)
if old3a in content:
    content = content.replace(old3a, new3a)
    print("OK 3a - user_profile table added")
else:
    print("ERROR 3a - pattern not found")

# ── Fix 3b: Them helper functions ────────────────────────────────────────────
old3b = "async def save_message(user_id: str, role: str, content: str) -> None:"
new3b = (
    "async def get_user_profile(user_id: str) -> dict:\n"
    "    async with aiosqlite.connect(DB_PATH) as db:\n"
    "        async with db.execute(\n"
    "            \"SELECT name, occupation, learning, notes FROM user_profile WHERE user_id = ?\",\n"
    "            (user_id,),\n"
    "        ) as cur:\n"
    "            row = await cur.fetchone()\n"
    "    if not row:\n"
    "        return {}\n"
    "    keys = [\"name\", \"occupation\", \"learning\", \"notes\"]\n"
    "    return {k: v for k, v in zip(keys, row) if v}\n"
    "\n"
    "\n"
    "async def save_user_profile(user_id: str, **kwargs) -> None:\n"
    "    if not kwargs:\n"
    "        return\n"
    "    async with aiosqlite.connect(DB_PATH) as db:\n"
    "        await db.execute(\n"
    "            \"INSERT INTO user_profile (user_id) VALUES (?) \"\n"
    "            \"ON CONFLICT(user_id) DO NOTHING\",\n"
    "            (user_id,),\n"
    "        )\n"
    "        for key, value in kwargs.items():\n"
    "            if key in (\"name\", \"occupation\", \"learning\", \"notes\"):\n"
    "                await db.execute(\n"
    "                    f\"UPDATE user_profile SET {key} = ? WHERE user_id = ?\",\n"
    "                    (value, user_id),\n"
    "                )\n"
    "        await db.commit()\n"
    "\n"
    "\n"
    "async def build_system_prompt(user_id: str, model_key: str) -> str:\n"
    "    base = get_system_prompt(model_key)\n"
    "    profile = await get_user_profile(user_id)\n"
    "    if not profile:\n"
    "        return base\n"
    "    lines = []\n"
    "    if profile.get(\"name\"):       lines.append(\"\\u7528\\u6236\\u59d3\\u540d\\uff1a\" + profile[\"name\"])\n"
    "    if profile.get(\"occupation\"): lines.append(\"\\u8077\\u696d\\uff1a\" + profile[\"occupation\"])\n"
    "    if profile.get(\"learning\"):   lines.append(\"\\u6b63\\u5728\\u5b78\\u7fd2\\uff1a\" + profile[\"learning\"])\n"
    "    if profile.get(\"notes\"):      lines.append(\"\\u5099\\u8a3b\\uff1a\" + profile[\"notes\"])\n"
    "    return base + \"\\n\\n\\u3010\\u7528\\u6236\\u8cc7\\u6599\\u3011\\n\" + \"\\n\".join(lines)\n"
    "\n"
    "\n"
    "async def save_message(user_id: str, role: str, content: str) -> None:"
)
if old3b in content:
    content = content.replace(old3b, new3b)
    print("OK 3b - profile helpers added")
else:
    print("ERROR 3b - pattern not found")

# ── Fix 3c: Them user_id param vao call_groq_text ────────────────────────────
old3c = (
    "async def call_groq_text(\n"
    "    history: list[dict],\n"
    "    model_id: str,\n"
    "    model_key: str = DEFAULT_MODEL_KEY,\n"
    ") -> str:\n"
    "    system = get_system_prompt(model_key)"
)
new3c = (
    "async def call_groq_text(\n"
    "    history: list[dict],\n"
    "    model_id: str,\n"
    "    model_key: str = DEFAULT_MODEL_KEY,\n"
    "    user_id: str | None = None,\n"
    ") -> str:\n"
    "    system = (\n"
    "        await build_system_prompt(user_id, model_key)\n"
    "        if user_id\n"
    "        else get_system_prompt(model_key)\n"
    "    )"
)
if old3c in content:
    content = content.replace(old3c, new3c)
    print("OK 3c - user_id param added")
else:
    print("ERROR 3c - pattern not found")

# ── Fix 3d: Them /profile command ────────────────────────────────────────────
old3d = "    if cmd == \"remind\":"
new3d = (
    "    if cmd == \"profile\":\n"
    "        profile = await get_user_profile(user_id)\n"
    "        if not arg:\n"
    "            if not profile:\n"
    "                return (\n"
    "                    \"Chua co thong tin ca nhan.\\n\"\n"
    "                    \"Cap nhat:\\n\"\n"
    "                    \"/profile name Ten ban\\n\"\n"
    "                    \"/profile job Nghe nghiep\\n\"\n"
    "                    \"/profile learning Tieng Trung B1\\n\"\n"
    "                    \"/profile note Ghi chu them\"\n"
    "                )\n"
    "            lines = [\"Thong tin cua ban:\\n\"]\n"
    "            if profile.get(\"name\"):       lines.append(\"Ten: \" + profile[\"name\"])\n"
    "            if profile.get(\"occupation\"): lines.append(\"Nghe: \" + profile[\"occupation\"])\n"
    "            if profile.get(\"learning\"):   lines.append(\"Dang hoc: \" + profile[\"learning\"])\n"
    "            if profile.get(\"notes\"):      lines.append(\"Ghi chu: \" + profile[\"notes\"])\n"
    "            return \"\\n\".join(lines)\n"
    "        parts2 = arg.split(maxsplit=1)\n"
    "        if len(parts2) < 2:\n"
    "            return \"Dung: /profile name|job|learning|note <noi dung>\"\n"
    "        field, value = parts2[0].lower(), parts2[1]\n"
    "        field_map = {\"name\": \"name\", \"job\": \"occupation\", \"learning\": \"learning\", \"note\": \"notes\"}\n"
    "        if field not in field_map:\n"
    "            return \"Field hop le: name, job, learning, note\"\n"
    "        await save_user_profile(user_id, **{field_map[field]: value})\n"
    "        return \"Da luu \" + field + \": \" + value\n"
    "\n"
    "    if cmd == \"remind\":"
)
if old3d in content:
    content = content.replace(old3d, new3d)
    print("OK 3d - /profile command added")
else:
    print("ERROR 3d - pattern not found")

# ── Fix 3e: Truyen user_id vao call_groq_text ────────────────────────────────
content = content.replace(
    "answer  = await call_groq_text(history, model_id, model_key=model_key)",
    "answer  = await call_groq_text(history, model_id, model_key=model_key, user_id=user_id)"
)
content = content.replace(
    "answer = await call_groq_text(history, model_id, model_key=model_key)",
    "answer = await call_groq_text(history, model_id, model_key=model_key, user_id=user_id)"
)
print("OK 3e - user_id passed to call_groq_text")

open("main.py", "w").write(content)
print("All done - check syntax next")
