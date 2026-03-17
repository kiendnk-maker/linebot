content = open("main.py").read()

# ── Fix 1: Thêm clean_transcript sau call_groq_whisper ───────────────────────
old1 = (
    "async def call_groq_whisper(audio_bytes: bytes) -> str:\n"
    "    async with httpx.AsyncClient() as http:\n"
    "        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)\n"
    "        try:\n"
    "            result = await client.audio.transcriptions.create(\n"
    "                file=(\"audio.m4a\", audio_bytes),\n"
    "                model=WHISPER_MODEL,\n"
    "            )\n"
    "            return result.text\n"
    "        except Exception as e:\n"
    "            return f\"⚠️ Whisper 錯誤: {str(e)[:150]}\""
)

new1 = (
    "async def call_groq_whisper(audio_bytes: bytes) -> str:\n"
    "    async with httpx.AsyncClient() as http:\n"
    "        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)\n"
    "        try:\n"
    "            result = await client.audio.transcriptions.create(\n"
    "                file=(\"audio.m4a\", audio_bytes),\n"
    "                model=WHISPER_MODEL,\n"
    "            )\n"
    "            return result.text\n"
    "        except Exception as e:\n"
    "            return f\"⚠️ Whisper 錯誤: {str(e)[:150]}\"\n"
    "\n"
    "\n"
    "async def clean_transcript(transcript: str) -> str:\n"
    "    \"\"\"\n"
    "    Dung gpt120b sua loi chinh ta, nghe nham tu Whisper.\n"
    "    Chi sua loi, khong thay doi y nghia hay them noi dung.\n"
    "    \"\"\"\n"
    "    async with httpx.AsyncClient() as http:\n"
    "        client = AsyncGroq(api_key=GROQ_API_KEY, http_client=http)\n"
    "        try:\n"
    "            resp = await client.chat.completions.create(\n"
    "                model=MODEL_REGISTRY[\"gpt120b\"][\"model_id\"],\n"
    "                messages=[{\n"
    "                    \"role\": \"user\",\n"
    "                    \"content\": (\n"
    "                        \"Day la transcript tu nhan dang giong noi tu dong, co the co loi nghe nham, \"\n"
    "                        \"sai chinh ta, hoac tu bi thay the sai nghia.\\n\"\n"
    "                        \"Nhiem vu: sua lai cho dung nghia nhat co the, giu nguyen ngon ngu goc.\\n\"\n"
    "                        \"Vi du loi thuong gap:\\n\"\n"
    "                        \"- 'cung mot' co the la '14h' hoac so gio khac\\n\"\n"
    "                        \"- 'thuc trinh' -> 'thuyet trinh'\\n\"\n"
    "                        \"Chi tra ve cau da sua, khong giai thich, khong them noi dung.\\n\\n\"\n"
    "                        f\"Transcript: {transcript}\"\n"
    "                    ),\n"
    "                }],\n"
    "                temperature=0.0,\n"
    "                max_tokens=300,\n"
    "            )\n"
    "            cleaned = resp.choices[0].message.content.strip()\n"
    "            return cleaned if cleaned else transcript\n"
    "        except Exception:\n"
    "            return transcript"
)

if old1 in content:
    content = content.replace(old1, new1)
    open("main.py", "w").write(content)
    print("OK 1 - clean_transcript added")
else:
    print("ERROR 1 - pattern not found")

# ── Fix 2: Audio handler — clean + reminder check ────────────────────────────
content = open("main.py").read()

old2 = (
    "            if \"⚠️\" not in transcript:\n"
    "                wants_reply = any(\n"
    "                    transcript.strip().lower().startswith(t.lower())\n"
    "                    for t in _REPLY_TRIGGERS\n"
    "                )\n"
    "                if wants_reply:\n"
    "                    clean_text = transcript.strip()\n"
    "                    for t in _REPLY_TRIGGERS:\n"
    "                        if clean_text.lower().startswith(t.lower()):\n"
    "                            clean_text = clean_text[len(t):].strip()\n"
    "                            break\n"
    "                    await save_message(user_id, \"user\", clean_text)\n"
    "                    model_key, model_id = await resolve_model(user_id, clean_text)\n"
    "                    history = await get_history_with_summary(user_id)\n"
    "                    answer  = await call_groq_text(history, model_id, model_key=model_key, user_id=user_id)\n"
    "                    await save_message(user_id, \"assistant\", answer)\n"
    "                    await maybe_summarize(user_id)\n"
    "                    reply = f\"🎤 {clean_text}\\n\\n{answer}\"\n"
    "                else:\n"
    "                    await save_message(user_id, \"user\", f\"[Voice]: {transcript}\")\n"
    "                    reply = f\"🎤 {transcript}\"\n"
    "            else:\n"
    "                reply = transcript"
)

new2 = (
    "            if \"⚠️\" not in transcript:\n"
    "                # Clean transcript truoc khi xu ly\n"
    "                transcript = await clean_transcript(transcript)\n"
    "                logger.info(f'AUDIO cleaned | user={user_id} | text={transcript[:50]!r}')\n"
    "\n"
    "                wants_reply = any(\n"
    "                    transcript.strip().lower().startswith(t.lower())\n"
    "                    for t in _REPLY_TRIGGERS\n"
    "                )\n"
    "                if wants_reply:\n"
    "                    clean_text = transcript.strip()\n"
    "                    for t in _REPLY_TRIGGERS:\n"
    "                        if clean_text.lower().startswith(t.lower()):\n"
    "                            clean_text = clean_text[len(t):].strip()\n"
    "                            break\n"
    "                    # Check reminder truoc khi goi LLM\n"
    "                    reminder_reply = await parse_reminder_nlp(user_id, clean_text)\n"
    "                    logger.info(f'REMINDER wants_reply | user={user_id} | result={reminder_reply is not None}')\n"
    "                    if reminder_reply:\n"
    "                        reply = f\"🎤 {clean_text}\\n\\n{reminder_reply}\"\n"
    "                    else:\n"
    "                        await save_message(user_id, \"user\", clean_text)\n"
    "                        model_key, model_id = await resolve_model(user_id, clean_text)\n"
    "                        history = await get_history_with_summary(user_id)\n"
    "                        answer  = await call_groq_text(history, model_id, model_key=model_key, user_id=user_id)\n"
    "                        await save_message(user_id, \"assistant\", answer)\n"
    "                        await maybe_summarize(user_id)\n"
    "                        reply = f\"🎤 {clean_text}\\n\\n{answer}\\n\\n[{MODEL_REGISTRY[model_key]['display']}]\"\n"
    "                else:\n"
    "                    # Transcribe only — clean + check reminder\n"
    "                    reminder_reply = await parse_reminder_nlp(user_id, transcript)\n"
    "                    logger.info(f'REMINDER transcribe | user={user_id} | result={reminder_reply is not None}')\n"
    "                    if reminder_reply:\n"
    "                        reply = f\"🎤 {transcript}\\n\\n{reminder_reply}\"\n"
    "                    else:\n"
    "                        await save_message(user_id, \"user\", f\"[Voice]: {transcript}\")\n"
    "                        reply = f\"🎤 {transcript}\"\n"
    "            else:\n"
    "                reply = transcript"
)

if old2 in content:
    content = content.replace(old2, new2)
    open("main.py", "w").write(content)
    print("OK 2 - audio handler updated")
else:
    print("ERROR 2 - pattern not found")

print("All done")
