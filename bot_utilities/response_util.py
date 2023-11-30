import aiohttp
from langdetect import detect

def split_response(response, max_length=1999):
    lines = response.splitlines()
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n"
            current_chunk += line

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

async def translate_to_en(text):
    detected_lang = detect(text)
    if detected_lang == "en":
        return text
    API_URL = "https://api.pawan.krd/gtranslate"
    async with aiohttp.ClientSession() as session:
        async with session.get(API_URL, params={"text": text,"from": detected_lang,"to": "en",}) as response:
            data = await response.json()
            translation = data.get("translated")
            return translation