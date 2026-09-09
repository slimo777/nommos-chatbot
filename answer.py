import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

from search import search

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.5-flash"
REWRITE_MODEL = "gemini-3.5-flash-lite"

BUSY = ("Nous recevons beaucoup de demandes en ce moment. "
        "Merci d'appeler le +212 667 215 070. / "
        "We're very busy right now — please call +212 667 215 070.")

SYSTEM = """You are the assistant for NOMMOS, a restaurant lounge in Tangier.

Answer ONLY from the CONTEXT provided. It comes from the restaurant's own
menu and information sheet.
- You are the NOMMOS assistant and nothing else. Never discuss being an AI,
  a chatbot, a model, or who created you. If asked, say you're here to help
  with the menu, prices and reservations.

Rules:
- If the answer is not in the context, say you don't have that information
  and suggest calling +212 667 215 070. Never guess, never invent.
- Never invent a dish, a price, or a fact. Prices are in dirhams (DH).
- Do not add adjectives the menu does not use. Do not call a dish famous,
  special or popular unless the menu says so.
- Reply in the same language the customer used (English, French or Arabic).
  Write Arabic in correct Arabic script only.
- Be brief and warm, like a good waiter. Two or three sentences.
- Never mention section names, categories, or that you are reading a document.
"""

REWRITE = """Rewrite the user's last message as a standalone question that
makes sense on its own, without the conversation.

- Keep the original language.
- If it is already standalone, return it unchanged.
- Return ONLY the question, nothing else.
"""


def generate(contents, system, tries=3):
    for attempt in range(tries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system),
            )
            return response.text
        except errors.ServerError as e:
            print(f"[generate] server error {getattr(e, 'code', None)}, retrying")
            time.sleep(1.5 * (attempt + 1))
        except errors.ClientError as e:
            if getattr(e, "code", None) != 429:
                raise
            print("[generate] rate limited")
            return None
    return None


def rewrite(question, history):
    if not history:
        return question
    convo = "\n".join(f"{t['role']}: {t['text']}" for t in history[-4:])
    result = generate(f"{convo}\nuser: {question}", REWRITE)
    return result.strip() if result else question


def answer(question, history=None, k=8, debug=False):
    history = history or []
    standalone = rewrite(question, history)
    hits = search(standalone, k=k)
    if not hits:
        return BUSY

    if debug:
        for score, text in hits:
            print(f"   {score:.3f}  {text.splitlines()[0][:45]}")

    context = "\n\n---\n\n".join(text for _, text in hits)

    convo = "\n".join(f"{t['role']}: {t['text']}" for t in history[-4:])
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"CONVERSATION SO FAR:\n{convo or '(none)'}\n\n"
        f"CUSTOMER QUESTION: {question}"
    )
    result = generate(prompt, SYSTEM)
    return result if result else BUSY


if __name__ == "__main__":
    history = []
    for q in ["how much is the Nommos?", "and the poke bowl?"]:
        print(f"\n=== {q}")
        reply = answer(q, history)
        print(reply)
        history.append({"role": "user", "text": q})
        history.append({"role": "assistant", "text": reply})