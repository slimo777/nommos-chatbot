import re

def load_chunks(path="nommos_menu.txt", granularity="item"):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]

    if granularity == "category":
        return blocks

    chunks = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        head, body = lines[0], lines[1:]
        if not body:
            chunks.append(head)
            continue
        avg_words = sum(len(l.split()) for l in body) / len(body)
        if avg_words > 9:          # prose section, keep whole
            chunks.append(block)
        else:                      # list section, one chunk per item
            chunks.extend(f"{head} — {line}" for line in body)
    return chunks

if __name__ == "__main__":
    for g in ("category", "item"):
        print(f"{g:9} {len(load_chunks(granularity=g)):4} chunks")