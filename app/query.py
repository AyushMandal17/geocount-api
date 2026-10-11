"""Rule-based question parser: 'How many cars?' -> ['small vehicle']."""
import re

CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
    "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter",
    "roundabout", "soccer ball field", "swimming pool",
]

KEYWORDS = {
    "large vehicle": ["large vehicle"], "small vehicle": ["small vehicle"],
    "vehicle": ["small vehicle", "large vehicle"], "car": ["small vehicle"],
    "truck": ["large vehicle"], "bus": ["large vehicle"], "lorry": ["large vehicle"],
    "plane": ["plane"], "airplane": ["plane"], "aircraft": ["plane"], "jet": ["plane"],
    "helicopter": ["helicopter"], "ship": ["ship"], "boat": ["ship"], "vessel": ["ship"],
    "storage tank": ["storage tank"], "oil tank": ["storage tank"], "tank": ["storage tank"],
    "swimming pool": ["swimming pool"], "pool": ["swimming pool"],
    "tennis court": ["tennis court"], "tennis": ["tennis court"],
    "basketball court": ["basketball court"],
    "baseball diamond": ["baseball diamond"], "baseball field": ["baseball diamond"],
    "soccer ball field": ["soccer ball field"], "soccer field": ["soccer ball field"],
    "football field": ["soccer ball field"],
    "ground track field": ["ground track field"], "track field": ["ground track field"],
    "running track": ["ground track field"],
    "harbor": ["harbor"], "harbour": ["harbor"],
    "roundabout": ["roundabout"], "traffic circle": ["roundabout"], "bridge": ["bridge"],
}


def parse_query(question: str):
    """Classes asked about, or None (= report everything detected)."""
    q = (question or "").lower()
    found: list[str] = []
    for kw in sorted(KEYWORDS, key=len, reverse=True):
        pat = rf"\b{re.escape(kw)}(?:s|es)?\b"
        if re.search(pat, q):
            for cls in KEYWORDS[kw]:
                if cls not in found:
                    found.append(cls)
            q = re.sub(pat, " ", q)
    return found or None


def _plural(name: str, n: int) -> str:
    return name if n == 1 else name + "s"


def format_answer(counts: dict) -> str:
    if not counts:
        return "No objects detected."
    if all(n == 0 for n in counts.values()):
        return f"No {', '.join(_plural(k, 2) for k in counts)} detected."
    if len(counts) == 1:
        (name, n), = counts.items()
        return f"There {'is' if n == 1 else 'are'} {n} {_plural(name, n)}."
    parts = [f"{n} {_plural(k, n)}" for k, n in counts.items()]
    return "Detected " + ", ".join(parts[:-1]) + " and " + parts[-1] + "."
