"""So khớp trích dẫn theo đúng cách `arena/scorer.py` so: casefold, NFC, gộp khoảng trắng.

Mô hình thật có thể viết thường, xuống dòng, hay gửi tiếng Việt dạng NFD; so khớp
tuyệt đối sẽ xoá nhầm những claim mà scorer vẫn chấm SUPPORTED.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")

#: Ngưỡng của scorer: claim ngắn hơn thì không tài liệu nào "đỡ" được.
MIN_SUPPORT_CHARS = 12
#: Dài hơn thì scorer chấm OVERLONG.
MAX_CLAIM_CHARS = 500


def norm(text) -> str:
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def quotable(text) -> bool:
    return MIN_SUPPORT_CHARS <= len(norm(text)) <= MAX_CLAIM_CHARS


def on_a_line(text, doc) -> bool:
    """`text` nằm nguyên văn trong MỘT dòng của `doc.body`."""
    needle = norm(text)
    return bool(needle) and any(needle in norm(line) for line in doc.body.splitlines())


def observed(ctx) -> str:
    return norm(ctx.observed_text)


def fully_read(doc, seen: str) -> bool:
    """Toàn văn tài liệu đã về nguyên vẹn trong quan sát (không phải snippet)."""
    return norm(doc.body) in seen
