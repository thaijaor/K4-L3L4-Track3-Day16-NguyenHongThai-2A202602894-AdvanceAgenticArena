"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU (chỉ một dòng): câu trong `claim["text"]` có xuất hiện NGUYÊN VĂN
trong bằng chứng agent đã thực sự đọc hay không —

    text in ctx.observed_text

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, ĐỪNG trông đợi "nêu cả hai phía" tự động cho
    recall đầy đủ: recall chấm THEO TỪNG required_fact bằng key terms
    của chính fact đó, không phải theo số vế đã trích dẫn — nếu nửa câu
    mô hình thực sự viết ra không phủ hết từ khoá của một fact (mô hình
    ghép câu ở chỗ NÓ chọn, không nhất thiết đúng ranh giới required_fact),
    fact đó vẫn 0 điểm dù trích dẫn đúng. Trên `pub-04-lam-viec-tu-xa` cụ
    thể, trần recall là 0.5 với MỌI harness đúng luật, vì đúng lý do đó —
    đo được, không phải suy đoán. Vẫn nên làm: `abstain: true` sau khi nêu
    cả hai phía được 0.5 recall + trọn 15 điểm honesty, và điểm recall lấy
    theo `max(...)` nên làm cả hai không bao giờ THIỆT — chỉ đừng trông
    đợi nó vượt sàn 0.5 trên brief này.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải xuất hiện nguyên văn trong `ctx.observed_text` và
phải thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài
liệu và không quan sát nào chứa nó.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.layers._match import norm, observed, on_a_line, quotable
from harness.middleware import Middleware

#: Chỗ mô hình dán hai nửa câu của hai nguồn khác nhau.
JOINERS = (" và ",)

#: Giới hạn của scorer: quá 4 claim/tài liệu là REDUNDANT, quá 10 claim là EXCESS.
MAX_CLAIMS_PER_DOC = 4
MAX_CLAIMS = 10

ABSTAIN_ANSWER = "Không đủ căn cứ trong tài liệu đã đọc để trả lời câu hỏi này."


def _source(ctx, seen, text, prefer=None):
    """doc_id của tài liệu có MỘT DÒNG chứa nguyên văn `text`, nếu agent đã thấy câu đó."""
    if not quotable(text) or norm(text) not in seen:
        return None
    if ctx.corpus is None:
        return prefer
    preferred = ctx.corpus.get(prefer) if prefer else None
    if preferred is not None and on_a_line(text, preferred):
        return preferred.doc_id
    for doc in ctx.corpus.docs:
        if on_a_line(text, doc):
            return doc.doc_id
    return None


def _split(ctx, seen, text):
    """Tách câu ghép tại chỗ dán; hai nửa phải thuộc hai tài liệu khác nhau."""
    for joiner in JOINERS:
        start = text.find(joiner)
        while start != -1:
            left, right = text[:start], text[start + len(joiner):]
            left_doc, right_doc = _source(ctx, seen, left), _source(ctx, seen, right)
            if left_doc and right_doc and left_doc != right_doc:
                return [{"text": left, "doc_id": left_doc},
                        {"text": right, "doc_id": right_doc}]
            start = text.find(joiner, start + 1)
    return None


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list):
            return report
        seen = observed(ctx)
        kept, keys, per_doc = [], set(), {}

        def keep(claim):
            key = (norm(claim["text"]), claim.get("doc_id"))
            doc_id = claim.get("doc_id")
            if key in keys or per_doc.get(doc_id, 0) >= MAX_CLAIMS_PER_DOC:
                return
            if len(kept) >= MAX_CLAIMS:
                return
            keys.add(key)
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            kept.append(claim)

        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                continue
            if _source(ctx, seen, claim["text"], claim.get("doc_id")):
                keep(claim)
                continue
            halves = _split(ctx, seen, claim["text"])
            if halves:
                for half in halves:
                    keep(half)
                report["abstain"] = True

        ctx.state["critic_dropped"] = len(claims) - len(kept)
        report["claims"] = kept
        if not kept:
            report["abstain"] = True
            report["citations"] = []
            report["answer"] = ABSTAIN_ANSWER
        else:
            report["citations"] = sorted({c["doc_id"] for c in kept if c.get("doc_id")})
        return report
