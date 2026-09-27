"""Fetch and text-extract a small set of official CNINFO announcement PDFs."""

from __future__ import annotations

import hashlib
import io
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_LLM_TEXT_CHARS = 30_000
SHANGHAI = ZoneInfo("Asia/Shanghai")


class CninfoPdfError(Exception):
    """An external CNINFO link returned unusable PDF data."""


def cninfo_pdf_url(detail_url: str) -> tuple[str, str, str]:
    parsed = urlsplit(detail_url)
    if parsed.hostname not in {"www.cninfo.com.cn", "cninfo.com.cn"}:
        raise CninfoPdfError("公告详情链接不是巨潮资讯官方域名")
    if parsed.path.rstrip("/") != "/new/disclosure/detail":
        raise CninfoPdfError("公告详情链接路径不是巨潮资讯披露详情页")
    query = parse_qs(parsed.query)
    announcement_id = query.get("announcementId", [""])[0]
    announcement_time = query.get("announcementTime", [""])[0]
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})", announcement_time)
    if not announcement_id.isdigit() or date_match is None:
        raise CninfoPdfError("公告详情链接缺少有效 announcementId 或发布日期")
    publication_date = date_match.group(1)
    return (
        f"https://static.cninfo.com.cn/finalpage/{publication_date}/{announcement_id}.PDF",
        announcement_id,
        publication_date,
    )


def _extract_pdf_text(payload: bytes) -> tuple[str, int]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(payload), strict=True)
    except PdfReadError as error:
        raise CninfoPdfError(f"巨潮公告 PDF 解析失败：{error}") from error
    if reader.is_encrypted:
        raise CninfoPdfError("巨潮公告 PDF 已加密，无法提取文本")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise CninfoPdfError(f"巨潮公告 PDF 页数超过限制：{len(reader.pages)}")
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return text, len(reader.pages)


def fetch_cninfo_full_text(
    detail_url: str,
    output_dir: str | Path,
    source_record_id: str,
    title: str,
    published_at: str,
) -> dict[str, object]:
    """Perform one bounded GET, verify PDF bytes, retain hashes and extracted text."""
    pdf_url, announcement_id, publication_date = cninfo_pdf_url(detail_url)
    request = Request(
        pdf_url,
        headers={"User-Agent": "adaptive-mas-research/1.0", "Accept": "application/pdf"},
        method="GET",
    )
    with urlopen(request, timeout=30) as response:
        status = int(response.status)
        content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
        payload = response.read(MAX_PDF_BYTES + 1)
    captured_at = datetime.now(SHANGHAI)
    if status != 200:
        raise CninfoPdfError(f"巨潮 PDF 返回 HTTP {status}")
    if len(payload) > MAX_PDF_BYTES:
        raise CninfoPdfError(f"巨潮公告 PDF 超过 {MAX_PDF_BYTES} 字节限制")
    if content_type != "application/pdf" or not payload.startswith(b"%PDF-"):
        raise CninfoPdfError(f"巨潮响应不是 PDF（Content-Type={content_type!r}）")
    text, page_count = _extract_pdf_text(payload)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / f"{announcement_id}.PDF"
    text_path = output / f"{announcement_id}.txt"
    pdf_path.write_bytes(payload)
    text_path.write_text(text, encoding="utf-8", newline="\n")
    pdf_sha256 = hashlib.sha256(payload).hexdigest()
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "status": "extracted",
        "source_record_id": source_record_id,
        "announcement_id": announcement_id,
        "publication_date": publication_date,
        "title": title,
        "published_at": published_at,
        "captured_at": captured_at.isoformat(),
        "pdf_url": pdf_url,
        "http_status": status,
        "content_type": content_type,
        "pdf_bytes": len(payload),
        "pdf_sha256": pdf_sha256,
        "text_sha256": text_sha256,
        "page_count": page_count,
        "extracted_character_count": len(text),
        "text_extraction_status": "text_extracted" if text else "no_text_layer",
        "llm_text_excerpt": text[:MAX_LLM_TEXT_CHARS],
        "llm_text_truncated": len(text) > MAX_LLM_TEXT_CHARS,
        "pdf_path": str(pdf_path),
        "text_path": str(text_path),
    }
