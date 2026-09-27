import hashlib

from adaptive_mas.data import cninfo_full_text
DETAIL_URL = (
    "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=600000&"
    "announcementId=1225570810&orgId=gssh0600000&announcementTime=2026-09-18+00%3A00%3A00"
)


def test_cninfo_detail_link_resolves_to_official_static_pdf():
    url, announcement_id, publication_date = cninfo_full_text.cninfo_pdf_url(DETAIL_URL)

    assert url == "https://static.cninfo.com.cn/finalpage/2026-09-18/1225570810.PDF"
    assert announcement_id == "1225570810"
    assert publication_date == "2026-09-18"


def test_bounded_fetch_checks_pdf_and_records_source_and_hashes(monkeypatch, tmp_path):
    payload = b"%PDF-test-payload"
    monkeypatch.setattr(
        cninfo_full_text,
        "urlopen",
        lambda request, timeout: _Response(payload),
    )
    monkeypatch.setattr(cninfo_full_text, "_extract_pdf_text", lambda value: ("公告全文", 1))

    result = cninfo_full_text.fetch_cninfo_full_text(
        DETAIL_URL,
        tmp_path,
        "announcement-source-id",
        "标题",
        "2026-09-18T00:00:00+08:00",
    )

    assert result["status"] == "extracted"
    assert result["source_record_id"] == "announcement-source-id"
    assert result["pdf_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["text_sha256"] == hashlib.sha256("公告全文".encode()).hexdigest()
    assert result["extracted_character_count"] == 4
    assert (tmp_path / "1225570810.PDF").read_bytes() == payload
    assert (tmp_path / "1225570810.txt").read_text(encoding="utf-8") == "公告全文"


class _Response:
    status = 200
    headers = {"Content-Type": "application/pdf"}

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, limit):
        return self.payload[:limit]
