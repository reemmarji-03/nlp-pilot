from core.english_nlp import load_text_from_file


def test_load_text_from_malformed_pdf_returns_empty_string(tmp_path):
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_text("not a real pdf", encoding="utf-8")

    assert load_text_from_file(str(bad_pdf)) == ""


def test_load_text_from_unsupported_extension_returns_empty_string(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")

    assert load_text_from_file(str(path)) == ""
