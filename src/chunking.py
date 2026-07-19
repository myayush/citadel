import re

import pdfplumber

from src import config

# An RBI numbered heading line: "12. Customer Due Diligence" or
# "12.5 Video based Customer Identification Process". Group 1 is the number and
# becomes section_ref. The title must start with a capital and be short, which is
# what keeps ordinary body sentences that happen to begin with a number out.
HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{3,80})$")


def read_pdf_pages(filepath):
    """Page text extraction lives here because ingest.py needs the same output."""
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def split_by_sections(pages_text):
    """Cuts the document at numbered headings so a citation can point at a clause."""
    sections = []
    current_ref = None
    current_lines = []

    for page in pages_text:
        for line in page.split("\n"):
            match = HEADING_RE.match(line)
            if match:
                if current_ref is not None:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append({"section_ref": current_ref, "content": content})
                current_ref = match.group(1)
                current_lines = [line]
            elif current_ref is not None:
                # Anything before the first heading is cover page and index. Dropped.
                current_lines.append(line)

    if current_ref is not None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({"section_ref": current_ref, "content": content})

    return sections


def enforce_max_size(sections, max_chars=1000, overlap_chars=150):
    """Caps chunk size: MiniLM truncates past 256 tokens (~1000 chars) and long excerpts blow up the prompt."""
    out = []
    for section in sections:
        content = section["content"]
        if len(content) <= max_chars:
            out.append(section)
            continue

        start = 0
        part = 1
        step = max_chars - overlap_chars
        while start < len(content):
            window = content[start:start + max_chars]
            ref = section["section_ref"] + "-part" + str(part)
            out.append({"section_ref": ref, "content": window})
            start = start + step
            part = part + 1
    return out


if __name__ == "__main__":
    filepath = config.PROJECT_ROOT / "data" / "raw" / "kyc_md.pdf"
    pages = read_pdf_pages(filepath)
    sections = split_by_sections(pages)
    chunks = enforce_max_size(sections)

    for chunk in chunks[:5]:
        print(chunk["section_ref"])
    print("sections:", len(sections))
    print("chunks:", len(chunks))