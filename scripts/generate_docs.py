"""Генерирует корпус data/raw/ из синтетических markdown-источников data/sources/.

Чётные по порядку файлы сохраняются в PDF, нечётные — в HTML, чтобы ingestion
работал с обоими форматами. Скрипт нужен только для разработки и в работе
сервиса не участвует. Зависимость: reportlab (requirements-dev.txt).
"""
import re
from pathlib import Path

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate

SRC_DIR = Path(__file__).resolve().parent.parent / "data" / "sources"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Встроенные шрифты PDF (Helvetica, Courier) не содержат кириллицы — нужен Unicode TTF.
_FONT_CANDIDATES = [
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]
BODY_FONT, BOLD_FONT = "Helvetica", "Helvetica-Bold"
for regular_path, bold_path in _FONT_CANDIDATES:
    if Path(regular_path).exists() and Path(bold_path).exists():
        pdfmetrics.registerFont(TTFont("BodyUnicode", regular_path))
        pdfmetrics.registerFont(TTFont("BodyUnicode-Bold", bold_path))
        BODY_FONT, BOLD_FONT = "BodyUnicode", "BodyUnicode-Bold"
        break
else:
    raise RuntimeError(
        "No Cyrillic-capable TTF font found for PDF generation. "
        "Install DejaVu Sans (Linux: apt-get install fonts-dejavu) or run on macOS."
    )

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1c", fontSize=18, leading=22, spaceAfter=14, spaceBefore=6, fontName=BOLD_FONT))
styles.add(ParagraphStyle(name="H2c", fontSize=13, leading=17, spaceAfter=8, spaceBefore=12, fontName=BOLD_FONT))
styles.add(ParagraphStyle(name="H3c", fontSize=11, leading=15, spaceAfter=6, spaceBefore=10, fontName=BOLD_FONT))
styles.add(ParagraphStyle(name="Bodyc", fontSize=10, leading=14, spaceAfter=8, alignment=TA_LEFT, fontName=BODY_FONT))
styles.add(ParagraphStyle(name="Tablec", fontName=BODY_FONT, fontSize=8, leading=11, spaceAfter=10))


def inline_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def md_to_flowables(md_text: str):
    flowables = []
    lines = md_text.splitlines()
    i = 0
    bullet_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer
        if bullet_buffer:
            flowables.append(
                ListFlowable(
                    [ListItem(Paragraph(inline_md(b), styles["Bodyc"])) for b in bullet_buffer],
                    bulletType="bullet",
                    leftIndent=18,
                )
            )
            bullet_buffer = []

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            flush_bullets()
            i += 1
            continue
        if line.startswith("| "):
            flush_bullets()
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            clean = [row for row in table_lines if not re.match(r"^\|[\s\-|]+\|$", row)]
            flowables.append(Paragraph("<br/>".join(c.strip() for c in clean), styles["Tablec"]))
            continue
        if line.startswith("# "):
            flush_bullets()
            flowables.append(Paragraph(inline_md(line[2:]), styles["H1c"]))
        elif line.startswith("## "):
            flush_bullets()
            flowables.append(Paragraph(inline_md(line[3:]), styles["H2c"]))
        elif line.startswith("### "):
            flush_bullets()
            flowables.append(Paragraph(inline_md(line[4:]), styles["H3c"]))
        elif re.match(r"^[-*]\s+", line):
            bullet_buffer.append(re.sub(r"^[-*]\s+", "", line))
        elif re.match(r"^\d+\.\s+", line):
            bullet_buffer.append(re.sub(r"^\d+\.\s+", "", line))
        else:
            flush_bullets()
            flowables.append(Paragraph(inline_md(line), styles["Bodyc"]))
        i += 1
    flush_bullets()
    return flowables


def md_to_html(md_text: str, title: str) -> str:
    lines = md_text.splitlines()
    body = []
    in_ul = False
    in_table = False
    table_rows = []

    def close_ul():
        nonlocal in_ul
        if in_ul:
            body.append("</ul>")
            in_ul = False

    def close_table():
        nonlocal in_table, table_rows
        if in_table:
            rows_html = []
            for idx, r in enumerate(table_rows):
                cells = [c.strip() for c in r.strip("|").split("|")]
                tag = "th" if idx == 0 else "td"
                rows_html.append("<tr>" + "".join(f"<{tag}>{inline_html(c)}</{tag}>" for c in cells) + "</tr>")
            body.append("<table>" + "".join(rows_html) + "</table>")
            table_rows = []
            in_table = False

    def inline_html(t: str) -> str:
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", t)
        t = re.sub(r"`([^`]+?)`", r"<code>\1</code>", t)
        return t

    for line in lines:
        if line.startswith("| "):
            if re.match(r"^\|[\s\-|]+\|$", line):
                continue
            in_table = True
            table_rows.append(line)
            continue
        else:
            close_table()

        if not line.strip():
            close_ul()
            continue
        if line.startswith("# "):
            close_ul()
            body.append(f"<h1>{inline_html(line[2:])}</h1>")
        elif line.startswith("## "):
            close_ul()
            body.append(f"<h2>{inline_html(line[3:])}</h2>")
        elif line.startswith("### "):
            close_ul()
            body.append(f"<h3>{inline_html(line[4:])}</h3>")
        elif re.match(r"^[-*]\s+", line):
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            # Маркер "- " остаётся в тексте пункта: так сгенерированы закоммиченные
            # data/raw/*.html, и их нужно воспроизводить без изменений (см. README).
            body.append(f"<li>{inline_html(line)}</li>")
        else:
            close_ul()
            body.append(f"<p>{inline_html(line)}</p>")
    close_ul()
    close_table()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: Georgia, serif; max-width: 800px; margin: 40px auto; line-height: 1.5; color: #222; }}
h1 {{ font-size: 1.6em; border-bottom: 2px solid #333; padding-bottom: 6px; }}
h2 {{ font-size: 1.25em; margin-top: 1.5em; }}
h3 {{ font-size: 1.05em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #999; padding: 6px 10px; text-align: left; font-size: 0.9em; }}
th {{ background: #eee; }}
</style>
</head>
<body>
{''.join(body)}
</body>
</html>
"""


def main():
    md_files = sorted(SRC_DIR.glob("*.md"))
    for idx, path in enumerate(md_files):
        text = path.read_text(encoding="utf-8")
        title_line = next((ln for ln in text.splitlines() if ln.startswith("# ")), path.stem)
        title = title_line.lstrip("# ").strip()
        stem = path.stem

        if idx % 2 == 0:
            out_path = RAW_DIR / f"{stem}.pdf"
            doc = SimpleDocTemplate(
                str(out_path), pagesize=LETTER,
                leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                topMargin=0.9 * inch, bottomMargin=0.9 * inch,
            )
            doc.build(md_to_flowables(text))
            print(f"PDF  -> {out_path.name}")
        else:
            out_path = RAW_DIR / f"{stem}.html"
            out_path.write_text(md_to_html(text, title), encoding="utf-8")
            print(f"HTML -> {out_path.name}")


if __name__ == "__main__":
    main()
