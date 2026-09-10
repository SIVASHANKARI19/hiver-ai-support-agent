"""Convert REPORT.md into a beautifully formatted REPORT.html."""

from pathlib import Path
import markdown_it

md_path = Path("REPORT.md")
with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# Enable tables and standard formatting
parser = markdown_it.MarkdownIt("commonmark").enable("table")
body_html = parser.render(text)

css = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.65;
    color: #24292e;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 24px;
    background-color: #ffffff;
}
h1, h2, h3, h4 { color: #1a1f2c; margin-top: 28px; margin-bottom: 12px; font-weight: 600; }
h1 { font-size: 26px; border-bottom: 2px solid #eaecef; padding-bottom: 10px; }
h2 { font-size: 20px; border-bottom: 1px solid #eaecef; padding-bottom: 6px; }
h3 { font-size: 16px; }
p, ul, ol { margin-top: 0; margin-bottom: 16px; }
li { margin-bottom: 4px; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 24px 0;
    font-size: 14px;
}
th, td {
    border: 1px solid #dfe2e5;
    padding: 10px 14px;
    text-align: left;
}
th {
    background-color: #f6f8fa;
    font-weight: 600;
}
tr:nth-child(even) { background-color: #fafbfc; }
code {
    background-color: #f3f4f6;
    padding: 3px 6px;
    border-radius: 4px;
    font-family: SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 88%;
    color: #032f62;
}
pre {
    background-color: #f6f8fa;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
}
blockquote {
    border-left: 4px solid #0969da;
    padding: 4px 16px;
    color: #57606a;
    background-color: #f6f8fa;
    margin: 16px 0;
    border-radius: 0 4px 4px 0;
}
hr { border: none; border-top: 1px solid #eaecef; margin: 32px 0; }
"""

full_html = "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n" \
            "<title>Hiver Take-Home Report: AI Support Agent for @AppleSupport</title>\n" \
            "<style>" + css + "</style>\n</head>\n<body>\n" + body_html + "\n</body>\n</html>"

out_path = Path("REPORT.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print("Generated REPORT.html successfully at:", out_path.resolve())
