"""Deterministic Markdown H1/H2/H3 parser for Parent–Child indexing."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .models import ParentChunk


DOC_ID_BY_STEM = {
    "01_订单与物流": "order_logistics",
    "02_退款退货与换货": "refund_return",
    "03_支付与交易异常": "payment_transaction",
    "04_发票账单与订阅": "invoice_subscription",
    "05_账户登录与安全": "account_security",
    "06_会员积分与营销": "membership_marketing",
    "07_商品质量与售后": "after_sales",
    "08_技术故障与错误码": "technical_errors",
    "09_隐私数据与账号管理": "privacy_account",
    "10_客服服务规范与人工升级": "customer_service",
}


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    source: str
    content: str
    parents: List[ParentChunk]


@dataclass
class _Section:
    level: int
    heading: str
    lines: List[str]
    parent_heading: str = ""


def stable_doc_id(path: str | Path) -> str:
    stem = Path(path).stem
    if stem in DOC_ID_BY_STEM:
        return DOC_ID_BY_STEM[stem]
    cleaned = re.sub(r"^\d+[_\- ]*", "", stem)
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", cleaned).strip("_").lower()
    return cleaned or "document"


def _slug(value: str) -> str:
    value = re.sub(r"^\d+(?:\.\d+)*[\s、.：:]*", "", value.strip())
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", value).strip("_").lower()
    return value or "section"


def _heading(line: str):
    match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    return (len(match.group(1)), match.group(2).strip()) if match else None


def _clean_lines(lines: Iterable[str]) -> List[str]:
    result = list(lines)
    while result and not result[0].strip():
        result.pop(0)
    while result and not result[-1].strip():
        result.pop()
    return result


def parse_markdown(path: str | Path, max_parent_chars: int = 1500, min_parent_chars: int = 80) -> ParsedDocument:
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    title = path.stem
    for line in lines:
        parsed = _heading(line)
        if parsed and parsed[0] == 1:
            title = parsed[1]
            break

    sections: List[_Section] = []
    current: _Section | None = None
    current_h2 = ""
    for line in lines:
        parsed = _heading(line)
        if parsed and parsed[0] == 2:
            if current is not None:
                current.lines = _clean_lines(current.lines)
                sections.append(current)
            current_h2 = parsed[1]
            current = _Section(2, parsed[1], [line], parsed[1])
        elif current is not None:
            current.lines.append(line)
    if current is not None:
        current.lines = _clean_lines(current.lines)
        sections.append(current)

    doc_id = stable_doc_id(path)
    parents: List[ParentChunk] = []
    for section in sections:
        section_text = "\n".join(section.lines).strip()
        if not section_text:
            continue
        h3_positions = [
            index for index, line in enumerate(section.lines)
            if (_heading(line) or (0, ""))[0] == 3
        ]
        if len(section_text) <= max_parent_chars or not h3_positions:
            parents.append(_make_parent(
                doc_id, title, path.name, section.heading, [title, section.heading], section_text,
            ))
            continue

        # Preserve the H2/H3 semantic boundary for long sections. A short
        # H2 preamble is attached to the first H3 in that same section.
        starts = [0] + h3_positions
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(section.lines)
            block = _clean_lines(section.lines[start:end])
            if not block:
                continue
            h3 = next(((_heading(line) or (0, ""))[1] for line in block if (_heading(line) or (0, ""))[0] == 3), section.heading)
            block_text = "\n".join(block).strip()
            if len(block_text) > max_parent_chars:
                for part_index, part in enumerate(_paragraph_groups(block_text, max_parent_chars)):
                    parents.append(_make_parent(
                        doc_id,
                        title,
                        path.name,
                        h3,
                        [title, section.heading, h3],
                        part,
                        suffix=f"_{part_index + 1:02d}",
                    ))
            else:
                parents.append(_make_parent(
                    doc_id, title, path.name, h3, [title, section.heading, h3], block_text,
                ))

    # A tiny heading-only fragment is never merged across H2 business areas.
    # Keep it attached to its own section so IDs remain stable and semantics
    # are not silently changed by future documents.
    return ParsedDocument(doc_id, title, path.name, content, parents)


def _paragraph_groups(text: str, max_chars: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text[:max_chars]]
    groups: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars and current and len(current) + len(paragraph) + 2 > max_chars:
            groups.append(current)
            current = paragraph
        elif len(paragraph) > max_chars:
            if current:
                groups.append(current)
                current = ""
            groups.extend(paragraph[index:index + max_chars] for index in range(0, len(paragraph), max_chars))
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        groups.append(current)
    return groups


def _make_parent(
    doc_id: str,
    title: str,
    source: str,
    section: str,
    section_path: Sequence[str],
    content: str,
    suffix: str = "",
) -> ParentChunk:
    slug_path = "#".join(_slug(value) for value in section_path[1:])
    parent_id = f"{doc_id}#{slug_path}{suffix}"
    return ParentChunk(
        doc_id=doc_id,
        parent_id=parent_id,
        title=title,
        section=section,
        section_path=list(section_path),
        source=source,
        content=content,
        metadata={
            "doc_id": doc_id,
            "parent_id": parent_id,
            "title": title,
            "section": section,
            "section_path": list(section_path),
            "source": source,
            "content": content,
        },
    )


def parse_directory(source_dir: str | Path, max_parent_chars: int = 1500, min_parent_chars: int = 80) -> List[ParsedDocument]:
    root = Path(source_dir)
    paths = sorted(root.glob("*.md"))
    return [parse_markdown(path, max_parent_chars, min_parent_chars) for path in paths]
