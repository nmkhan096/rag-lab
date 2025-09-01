# # DOCX to JSON chunks

import re
import json
import argparse
from uuid import uuid4
from docx import Document

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])

def split_sections(text, section_headers):
    section_map = {}
    pattern = "|".join([re.escape(h) for h in section_headers])
    splits = re.split(f"(?=({pattern}))", text)

    current_section = None
    for item in splits:
        if item in section_headers:
            current_section = item
            section_map[current_section] = ""
        elif current_section:
            section_map[current_section] += item.strip() + "\n"

    return section_map

def chunk_section_content(section_name, content):
    chunks = []
    raw_chunks = re.split(r"\n{2,}|•", content)
    for raw in raw_chunks:
        text = raw.strip()
        if len(text) > 30:
            chunks.append({
                "id": str(uuid4()),
                "text": text,
                "metadata": {
                    "section": section_name
                }
            })
    return chunks

def parse_resume_to_chunks(file_path, section_headers):
    resume_txt = extract_text_from_docx(file_path)
    section_map = split_sections(resume_txt, section_headers)

    all_chunks = []
    for section, content in section_map.items():
        chunks = chunk_section_content(section, content)
        all_chunks.extend(chunks)

    return all_chunks

def save_chunks_to_json(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a DOCX resume into structured JSON chunks.")
    parser.add_argument("resume_path", type=str, help="Path to the resume DOCX file")
    parser.add_argument("--sections", nargs="+", default=["Work Experience", "Projects", "Skills", "Education"], help="List of section headers to split by")
    parser.add_argument("--output", type=str, default="parsed_resume_chunks.json", help="Path to save the output JSON")

    args = parser.parse_args()

    chunks = parse_resume_to_chunks(args.resume_path, args.sections)
    save_chunks_to_json(chunks, args.output)
