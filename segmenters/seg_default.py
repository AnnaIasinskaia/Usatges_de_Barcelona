"""Default segmenter: split by paragraphs, merge to target word count."""
from .seg_common import clean_text, is_apparatus_line, validate_segments


def segment_default(text, source_name, max_segment_words=150):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip() and len(p.strip()) > 15]
    paragraphs = [p for p in paragraphs if not is_apparatus_line(p)]
    segments = []
    current_text = ""
    current_words = 0
    seg_idx = 1
    for para in paragraphs:
        para = clean_text(para)
        if len(para) < 10:
            continue
        words = len(para.split())
        if current_words + words > max_segment_words and current_text:
            if len(current_text.strip()) >= 30:
                segments.append((f"{source_name}_S{seg_idx}", current_text.strip()))
                seg_idx += 1
            current_text = para + " "
            current_words = words
        else:
            current_text += para + " "
            current_words += words
    if current_text.strip() and len(current_text.strip()) >= 30:
        segments.append((f"{source_name}_S{seg_idx}", current_text.strip()))
    return validate_segments(segments, source_name)


if __name__ == "__main__":
    sample = "First paragraph with enough text to pass the minimum length.\n" * 20
    segs = segment_default(sample, "Test", 50)
    print(f"Default: {len(segs)} segments")
    for sid, txt in segs[:3]:
        print(f"  {sid}: {txt[:80]}")
