"""
Source segmentation dispatcher.
Routes to individual segmenter files based on source_name.
"""
import logging
from seg_default import segment_default
from seg_evangelium import segment_evangelium
from seg_corpus_juris import segment_corpus_juris
from seg_lex_visigothorum import segment_lex_visigothorum
from seg_exceptiones_petri import segment_exceptiones_petri
from seg_etymologiae import segment_etymologiae
from seg_common import validate_segments

log = logging.getLogger(__name__)

_REGISTRY = {
    "Evangelium":   segment_evangelium,
    "CorpusJuris":  segment_corpus_juris,
    "LexVisigoth":  segment_lex_visigothorum,
    "ExceptPetri":  segment_exceptiones_petri,
    "Etymologiae":  segment_etymologiae,
}


def segment_source(text, source_name, cfg=None):
    """
    Segment a source text. Dispatches by source_name.

    Args:
        text: raw text string
        source_name: key from config SOURCES dict
        cfg: optional dict or int.
             - If dict: extracts max_segment_words from it
             - If int: used directly as max_segment_words
             - If None: defaults to 150

    Raises TypeError on bad segment types (no silent warnings).
    """
    # Normalize cfg -> max_segment_words (int)
    if cfg is None:
        max_words = 150
    elif isinstance(cfg, dict):
        max_words = cfg.get("max_segment_words", 150)
    elif isinstance(cfg, (int, float)):
        max_words = int(cfg)
    else:
        max_words = 150

    segmenter = _REGISTRY.get(source_name, None)
    if segmenter is not None:
        segments = segmenter(text, source_name, max_words)
    else:
        segments = segment_default(text, source_name, max_words)

    # Final strict validation
    segments = validate_segments(segments, source_name)
    return segments
