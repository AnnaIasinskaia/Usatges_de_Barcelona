"""
Source segmentation dispatcher.

Routes to individual segmenter files based on source_name.
Supports both classic Latin sources and gramoty corpora.
"""

import logging

from segmenters.seg_default import segment_default
from segmenters.seg_evangelium import segment_evangelium
from segmenters.seg_corpus_juris import segment_corpus_juris
from segmenters.seg_lex_visigothorum import segment_lex_visigothorum
from segmenters.seg_exceptiones_petri import segment_exceptiones_petri
from segmenters.seg_etymologiae import segment_etymologiae
from segmenters.seg_common import validate_segments
from segmenters.seg_costums_tortosa import segment_costums_tortosa
from segmenters.seg_consuetudines_numbered import segment_consuetudines_numbered
# IMPORTANT:
# Gramoty must always use the new merged segmenter explicitly.
# No dynamic import / fallback probing is allowed here, otherwise an older
# implementation could be picked up by accident.
from segmenters.seg_gramoty_stable_merged import segment_gramoty as segment_gramoty_merged

log = logging.getLogger(__name__)

_REGISTRY = {
    "Evangelium": segment_evangelium,
    "CorpusJuris": segment_corpus_juris,
    "LexVisigoth": segment_lex_visigothorum,
    "ExceptPetri": segment_exceptiones_petri,
    "Etymologiae": segment_etymologiae,
    "Gramoty911": segment_gramoty_merged,
    "Gramoty12": segment_gramoty_merged,
    "Gramoty_I": segment_gramoty_merged,
    "Gramoty_II": segment_gramoty_merged,
    "Gramoty1": segment_gramoty_merged,
    "Gramoty2": segment_gramoty_merged,
    "GramotyVol1": segment_gramoty_merged,
    "GramotyVol2": segment_gramoty_merged,
    "CostumsTortosa":   segment_costums_tortosa,
    "CostMiraveta":     segment_consuetudines_numbered,
    "CostLleida":       segment_consuetudines_numbered,
    "CostTarregi":      segment_consuetudines_numbered,
    "RecogProc":        segment_consuetudines_numbered,
}


def segment_source(text, source_name, cfg=None):
    """
    Segment a source text.

    Dispatches by source_name.

    Args:
        text: raw text string
        source_name: key from config SOURCES / GRAMOTY dict
        cfg: optional dict or int
            - If dict: extracts max_segment_words from it
            - If int: used directly as max_segment_words
            - If None: defaults to 150

    Returns:
        list[tuple[str, str]]

    Raises:
        TypeError on bad segment types (no silent warnings).
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

    segmenter = _REGISTRY.get(source_name)

    if segmenter is not None:
        try:
            segments = segmenter(text, source_name, max_words)
        except TypeError:
            # Extra compatibility: some segmenters may accept only (text, source_name)
            try:
                segments = segmenter(text, source_name)
            except TypeError:
                # And some may accept only (text)
                segments = segmenter(text)
    else:
        segments = segment_default(text, source_name, max_words)

    # Final strict validation
    segments = validate_segments(segments, source_name)
    return segments
