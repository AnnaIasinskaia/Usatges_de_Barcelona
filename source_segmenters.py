"""
Source segmentation dispatcher.

Routes to individual segmenter files based on source_name.
Supports both classic Latin sources and gramoty corpora.
"""

import logging
from importlib import import_module

from seg_default import segment_default
from seg_evangelium import segment_evangelium
from seg_corpus_juris import segment_corpus_juris
from seg_lex_visigothorum import segment_lex_visigothorum
from seg_exceptiones_petri import segment_exceptiones_petri
from seg_etymologiae import segment_etymologiae
from seg_common import validate_segments

log = logging.getLogger(__name__)


def _load_gramoty_segmenter():
    """
    Load gramoty segmenter from seg_gramoty_stable.py.

    We do this dynamically because the exact exported function name
    may vary between revisions of the repo.
    """
    try:
        mod = import_module("seg_gramoty_stable")
    except Exception as exc:
        log.warning("Could not import seg_gramoty_stable: %s", exc)
        return None

    candidate_names = [
        "segment_gramoty_stable",
        "segment_gramoty",
        "segment_gramoty911",
        "segment_gramoty12",
        "segment_charters",
        "segment_source",
        "segment",
    ]

    for name in candidate_names:
        fn = getattr(mod, name, None)
        if callable(fn):
            log.info("Loaded gramoty segmenter: seg_gramoty_stable.%s", name)
            return fn

    log.warning(
        "seg_gramoty_stable imported, but no known segment function was found. "
        "Checked: %s",
        ", ".join(candidate_names),
    )
    return None


_GRAMOTY_SEGMENTER = _load_gramoty_segmenter()

_REGISTRY = {
    "Evangelium": segment_evangelium,
    "CorpusJuris": segment_corpus_juris,
    "LexVisigoth": segment_lex_visigothorum,
    "ExceptPetri": segment_exceptiones_petri,
    "Etymologiae": segment_etymologiae,
}

# Register both tomes / aliases if gramoty segmenter is available.
if _GRAMOTY_SEGMENTER is not None:
    for key in [
        "Gramoty911",
        "Gramoty12",
        "Gramoty_I",
        "Gramoty_II",
        "Gramoty1",
        "Gramoty2",
        "GramotyVol1",
        "GramotyVol2",
    ]:
        _REGISTRY[key] = _GRAMOTY_SEGMENTER


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