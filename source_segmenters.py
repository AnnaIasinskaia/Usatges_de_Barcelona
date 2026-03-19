"""
Source segmentation dispatcher.

Маршрутизация только на новые *_unified сегментеры, которые работают
с путями к файлам, а не с уже загруженными текстами.

Опора — на список сегментеров и source_name из test_unified_segmenters.py.
Никаких алиасов вроде Gramoty_I / GramotyVol1 и т.п. здесь больше нет.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


from segmenters.seg_corpus_juris import segment_corpus_juris_unified
from segmenters.seg_evangelium import segment_evangelium_unified
from segmenters.seg_lex_visigothorum import segment_lex_visigothorum_unified
from segmenters.seg_exceptiones_petri import segment_exceptiones_petri_unified
from segmenters.seg_etymologiae import segment_etymologiae_unified
from segmenters.seg_costums_tortosa import segment_costums_tortosa_unified
from segmenters.seg_lleida import segment_lleida_unified
from segmenters.seg_miravet import segment_miravet_unified
from segmenters.seg_orty import segment_orty_unified
from segmenters.seg_privileges import segment_privileges_unified
from segmenters.seg_tarregi import segment_tarregi_unified
from segmenters.seg_vald_aran import segment_vald_aran_unified
from segmenters.seg_zhaime1295 import segment_zhaime1295_unified
from segmenters.seg_zhaime1301 import segment_zhaime1301_unified
from segmenters.seg_gramoty_911 import segment_gramoty_911_unified
from segmenters.seg_gramoty_12 import segment_gramoty_12_unified
from segmenters.seg_usatges import segment_usatges_unified


SegmenterFunc = Callable[[str | Path, str], list[tuple[str, str]]]


_SEGMENTERS: dict[str, SegmenterFunc] = {
    "CorpusJuris": segment_corpus_juris_unified,
    "Evangelium": segment_evangelium_unified,
    "LexVisigoth": segment_lex_visigothorum_unified,
    "ExceptPetri": segment_exceptiones_petri_unified,
    "Etymologiae": segment_etymologiae_unified,
    "ObychaiTortosy1272to1279": segment_costums_tortosa_unified,
    "ObychaiLleidy12271228": segment_lleida_unified,
    "ObychaiMiraveta1319Fix": segment_miravet_unified,
    "ObychaiOrty1296": segment_orty_unified,
    "RecognovrentProceres12831284": segment_privileges_unified,
    "ObychaiTarregi1290E": segment_tarregi_unified,
    "ObychaiValdArana1313": segment_vald_aran_unified,
    "PragmatikaZhaumeII1295": segment_zhaime1295_unified,
    "PragmatikaZhaumeII1301": segment_zhaime1301_unified,
    "Gramoty911": segment_gramoty_911_unified,
    "Gramoty12": segment_gramoty_12_unified,
    "UsatgesBarcelona": segment_usatges_unified,
}


def segment_source(source_file: str | Path, source_name: str, cfg=None) -> list[tuple[str, str]]:
    """
    Запускает unified-сегментер по source_name.

    Параметры
    ---------
    source_file : str | Path
        Путь к файлу источника.
    source_name : str
        Имя источника из test_unified_segmenters.py.
    cfg : Any
        Оставлено только для совместимости со старым кодом.
        На сегментацию больше не влияет и игнорируется.

    Возвращает
    ----------
    list[tuple[str, str]]
        Список сегментов в едином формате.
    """
    try:
        segmenter = _SEGMENTERS[source_name]
    except KeyError as exc:
        available = ", ".join(sorted(_SEGMENTERS))
        raise KeyError(
            f"Unknown source_name: {source_name!r}. "
            f"Available values: {available}"
        ) from exc

    return segmenter(source_file, source_name)


def get_available_segmenters() -> dict[str, SegmenterFunc]:
    """
    Удобно для отладки и тестов: возвращает копию таблицы маршрутизации.
    """
    return dict(_SEGMENTERS)