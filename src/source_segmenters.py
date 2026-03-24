"""
Source segmentation dispatcher.

Маршрутизация только на unified-сегментеры.
Строгий контракт сегментеров:
    segment_<source>_unified(source_file, source_name) -> list[tuple[str, str]]
где каждый элемент — пара (segment_id, segment_text).

Никаких legacy-алиасов, cfg-параметров и fallback-логики здесь нет.
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
from segmenters.seg_perpignan import segment_perpignan_unified


SegmenterFunc = Callable[[str | Path, str], list[tuple[str, str]]]


_SEGMENTERS: dict[str, SegmenterFunc] = {
    "CorpusJuris": segment_corpus_juris_unified,
    "Evangelium": segment_evangelium_unified,
    "LexVisigothorum": segment_lex_visigothorum_unified,
    "ExceptPetri": segment_exceptiones_petri_unified,
    "Etymologiae": segment_etymologiae_unified,
    "CostumsDeTortosa": segment_costums_tortosa_unified,
    "CostumsDeLleida": segment_lleida_unified,
    "ConstitucionesBaiulieMirabeti": segment_miravet_unified,
    "CostumsDeOrta": segment_orty_unified,
    "RecognovrentProceres": segment_privileges_unified,
    "CostumresDeTarrega": segment_tarregi_unified,
    "CostumsDeValdAran": segment_vald_aran_unified,
    "PragmaticaJaimeII1295": segment_zhaime1295_unified,
    "PragmaticaJaimeII1301": segment_zhaime1301_unified,
    "Acta911": segment_gramoty_911_unified,
    "Acta12": segment_gramoty_12_unified,
    "UsatgesBarcelona": segment_usatges_unified,
    "CostumsDePerpinya": segment_perpignan_unified,
}


def segment_source(source_file: str | Path, source_name: str) -> list[tuple[str, str]]:
    """
    Запускает unified-сегментер по source_name.

    Параметры
    ---------
    source_file : str | Path
        Путь к файлу источника.
    source_name : str
        Каноническое имя источника.

    Возвращает
    ----------
    list[tuple[str, str]]
        Список сегментов в строгом формате:
        [(segment_id, segment_text), ...]

    Исключения
    ----------
    KeyError
        Если source_name не зарегистрирован в таблице сегментеров.
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
    """Возвращает копию таблицы маршрутизации сегментеров."""
    return dict(_SEGMENTERS)