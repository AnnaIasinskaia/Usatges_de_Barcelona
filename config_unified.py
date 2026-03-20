#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config_unified.py

Конфиг для unified pipeline.
- CORPORA: одна точка правды по source_name -> path/display/color
- сегментеры самодостаточные, поэтому конфиг НЕ управляет сегментацией
- стороны графа и направления сопоставления задаются через experiment
- smoke test оформлен как обычный experiment с id="test"

На текущем этапе конфиг разделён на несколько типов настроек:
- MODEL_DEFAULTS: параметры вычисления сходства и BorrowScore
- LOGGING_DEFAULTS: параметры человекочитаемого логгирования пайплайна
- viz/output/chunking/aggregation: настройки отдельных стадий эксперимента
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_ROOT = Path("output_unified")


# ---------------------- Корпуса ----------------------

# Ключи должны совпадать с source_name из:
# - source_segmenters.py
# - test_unified_segmenters.py

CORPORA = {
    "CorpusJuris": {
        "path": DATA_DIR / "Corpus_Juris_Civilis_v2.txt",
        "kind": "latin_source",
        "display_ru": "Свод\nЮстиниана",
        "color": "#d62728",
    },
    "Evangelium": {
        "path": DATA_DIR / "Evangelium_v2.txt",
        "kind": "latin_source",
        "display_ru": "Евангелие\n(Вульгата)",
        "color": "#1f77b4",
    },
    "LexVisigoth": {
        "path": DATA_DIR / "legesvisigothor00zeumgoog_text.txt",
        "kind": "latin_source",
        "display_ru": "Вестготская\nправда",
        "color": "#9467bd",
    },
    "ExceptPetri": {
        "path": DATA_DIR / "Exeptionis_Legum_Romanorum_Petri_v3.txt",
        "kind": "latin_source",
        "display_ru": "Извлечения\nПетра",
        "color": "#ff7f0e",
    },
    "Etymologiae": {
        "path": DATA_DIR / "Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt",
        "kind": "latin_source",
        "display_ru": "Этимологии\nИсидора",
        "color": "#2ca02c",
    },
    "ObychaiTortosy1272to1279": {
        "path": DATA_DIR / "ObychaiTortosy1272to1279_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nТортосы",
        "color": "#8c564b",
    },
    "ObychaiLleidy12271228": {
        "path": DATA_DIR / "ObychaiLleidy12271228_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nЛьейды",
        "color": "#e377c2",
    },
    "ObychaiMiraveta1319Fix": {
        "path": DATA_DIR / "ObychaiMiraveta1319Fix_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nМиравета",
        "color": "#7f7f7f",
    },
    "ObychaiOrty1296": {
        "path": DATA_DIR / "ObychaiOrty1296_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nОрты",
        "color": "#bcbd22",
    },
    "RecognovrentProceres12831284": {
        "path": DATA_DIR / "RecognovrentProceres12831284_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Recognovrent\nProceres",
        "color": "#17becf",
    },
    "ObychaiTarregi1290E": {
        "path": DATA_DIR / "ObychaiTarregi1290E_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nТарреги",
        "color": "#aec7e8",
    },
    "ObychaiValdArana1313": {
        "path": DATA_DIR / "ObychaiValdArana1313_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nВаль-д’Арана",
        "color": "#98df8a",
    },
    "PragmatikaZhaumeII1295": {
        "path": DATA_DIR / "PragmatikaZhaumeII1295_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Прагматика\nЖауме II (1295)",
        "color": "#ff9896",
    },
    "PragmatikaZhaumeII1301": {
        "path": DATA_DIR / "PragmatikaZhaumeII1301_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Прагматика\nЖауме II (1301)",
        "color": "#c5b0d5",
    },
    "Gramoty911": {
        "path": DATA_DIR / "Gramoty911.txt",
        "kind": "charters",
        "display_ru": "Грамоты\nIX–XI вв.",
        "color": "#2c7fb8",
    },
    "Gramoty12": {
        "path": DATA_DIR / "Gramoty12.txt",
        "kind": "charters",
        "display_ru": "Грамоты\nXII в.",
        "color": "#f03b20",
    },
    "UsatgesBarcelona": {
        "path": DATA_DIR / "Bastardas_Usatges_de_Barcelona_djvu.txt",
        "kind": "usatges",
        "display_ru": "Обычаи\nБарселоны",
        "color": "#17becf",
    },
}


GROUPS = {
    "LATIN_SOURCES": [
        "Evangelium",
        "CorpusJuris",
        "Etymologiae",
        "LexVisigoth",
        "ExceptPetri",
    ],
    "CATALAN_SOURCES": [
        "ObychaiTortosy1272to1279",
        "ObychaiLleidy12271228",
        "ObychaiMiraveta1319Fix",
        "ObychaiOrty1296",
        "RecognovrentProceres12831284",
        "ObychaiTarregi1290E",
        "ObychaiValdArana1313",
        "PragmatikaZhaumeII1295",
        "PragmatikaZhaumeII1301",
    ],
    "GRAMOTY": [
        "Gramoty911",
        "Gramoty12",
    ],
}


# ---------------------- Общие настройки модели ----------------------

# Здесь лежат именно алгоритмические параметры.
# Они должны влиять на candidate selection / scoring / alignment,
# а не на формат вывода или логирование.
MODEL_DEFAULTS = {
    "use_collatinus": False,
    "min_lemma_length": 3,
    "ngram_range": (1, 3),
    "max_df": 0.50,
    "min_df": 2,
    "tfidf_cosine_threshold": 0.08,
    "alpha": 0.30,
    "beta": 0.40,
    "gamma": 0.30,
    "final_threshold": 0.10,
    "soft_cosine_max_terms": 500,

    # Эвристика включения soft cosine:
    # если cos_sim + beta * tess > final_threshold * soft_cosine_gate_factor,
    # тогда считается soft cosine, иначе он пропускается как дорогой этап.
    "soft_cosine_gate_factor": 0.50,

    "sw_match": 2,
    "sw_mismatch": -1,
    "sw_gap": -1,
    "sw_lev_bonus_threshold": 2,
    "sw_max_seq_len": 300,
    "sw_min_score": 0.0,
}


# ---------------------- Общие настройки логгирования ----------------------

# Эти параметры не меняют результат вычислений.
# Они управляют только тем, насколько детально pipeline пишет ход исполнения.
LOGGING_DEFAULTS = {
    # Частота progress-логов для scoring loop.
    # Для коротких прогонов можно ставить меньше, для длинных — больше.
    "scoring_progress_every": 1000,

    # Заготовка под следующие итерации:
    # unified pipeline может начать читать это значение вместо хардкода,
    # если понадобится более детальный прогресс на этапе лемматизации.
    "lemmatize_progress_every": None,

    # Заготовка под возможный progress на этапе генерации кандидатов.
    "candidate_progress_every": None,
}


# ---------------------- Эксперименты ----------------------

EXPERIMENTS = {
    "test": {
        "description": "Smoke test: Evangelium → UsatgesBarcelona",
        "graph_sides": {
            "left": ["Evangelium"],
            "right": ["UsatgesBarcelona"],
        },
        "mappings": [
            {"from": ["Evangelium"], "to": ["UsatgesBarcelona"]},
        ],
        "chunking": {
            "enabled": False,
        },
        "candidate_selection": {
            "threshold": 0.08,
            "top_k_per_left": 3,
        },
        "model": dict(MODEL_DEFAULTS),
        "logging": dict(LOGGING_DEFAULTS, scoring_progress_every=100),
        "alignment": {
            "enabled": False,
        },
        "aggregation": {
            "left_node_level": "parent",
            "right_node_level": "parent",
            "weight_mode": "max",
            "min_hits": 1,
            "keep_best_evidence": True,
        },
        "viz": {
            "enabled": True,
            "straight_edges": True,
            "edge_color_by": "left_corpus",
            "label_left": False,
            "label_right": True,
            "top_n_edges": 100,
        },
        "output": {
            "dir": OUTPUT_ROOT / "test",
            "write_detail_csv": True,
            "write_graph_csv": True,
            "write_gexf": True,
            "write_png": True,
        },
    },

    "latin_to_usatges": {
        "description": "Поиск заимствований: латинские источники → Usatges",
        "graph_sides": {
            "left": ["@LATIN_SOURCES"],
            "right": ["UsatgesBarcelona"],
        },
        "mappings": [
            {"from": ["@LATIN_SOURCES"], "to": ["UsatgesBarcelona"]},
        ],
        "chunking": {
            "enabled": False,
        },
        "candidate_selection": {
            "threshold": 0.08,
            "top_k_per_left": None,
        },
        "model": dict(MODEL_DEFAULTS),
        "logging": dict(LOGGING_DEFAULTS, scoring_progress_every=250),
        "alignment": {
            "enabled": True,
        },
        "aggregation": {
            "left_node_level": "corpus",
            "right_node_level": "parent",
            "weight_mode": "max",
            "min_hits": 2,
            "keep_best_evidence": True,
        },
        "viz": {
            "enabled": True,
            "straight_edges": True,
            "edge_color_by": "left_corpus",
            "label_left": True,
            "label_right": True,
            "top_n_edges": 30,
        },
        "output": {
            "dir": OUTPUT_ROOT / "latin_to_usatges",
            "write_detail_csv": True,
            "write_graph_csv": True,
            "write_gexf": True,
            "write_png": True,
        },
    },
    "usatges_to_other_codes": {
        "description": "Поиск заимствований: Usatges → Other_codes",
        "graph_sides": {
            "left": ["UsatgesBarcelona"],
            "right": ["@CATALAN_SOURCES"],
        },
        "mappings": [
            {"from": ["UsatgesBarcelona"], "to": ["@CATALAN_SOURCES"]},
        ],
        "chunking": {
            "enabled": False,
        },
        "candidate_selection": {
            "threshold": 0.08,
            "top_k_per_left": None,
        },
        "model": dict(MODEL_DEFAULTS),
        "logging": dict(LOGGING_DEFAULTS, scoring_progress_every=250),
        "alignment": {
            "enabled": True,
        },
        "aggregation": {
            "left_node_level": "parent",
            "right_node_level": "parent",
            "weight_mode": "max",
            "min_hits": 2,
            "keep_best_evidence": True,
        },
        "viz": {
            "enabled": True,
            "straight_edges": True,
            "edge_color_by": "left_corpus",
            "label_left": True,
            "label_right": True,
            "top_n_edges": 30,
        },
        "output": {
            "dir": OUTPUT_ROOT / "usatges_to_other_codes",
            "write_detail_csv": True,
            "write_graph_csv": True,
            "write_gexf": True,
            "write_png": True,
        },
    },

    "left_to_gramoty": {
        "description": "Поиск заимствований: (латинские источники + Usatges) → грамоты",
        "graph_sides": {
            "left": ["@LATIN_SOURCES", "UsatgesBarcelona"],
            "right": ["@GRAMOTY"],
        },
        "mappings": [
            {"from": ["@LATIN_SOURCES", "UsatgesBarcelona"], "to": ["@GRAMOTY"]},
        ],
        "chunking": {
            "enabled": True,
            "mode": "sliding_window_words",
            "window_words": 180,
            "overlap_words": 60,
            "min_words": 20,
            "per_corpus": {
                "Gramoty911": {"window_words": 220, "overlap_words": 80, "min_words": 30},
                "Gramoty12": {"window_words": 220, "overlap_words": 80, "min_words": 30},
                "UsatgesBarcelona": {"enabled": False},
            },
        },
        "candidate_selection": {
            "threshold": 0.08,
            "top_k_per_left": 5,
        },
        "model": dict(
            MODEL_DEFAULTS,
            final_threshold=0.12,
        ),
        "logging": dict(LOGGING_DEFAULTS, scoring_progress_every=250),
        "alignment": {
            "enabled": True,
        },
        "aggregation": {
            "left_node_level": "corpus",
            "right_node_level": "parent",
            "weight_mode": "max",
            "min_hits": 2,
            "keep_best_evidence": True,
        },
        "viz": {
            "enabled": True,
            "straight_edges": True,
            "edge_color_by": "left_corpus",
            "label_left": True,
            "label_right": True,
            "top_n_edges": 300,
        },
        "output": {
            "dir": OUTPUT_ROOT / "left_to_gramoty",
            "write_detail_csv": True,
            "write_graph_csv": True,
            "write_gexf": True,
            "write_png": True,
        },
    },
}
