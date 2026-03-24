#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config.py

Конфиг для новой unified-архитектуры.

Ключевые принципы:
- CORPORA: одна точка правды по source_name -> path/display/color
- сегментеры самодостаточные, поэтому конфиг НЕ управляет сегментацией
- стороны графа и направления сопоставления задаются через experiment
- TF-IDF используется только как retrieval-ranking engine
- retrieval управляется одним параметром budget, без threshold/top_k_per_left
- после retrieval всегда считаются 4 метрики:
    * cos_sim
    * tesserae
    * soft_cos
    * sw_norm
- дальше pipeline делает:
    * Pareto filtering
    * rank aggregation
    * top-N selection for graph

На этом этапе config разделён на несколько типов настроек:
- MODEL_DEFAULTS: только параметры препроцессинга / TF-IDF / дорогих метрик
- LOGGING_DEFAULTS: человекочитаемое логгирование
- retrieval / pareto / selection / aggregation / viz / output: стадии эксперимента
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_ROOT = Path("output")


# ---------------------- Корпуса ----------------------

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
    "LexVisigothorum": {
        "path": DATA_DIR / "legesvisigothor00zeumgoog_text.txt",
        "kind": "latin_source",
        "display_ru": "Вестготская\nправда",
        "color": "#9467bd",
    },
    "ExceptPetri": {
        "path": DATA_DIR / "Exeptionis_Legum_Romanorum_Petri_v4.txt",
        "kind": "latin_source",
        "display_ru": "Составленные Петром \nизвлечения из римских законов",
        "color": "#ff7f0e",
    },
    "Etymologiae": {
        "path": DATA_DIR / "Isidori_Hispalensis_Episcopi_Etymologiarum_v2.txt",
        "kind": "latin_source",
        "display_ru": "Этимологии\nИсидора",
        "color": "#2ca02c",
    },
    "CostumsDeTortosa": {
        "path": DATA_DIR / "ObychaiTortosy1272to1279_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nТортосы",
        "color": "#8c564b",
    },
    "CostumsDeLleida": {
        "path": DATA_DIR / "ObychaiLleidy12271228_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nЛьейды",
        "color": "#e377c2",
    },
    "ConstitucionesBaiulieMirabeti": {
        "path": DATA_DIR / "ObychaiMiraveta1319Fix_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nМиравета",
        "color": "#7f7f7f",
    },
    "CostumsDeOrta": {
        "path": DATA_DIR / "ObychaiOrty1296_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nОрты",
        "color": "#bcbd22",
    },
    "RecognovrentProceres": {
        "path": DATA_DIR / "RecognovrentProceres12831284_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Recognovrent\nProceres",
        "color": "#17becf",
    },
    "CostumresDeTarrega": {
        "path": DATA_DIR / "ObychaiTarregi1290E_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nТарреги",
        "color": "#aec7e8",
    },
    "CostumsDeValdAran": {
        "path": DATA_DIR / "ObychaiValdArana1313_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nВаль-д’Арана",
        "color": "#98df8a",
    },
    "PragmaticaJaimeII1295": {
        "path": DATA_DIR / "PragmatikaZhaumeII1295_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Прагматика\nЖауме II (1295)",
        "color": "#ff9896",
    },
    "PragmaticaJaimeII1301": {
        "path": DATA_DIR / "PragmatikaZhaumeII1301_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Прагматика\nЖауме II (1301)",
        "color": "#c5b0d5",
    },
    "Acta911": {
        "path": DATA_DIR / "Gramoty911.txt",
        "kind": "charters",
        "display_ru": "Грамоты\nIX–XI вв.",
        "color": "#2c7fb8",
    },
    "Acta12": {
        "path": DATA_DIR / "Gramoty12.txt",
        "kind": "charters",
        "display_ru": "Грамоты\nXII в.",
        "color": "#f03b20",
    },
    "UsatgesBarcelona": {
        "path": DATA_DIR / "Bastardas_Usatges_de_Barcelona_djvu.txt",
        "kind": "usatges",
        "display_ru": "Барселонские\nОбычаи ",
        "color": "#17becf",
    },
    "CostumsDePerpinya": {
        "path": DATA_DIR / "Customs_of_Perpignan_v2.txt",
        "kind": "catalan_source",
        "display_ru": "Обычаи\nПерпиньяна",
        "color": "#6b1fb1",
    },
}


GROUPS = {
    "LATIN_SOURCES": [
        "Evangelium",
        "CorpusJuris",
        "Etymologiae",
        "LexVisigothorum",
        "ExceptPetri",
    ],
    "CATALAN_SOURCES": [
        "CostumsDeTortosa",
        "CostumsDeLleida",
        "ConstitucionesBaiulieMirabeti",
        "CostumsDeOrta",
        "RecognovrentProceres",
        "CostumresDeTarrega",
        "CostumsDeValdAran",
        "PragmaticaJaimeII1295",
        "PragmaticaJaimeII1301",
        "CostumsDePerpinya"
    ],
    "GRAMOTY": [
        "Acta911",
        "Acta12",
    ],
}


# ---------------------- Общие настройки модели ----------------------

# Здесь лежат только параметры препроцессинга, TF-IDF-представления
# и дорогих similarity-метрик. BorrowScore/threshold-гейты удалены.
MODEL_DEFAULTS = {
    "min_lemma_length": 3,
    "ngram_range": (1, 3),
    "max_df": 0.50,
    "min_df": 2,
    "soft_cosine_max_terms": 500,

    "sw_match": 2,
    "sw_mismatch": -1,
    "sw_gap": -1,
    "sw_lev_bonus_threshold": 2,
    "sw_max_seq_len": 300,
}


# ---------------------- Общие настройки логгирования ----------------------

LOGGING_DEFAULTS = {
    "scoring_progress_every": 1000,
    "lemmatize_progress_every": None,
    "candidate_progress_every": None,
}


# ---------------------- Базовые блоки новой архитектуры ----------------------

# retrieval.budget — не semantic threshold, а вычислительный budget на дорогую стадию.
RETRIEVAL_DEFAULTS = {
    "mode": "pair_quota",
    "budget": 5000,
    "pair_budget_strategy": "weighted_sqrt",
    "min_pair_budget": 100,
    "max_pair_budget": None,
    "per_left_leaf_cap": 10,
    "per_right_leaf_cap": 10,
    "global_budget_after_merge": 5000,
}

# Пока используем только первый Pareto-frontier.
PARETO_DEFAULTS = {
    "keep_layers": 1,
}

# top-N graph edges после rank aggregation и graph aggregation.
SELECTION_DEFAULTS = {
    "graph_top_n": 30,
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
        "retrieval": dict(RETRIEVAL_DEFAULTS, budget=20),
        "pareto": dict(PARETO_DEFAULTS, keep_layers=1),
        "selection": dict(SELECTION_DEFAULTS, graph_top_n=50),
        "model": dict(MODEL_DEFAULTS),
        "logging": dict(LOGGING_DEFAULTS, scoring_progress_every=100),
        "aggregation": {
            "left_node_level": "parent",
            "right_node_level": "parent",
            "weight_mode": "max",
            "min_hits": 1,
        },
        "viz": {
            "enabled": True,
            "straight_edges": True,
            "edge_color_by": "left_corpus",
            "label_left": True,
            "label_right": True,
            "top_n_edges": 50,
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
            "enabled": True,
            "mode": "sliding_window_words",
            "window_words": 180,
            "overlap_words": 60,
            "min_words": 40,
            "per_corpus": {
                "CorpusJuris": {"enabled": False},

                "Evangelium": {
                    "window_words": 140,
                    "overlap_words": 50,
                    "min_words": 40,
                },

                "LexVisigothorum": {
                    "window_words": 170,
                    "overlap_words": 60,
                    "min_words": 40,
                },

                "ExceptPetri": {
                    "window_words": 170,
                    "overlap_words": 50,
                    "min_words": 40,
                },

                "Etymologiae": {
                    "window_words": 170,
                    "overlap_words": 60,
                    "min_words": 40,
                },

                "UsatgesBarcelona": {"enabled": False},
            },
        },
        "retrieval": dict(
            RETRIEVAL_DEFAULTS,
            mode="pair_quota",
            budget=5000,
            pair_budget_strategy="weighted_sqrt",
            min_pair_budget=300,
            per_left_leaf_cap=12,
            per_right_leaf_cap=12,
            global_budget_after_merge=5000,
        ),
        "pareto": dict(PARETO_DEFAULTS, keep_layers=3),
        "selection": dict(SELECTION_DEFAULTS, graph_top_n=50),
        "model": dict(MODEL_DEFAULTS),
        "logging": dict(LOGGING_DEFAULTS, scoring_progress_every=250),
        "aggregation": {
            "left_node_level": "corpus",
            "right_node_level": "parent",
            "weight_mode": "max",
            "min_hits": 1,
        },
        "viz": {
            "enabled": True,
            "straight_edges": True,
            "edge_color_by": "left_corpus",
            "label_left": True,
            "label_right": True,
            "top_n_edges": 50,
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
        "description": "Поиск заимствований: Usatges → other codes",
        "graph_sides": {
            "left": ["UsatgesBarcelona"],
            "right": ["@CATALAN_SOURCES"],
        },
        "mappings": [
            {"from": ["UsatgesBarcelona"], "to": ["@CATALAN_SOURCES"]},
        ],
        "chunking": {
            "enabled": True,
            "mode": "sliding_window_words",
            "window_words": 180,
            "overlap_words": 60,
            "min_words": 40,
            "per_corpus": {
                "UsatgesBarcelona": {"enabled": False},

                "CostumsDeTortosa": {
                    "window_words": 180,
                    "overlap_words": 60,
                    "min_words": 40,
                },

                "CostumsDeValdAran": {
                    "window_words": 170,
                    "overlap_words": 50,
                    "min_words": 40,
                },

                "CostumsDeLleida": {"enabled": False},
                "ConstitucionesBaiulieMirabeti": {"enabled": False},
                "CostumsDeOrta": {"enabled": False},
                "RecognovrentProceres": {"enabled": False},
                "CostumresDeTarrega": {"enabled": False},
                "PragmaticaJaimeII1295": {"enabled": False},
                "PragmaticaJaimeII1301": {"enabled": False},
            },
        },
        "retrieval": dict(
            RETRIEVAL_DEFAULTS,
            mode="pair_quota",
            budget=5000,
            pair_budget_strategy="weighted_sqrt",
            min_pair_budget=300,
            per_left_leaf_cap=12,
            per_right_leaf_cap=12,
            global_budget_after_merge=5000,
        ),
        "pareto": dict(PARETO_DEFAULTS, keep_layers=7),
        "selection": dict(SELECTION_DEFAULTS, graph_top_n=50),
        "model": dict(MODEL_DEFAULTS),
        "logging": dict(LOGGING_DEFAULTS, scoring_progress_every=250),
        "aggregation": {
            "left_node_level": "parent",
            "right_node_level": "parent",
            "weight_mode": "max",
            "min_hits": 1,
        },
        "viz": {
            "enabled": True,
            "straight_edges": True,
            "edge_color_by": "right_corpus",
            "label_left": True,
            "label_right": True,
            "top_n_edges": 50,
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
            "min_words": 30,
            "per_corpus": {
                "CorpusJuris": {"enabled": False},

                "Evangelium": {
                    "window_words": 140,
                    "overlap_words": 50,
                    "min_words": 40,
                },

                "LexVisigothorum": {
                    "window_words": 170,
                    "overlap_words": 60,
                    "min_words": 40,
                },

                "ExceptPetri": {
                    "window_words": 170,
                    "overlap_words": 50,
                    "min_words": 40,
                },

                "Etymologiae": {
                    "window_words": 170,
                    "overlap_words": 60,
                    "min_words": 40,
                },

                "UsatgesBarcelona": {"enabled": False},

                "Acta911": {
                    "window_words": 220,
                    "overlap_words": 80,
                    "min_words": 30,
                },
                "Acta12": {
                    "window_words": 220,
                    "overlap_words": 80,
                    "min_words": 30,
                },
            },
        },
        "retrieval": dict(
            RETRIEVAL_DEFAULTS,
            mode="pair_quota",
            budget=5000,
            pair_budget_strategy="weighted_sqrt",
            min_pair_budget=300,
            per_left_leaf_cap=12,
            per_right_leaf_cap=12,
            global_budget_after_merge=5000,
        ),
        "pareto": dict(PARETO_DEFAULTS, keep_layers=2),
        "selection": dict(SELECTION_DEFAULTS, graph_top_n=50),
        "model": dict(MODEL_DEFAULTS),
        "logging": dict(LOGGING_DEFAULTS, scoring_progress_every=250),
        "aggregation": {
            "left_node_level": "corpus",
            "right_node_level": "parent",
            "weight_mode": "max",
            "min_hits": 2,
        },
        "viz": {
            "enabled": True,
            "straight_edges": True,
            "edge_color_by": "left_corpus",
            "label_left": True,
            "label_right": True,
            "top_n_edges": 50,
        },
        "output": {
            "dir": OUTPUT_ROOT / "left_to_gramoty",
            "write_detail_csv": True,
            "write_graph_csv": True,
            "write_gexf": True,
            "write_png": True,
        },
    },
    "catalan_plus_usatges_upper_triangle": {
        "description": "Все каталонские источники + Usatges, верхний треугольник без self-mapping",
        "graph_sides": {
            "left": ["@CATALAN_SOURCES", "UsatgesBarcelona"],
            "right": ["@CATALAN_SOURCES", "UsatgesBarcelona"],
        },
        "mappings": [
            {
                "from": ["CostumsDeTortosa"],
                "to": [
                    "CostumsDeLleida",
                    "ConstitucionesBaiulieMirabeti",
                    "CostumsDeOrta",
                    "RecognovrentProceres",
                    "CostumresDeTarrega",
                    "CostumsDeValdAran",
                    "PragmaticaJaimeII1295",
                    "PragmaticaJaimeII1301",
                    "UsatgesBarcelona",
                ],
            },
            {
                "from": ["CostumsDeLleida"],
                "to": [
                    "ConstitucionesBaiulieMirabeti",
                    "CostumsDeOrta",
                    "RecognovrentProceres",
                    "CostumresDeTarrega",
                    "CostumsDeValdAran",
                    "PragmaticaJaimeII1295",
                    "PragmaticaJaimeII1301",
                    "UsatgesBarcelona",
                ],
            },
            {
                "from": ["ConstitucionesBaiulieMirabeti"],
                "to": [
                    "CostumsDeOrta",
                    "RecognovrentProceres",
                    "CostumresDeTarrega",
                    "CostumsDeValdAran",
                    "PragmaticaJaimeII1295",
                    "PragmaticaJaimeII1301",
                    "UsatgesBarcelona",
                ],
            },
            {
                "from": ["CostumsDeOrta"],
                "to": [
                    "RecognovrentProceres",
                    "CostumresDeTarrega",
                    "CostumsDeValdAran",
                    "PragmaticaJaimeII1295",
                    "PragmaticaJaimeII1301",
                    "UsatgesBarcelona",
                ],
            },
            {
                "from": ["RecognovrentProceres"],
                "to": [
                    "CostumresDeTarrega",
                    "CostumsDeValdAran",
                    "PragmaticaJaimeII1295",
                    "PragmaticaJaimeII1301",
                    "UsatgesBarcelona",
                ],
            },
            {
                "from": ["CostumresDeTarrega"],
                "to": [
                    "CostumsDeValdAran",
                    "PragmaticaJaimeII1295",
                    "PragmaticaJaimeII1301",
                    "UsatgesBarcelona",
                ],
            },
            {
                "from": ["CostumsDeValdAran"],
                "to": [
                    "PragmaticaJaimeII1295",
                    "PragmaticaJaimeII1301",
                    "UsatgesBarcelona",
                ],
            },
            {
                "from": ["PragmaticaJaimeII1295"],
                "to": [
                    "PragmaticaJaimeII1301",
                    "UsatgesBarcelona",
                ],
            },
            {
                "from": ["PragmaticaJaimeII1301"],
                "to": [
                    "UsatgesBarcelona",
                ],
            },
        ],
        "chunking": {
            "enabled": True,
            "mode": "sliding_window_words",
            "window_words": 180,
            "overlap_words": 60,
            "min_words": 40,
            "per_corpus": {
                "UsatgesBarcelona": {"enabled": False},

                "CostumsDeTortosa": {
                    "window_words": 180,
                    "overlap_words": 60,
                    "min_words": 40,
                },

                "CostumsDeValdAran": {
                    "window_words": 170,
                    "overlap_words": 50,
                    "min_words": 40,
                },

                "CostumsDeLleida": {"enabled": False},
                "ConstitucionesBaiulieMirabeti": {"enabled": False},
                "CostumsDeOrta": {"enabled": False},
                "RecognovrentProceres": {"enabled": False},
                "CostumresDeTarrega": {"enabled": False},
                "PragmaticaJaimeII1295": {"enabled": False},
                "PragmaticaJaimeII1301": {"enabled": False},
            },
        },
        "retrieval": dict(
            RETRIEVAL_DEFAULTS,
            mode="pair_quota",
            budget=16000,
            pair_budget_strategy="weighted_sqrt",
            min_pair_budget=300,
            per_left_leaf_cap=12,
            per_right_leaf_cap=12,
            global_budget_after_merge=16000,
        ),
        "pareto": dict(PARETO_DEFAULTS, keep_layers=11),
        "selection": dict(SELECTION_DEFAULTS, graph_top_n=100),
        "model": dict(MODEL_DEFAULTS),
        "logging": dict(LOGGING_DEFAULTS, scoring_progress_every=250),
        "aggregation": {
            "left_node_level": "corpus",
            "right_node_level": "corpus",
            "weight_mode": "max",
            "min_hits": 1,
        },
        "viz": {
            "enabled": True,
            "straight_edges": True,
            "edge_color_by": "left_corpus",
            "label_left": True,
            "label_right": True,
            "top_n_edges": 100,
        },
        "output": {
            "dir": OUTPUT_ROOT / "catalan_plus_usatges_upper_triangle",
            "write_detail_csv": True,
            "write_graph_csv": True,
            "write_gexf": True,
            "write_png": True,
        },
    },
}
