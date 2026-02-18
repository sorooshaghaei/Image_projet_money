from typing import Dict, Tuple

# Official euro-coin diameters (mm), used with a per-image global px/mm scale fit.
COIN_DIAMETER_MM: Dict[int, float] = {
    1: 16.25,
    2: 18.75,
    5: 21.25,
    10: 19.75,
    20: 22.25,
    50: 24.25,
    100: 23.25,  # 1 EUR
    200: 25.75,  # 2 EUR
}

COLOR_BRONZE = "bronze"
COLOR_GOLD = "gold"
COLOR_BIMETAL_1E = "bimetal_gold_ring"
COLOR_BIMETAL_2E = "bimetal_silver_ring"
COLOR_UNKNOWN = "unknown"

COLOR_TO_DENOMS: Dict[str, Tuple[int, ...]] = {
    COLOR_BRONZE: (1, 2, 5),
    COLOR_GOLD: (10, 20, 50),
    COLOR_BIMETAL_1E: (100,),
    COLOR_BIMETAL_2E: (200,),
    COLOR_UNKNOWN: (1, 2, 5, 10, 20, 50, 100, 200),
}
