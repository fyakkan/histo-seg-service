# Test fixtures

Eight 256×256 tiles (0.5 µm/px) with their masks, copied from test-split patients of the
BCSS dataset (Amgad et al., Bioinformatics 2019; CC0 1.0) by `make fixtures`.
Five contain blood vessel pixels and three do not; `tiles.csv` lists their pixel counts.
Masks hold the original BCSS class codes (0 = outside_roi, 1 = tumor, 18 = blood_vessel).
