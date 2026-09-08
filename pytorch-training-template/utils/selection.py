def initialize_best_metric(mode):
    if mode == "min":
        return float("inf")
    if mode == "max":
        return float("-inf")
    raise ValueError(
        f"Unsupported selection mode: {mode}"
    )

def is_better(
    current,
    best,
    mode,
):
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError(
        f"Unsupported selection mode: {mode}"
    )