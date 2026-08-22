from .baselines import run_baseline_ablation
from .corollary import run_corollary_ablation
from .delta import run_delta_ablation
from .models import run_model_ablation
from .timing import run_timing_ablation
from .weights import run_weight_ablation

RUNNERS = {
    "ablation_corollary": run_corollary_ablation,
    "ablation_delta": run_delta_ablation,
    "ablation_models": run_model_ablation,
    "ablation_timing": run_timing_ablation,
    "ablation_weights": run_weight_ablation,
    "ablation_baselines": run_baseline_ablation,
}

__all__ = ["RUNNERS"]
