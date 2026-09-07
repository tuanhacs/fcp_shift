FIXED_ALPHA_LABEL = r"$\widehat{\mathrm{FCP}}_{\mathrm{fix}}^{w,\delta,\alpha}$"
UNIFORM_ALPHA_LABEL = r"$\widehat{\mathrm{FCP}}_{\mathrm{unif}}^{w,\delta,\alpha}$"
FIXED_BETA_LABEL = r"$\widehat{\mathrm{FCP}}_{\mathrm{fix}}^{w,\delta,\beta}$"
UNIFORM_BETA_LABEL = r"$\widehat{\mathrm{FCP}}_{\mathrm{unif}}^{w,\delta,\beta}$"

EMPIRICAL_COLOR = "#111111"
TARGET_COLOR = "#111111"
FIXED_ALPHA_COLOR = "#0072B2"
UNIFORM_ALPHA_COLOR = "#D55E00"
FIXED_BETA_COLOR = "#009E73"
UNIFORM_BETA_COLOR = "#CC79A7"

GOAL_LABELS = {
    1: FIXED_ALPHA_LABEL,
    2: UNIFORM_ALPHA_LABEL,
    3: FIXED_BETA_LABEL,
    4: UNIFORM_BETA_LABEL,
}

DATASET_DISPLAY_NAMES = {
    "adult": "Adult",
    "allstate": "Allstate Claims Severity",
    "diamonds": "Diamonds",
    "electricity": "Electricity",
    "fashion_mnist": "Fashion-MNIST",
    "mnist": "MNIST",
    "year": "YearPredictionMSD",
}


def display_dataset_name(name: str) -> str:
    return DATASET_DISPLAY_NAMES.get(name, name.replace("_", " ").title())
