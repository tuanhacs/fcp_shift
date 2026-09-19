FIXED_ALPHA_LABEL = r"$\widehat{\mathrm{FCP}}_{\mathrm{fix}}^{w,\delta,\alpha}$"
UNIFORM_ALPHA_LABEL = r"$\widehat{\mathrm{FCP}}_{\mathrm{unif}}^{w,\delta,\alpha}$"
FIXED_BETA_LABEL = r"$\widehat{\mathrm{FCP}}_{\mathrm{fix}}^{w,\delta,\beta}$"
UNIFORM_BETA_LABEL = r"$\widehat{\mathrm{FCP}}_{\mathrm{unif}}^{w,\delta,\beta}$"

# Empirical pass-rate labels used when validating the four probabilistic
# guarantees. These are deliberately distinct from the bound/selection labels
# above, which are still used by timing and diagnostic figures.
FIXED_ALPHA_PASS_LABEL = r"$\widehat{\mathfrak{P}}_{\mathrm{fix}}(\alpha)$"
UNIFORM_ALPHA_PASS_LABEL = r"$\widehat{\mathfrak{P}}_{\mathrm{unif}}$"
FIXED_BETA_PASS_LABEL = r"$\widehat{\mathfrak{P}}_{\mathrm{fix}}^{-1}(\beta)$"
UNIFORM_BETA_PASS_LABEL = r"$\widehat{\mathfrak{P}}_{\mathrm{unif}}^{-1}$"

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

GOAL_PASS_LABELS = {
    1: FIXED_ALPHA_PASS_LABEL,
    2: UNIFORM_ALPHA_PASS_LABEL,
    3: FIXED_BETA_PASS_LABEL,
    4: UNIFORM_BETA_PASS_LABEL,
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
