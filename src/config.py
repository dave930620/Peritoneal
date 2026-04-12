"""
Central configuration for the PD prescription recommendation system.

Modify constants here to change paths, features, or hyperparameters
without touching model or training code.
"""

import os

# =============================================================================
# File paths (can be overridden via environment variables)
# =============================================================================

DATA_CSV         = os.environ.get("DATA_CSV",      "final_nighttime_data.csv")
FEATURE_INFO_PATH = os.environ.get("FEATURE_INFO", "feature_info.pkl")
F1_MODEL_PATH    = os.environ.get("F1_MODEL",      "saint_pd_model.pth")
REPORT_DIR       = os.environ.get("REPORT_DIR",    "report_model2")
CATBOOST_DIR     = os.environ.get("CATBOOST_DIR",  "catboost_models")

# =============================================================================
# Column identifiers
# =============================================================================

PATIENT_ID_COL     = "PatientID"
TIME_COL           = "記錄時間"
OUTCOME_COL        = "PD Kt/V"
CLINICAL_THRESHOLD = 1.7   # PD Kt/V adequacy threshold (used in evaluation + loss)

# =============================================================================
# Feature exclusion list
#
# These columns are NEVER included in model inputs.
# Reasons are annotated inline.
# =============================================================================

EXCLUDE_COLUMNS = [
    "記錄時間",                      # Timestamp – identifier, not a clinical feature
    "PatientID",                     # Patient identifier
    "RRF Kt/V",                      # Alternative outcome variant (leakage risk)
    "total Kt/V",                    # Alternative outcome variant (leakage risk)
    "PCl rate (↑/wk）",             # Derived from outcome (leakage)
    "RRF (↑/wk）",                  # Derived from outcome (leakage)
    "total Cl rate (↑/wk）",        # Derived from outcome (leakage)
    "Std total Cl rate(↑/wk)民",    # Derived from outcome (leakage)
    "Fluid Exchange System",         # Collinear with long term PD system; not independently predictive
    "Volume (L)",                    # REMOVED: unreliable prescription artifact (夜間灌注量)
                                     # Contains many zeros and inconsistent recording
]

# =============================================================================
# Discrete (categorical) feature columns
#
# All other columns in the dataset are treated as continuous.
# =============================================================================

DISCRETE_COLUMNS = [
    "SEX", "DM type", "long term PD system", "DOResult", "DPResult",
    "HBsAg", "Anti-HCV", "infection", "BLOOD GROUP",
    "Primary disease categories", "primary disease subclass",
    "EPO", "active Vit D", "antihypertensive",
    "Iron therapy", "PTx",
    "other systemic disease-1",  "other systemic disease-2",  "other systemic disease-3",
    "other systemic disease-4",  "other systemic disease-5",  "other systemic disease-6",
    "other systemic disease-7",  "other systemic disease-8",  "other systemic disease-9",
    "other systemic disease-10", "other systemic disease-12",
    "complication-0",  "complication-1",  "complication-2",  "complication-3",
    "complication-4",  "complication-5",  "complication-6",  "complication-7",
    "complication-8",  "complication-9",  "complication-10", "complication-11",
    "complication-12", "complication-14", "complication-15", "complication-17",
    "complication-18", "complication-19", "complication-20", "complication-21",
    "complication-22", "complication-23", "complication-24", "complication-25",
    "complication-26", "complication-28", "complication-29", "complication-30",
    "complication-31", "complication-32", "complication-33",
]

# =============================================================================
# Prescription variables controlled by F2
#
# Volume (L) is intentionally excluded — it is unreliable and removed from
# the data pipeline. total vol/day is a separate, derived variable and IS included.
# =============================================================================

# Single categorical prescription variable
CAT_RX = "long term PD system"

# Continuous prescription variables
CONT_RX = [
    "No. of bag/day",
    "total vol/day",
    "glucose_total",
    "calcium_total",
    "night time PD",
    "Fluid change times",
    "glucose_total_n",
    "calcium_total_n",
]

# Clinical step sizes for rounding outputs to realistic prescription values
STEP_MAP = {
    "No. of bag/day":     1.0,
    "total vol/day":      0.5,
    "glucose_total":      0.25,
    "calcium_total":      0.25,
    "night time PD":      1.0,
    "Fluid change times": 1.0,
    "glucose_total_n":    0.25,
    "calcium_total_n":    0.25,
}

# =============================================================================
# F1 (SAINT) model hyperparameters
# =============================================================================

F1_HIDDEN_SIZE  = 64
F1_NUM_HEADS    = 8
F1_NUM_LAYERS   = 6
F1_DROPOUT      = 0.1
F1_OUTPUT_SIZE  = 1
F1_EPOCHS       = 200
F1_BATCH_SIZE   = 32
F1_LR           = 0.001
F1_WEIGHT_DECAY = 0.01

# =============================================================================
# F2 training hyperparameters
# =============================================================================

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.75, 0.10, 0.15

F2_BATCH_SIZE = 256
F2_HIDDEN     = 128
F2_DROPOUT    = 0.25
GRAD_CLIP     = 1.0

# Stage A — CatBoost doctor mimic
CATBOOST_ITERATIONS  = 500
CATBOOST_LR          = 0.05
CATBOOST_DEPTH       = 6
CATBOOST_EARLY_STOP  = 50

# Stage B — MLP effect optimization
LR_B     = 3e-4
EPOCHS_B = 25

# =============================================================================
# Stage B: trust-region & loss weights
# =============================================================================

# Trust-region radii in z-space (separate for PASS and FAIL cases)
EPS_CONT_Z_PASS   = 0.75
EPS_CONT_Z_FAIL   = 3.00
EPS_CAT_SOFT_PASS = 0.15
EPS_CAT_SOFT_FAIL = 0.35

LAMBDA_CONSTRAINT = 1.0
LAMBDA_PROX_CONT  = 0.05
LAMBDA_PROX_CAT   = 0.01
DELTA_WIN         = 0       # Win margin for threshold reporting

THR_GATE       = 1.7        # Same as CLINICAL_THRESHOLD; used in PASS/FAIL gating
W_EFFECT_PASS  = 0.05       # Mild effect push when doctor already passes
W_EFFECT_FAIL  = 5.00       # Strong effect push when doctor fails
W_PROX_FAIL    = 0.00       # Relax proximal penalty for FAIL cases

LAMBDA_THR_FAIL = 1.0       # Extra penalty for staying below threshold in FAIL cases
LAMBDA_THR_PASS = 0.0

# Curriculum scheduling: tighten constraints early, relax later
CURR_STRONG_FRAC = 0.3
CURR_MID_FRAC    = 0.7
