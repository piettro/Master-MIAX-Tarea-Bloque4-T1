"""Central configuration: paths, column names and hyper-parameters.

Every magic number used by the pipeline lives here. Modules receive the
relevant dataclass instead of hard-coding values, which keeps experiments
reproducible and makes the ``--quick`` smoke-test mode a one-liner.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "application_train.csv"
RAW_ZIP_PATH = DATA_DIR / "application_train.zip"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
# NOTE: improved over original submission — Keras Tuner writes thousands of
# small files; keeping them outside OneDrive-synced folders avoids file locks.
TUNER_DIR = Path(tempfile.gettempdir()) / "b4t1_keras_tuner"
DATASET_URL = (
    "https://www.kaggle.com/competitions/home-credit-default-risk/data"
)

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
TARGET_COLUMN = "TARGET"
SENSITIVE_COLUMN = "CODE_GENDER"
GENDER_ENCODING = {"M": 0, "F": 1}
INCOME_COLUMN = "AMT_INCOME_TOTAL"
CREDIT_COLUMN = "AMT_CREDIT"
ANNUITY_COLUMN = "AMT_ANNUITY"
AMOUNT_COLUMNS = (INCOME_COLUMN, CREDIT_COLUMN, ANNUITY_COLUMN)
BIRTH_COLUMN = "DAYS_BIRTH"
AGE_COLUMN = "AGE_YEARS"
DAYS_PER_YEAR = 365.0
EXT_SOURCE_COLUMNS = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
MISSING_FLAG_SUFFIX = "_MISSING"
EXT_MISSING_COLUMNS = tuple(
    f"{col}{MISSING_FLAG_SUFFIX}" for col in EXT_SOURCE_COLUMNS
)
RAW_COLUMNS = (
    TARGET_COLUMN,
    SENSITIVE_COLUMN,
    *AMOUNT_COLUMNS,
    BIRTH_COLUMN,
    *EXT_SOURCE_COLUMNS,
)
# Continuous features are imputed (train median) and standardised.
CONTINUOUS_COLUMNS = (*AMOUNT_COLUMNS, AGE_COLUMN, *EXT_SOURCE_COLUMNS)
# Binary features are passed through untouched (already 0/1).
BINARY_COLUMNS = (SENSITIVE_COLUMN, *EXT_MISSING_COLUMNS)
FEATURE_COLUMNS = (SENSITIVE_COLUMN, *CONTINUOUS_COLUMNS,
                   *EXT_MISSING_COLUMNS)

CLASS_LABELS = {0: "Good payer", 1: "Bad payer"}

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------
EPSILON = 1e-8
PROBABILITY_CLIP = 1e-7
PREDICT_BATCH_SIZE = 8192


@dataclass(frozen=True)
class DataConfig:
    """Data loading and splitting options.

    Attributes:
        csv_path: Location of the extracted Home Credit CSV.
        zip_path: Location of the zipped CSV (used if the CSV is absent).
        val_size: Fraction of all rows used for validation.
        test_size: Fraction of all rows held out for the final test.
        sample_size: Optional stratified subsample size (quick mode).
    """

    csv_path: Path = RAW_CSV_PATH
    zip_path: Path = RAW_ZIP_PATH
    # Same 80/10/10 split the professor used in the class example.
    val_size: float = 0.10
    test_size: float = 0.10
    sample_size: int | None = None


@dataclass(frozen=True)
class Topology:
    """Architecture + optimiser settings of the credit classifier.

    Attributes:
        hidden_units: Units of each hidden dense layer.
        dropout_rates: Dropout rate after each hidden layer.
        activation: Activation of the hidden layers.
        learning_rate: Adam learning rate.
    """

    hidden_units: tuple[int, ...] = (64, 32)
    dropout_rates: tuple[float, ...] = (0.2, 0.2)
    activation: str = "relu"
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class DebtRatioConfig:
    """Settings of the custom debt-ratio layer.

    Attributes:
        ratio_caps: Saturation scale of (annuity/income, credit/income).
            Values well above the cap are squashed towards 1 by ``tanh``.
        exponent_bounds: Allowed range of the trainable shape exponents
            (same bounds as the professor's ``ExponentLayer``).
        log_ratio_clip: Clip for log-ratios before exponentiation.
    """

    ratio_caps: tuple[float, float] = (0.5, 10.0)
    exponent_bounds: tuple[float, float] = (0.1, 3.0)
    log_ratio_clip: float = 20.0


@dataclass(frozen=True)
class TrainingConfig:
    """Generic training loop settings.

    Attributes:
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size (large enough for a stable batch-level
            Pearson correlation inside the FAIR loss).
        patience: Early-stopping patience on ``val_loss``.
    """

    epochs: int = 40
    batch_size: int = 512
    patience: int = 6


@dataclass(frozen=True)
class FairnessConfig:
    """FAIR loss and model-selection settings.

    Attributes:
        lambdas: Values of the manual lambda sweep (0 = Base model).
        penalty: ``"squared"`` (professor's corr^2) or ``"abs"`` (|corr|).
        corr_threshold: Max |corr(y_hat, s)| accepted for "fair" models.
        decision_threshold: Probability threshold for the hard class.
    """

    lambdas: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0,
                                  50.0)
    penalty: str = "squared"
    corr_threshold: float = 0.05
    decision_threshold: float = 0.5


@dataclass(frozen=True)
class TunerConfig:
    """Keras Tuner (AutoML) search space and budget.

    Attributes:
        max_trials: Number of random-search trials.
        epochs: Max epochs per trial.
        patience: Early-stopping patience per trial.
        max_hidden_layers: Upper bound on the number of hidden layers.
        units_choices: Candidate widths for each hidden layer.
        activations: Candidate hidden activations.
        dropout_range: (min, max, step) for dropout; min > 0 keeps
            MC-dropout uncertainty meaningful for every topology.
        learning_rates: Candidate Adam learning rates.
        lambda_range: (min, max) of the log-uniform FAIR lambda.
        fairness_weight: Weight of |corr| in the scalarised objective
            ``val_auc - fairness_weight * val_fair_corr``.
        directory: Where Keras Tuner stores its trial state.
    """

    max_trials: int = 20
    epochs: int = 15
    patience: int = 3
    max_hidden_layers: int = 3
    units_choices: tuple[int, ...] = (16, 32, 64, 128)
    activations: tuple[str, ...] = ("relu", "elu", "tanh")
    dropout_range: tuple[float, float, float] = (0.1, 0.4, 0.1)
    learning_rates: tuple[float, ...] = (1e-2, 1e-3, 5e-4)
    lambda_range: tuple[float, float] = (0.1, 50.0)
    fairness_weight: float = 1.0
    directory: Path = TUNER_DIR


@dataclass(frozen=True)
class UncertaintyConfig:
    """Settings of the two uncertainty estimators.

    Attributes:
        hidden_units: Hidden layers of the auxiliary error model.
        learning_rate: Adam learning rate of the error model.
        use_prediction_feature: Feed the classifier's probability to the
            error model ("second option" shown in class).
        mc_samples: Stochastic forward passes for MC-dropout variance.
    """

    hidden_units: tuple[int, ...] = (32, 16)
    learning_rate: float = 1e-3
    use_prediction_feature: bool = True
    mc_samples: int = 50


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration aggregating every sub-config.

    Attributes:
        seed: Global random seed.
        output_dir: Folder receiving figures, tables and models.
        deterministic_ops: Ask TensorFlow for deterministic kernels.
    """

    seed: int = 42
    output_dir: Path = OUTPUT_DIR
    deterministic_ops: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    topology: Topology = field(default_factory=Topology)
    debt_ratio: DebtRatioConfig = field(default_factory=DebtRatioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    tuner: TunerConfig = field(default_factory=TunerConfig)
    uncertainty: UncertaintyConfig = field(
        default_factory=UncertaintyConfig
    )

    def quick(self) -> PipelineConfig:
        """Return a tiny, fast variant used for smoke tests.

        Returns:
            A copy of this configuration with a small data sample, few
            epochs and few tuner trials.
        """
        return replace(
            self,
            data=replace(self.data, sample_size=6000),
            training=replace(self.training, epochs=3, patience=2),
            fairness=replace(self.fairness, lambdas=(0.0, 5.0)),
            tuner=replace(self.tuner, max_trials=2, epochs=2, patience=1),
            uncertainty=replace(self.uncertainty, mc_samples=5),
        )
