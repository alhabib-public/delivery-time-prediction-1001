"""Analysis layer: notebook display helpers, builders and post-analysis reporters.

Mirrors the ml-pipe split ``analysis/{builder,reporter}``: reporters own *how* results are
shown (one ``display(...)`` method each, no data in the constructor), builders own *how*
auxiliary objects are constructed (``build(...)``), and ``display`` wraps IPython so the
same code prints readable text when run from the shell.
"""

from .builders import BaselineBuilder, MetricsBuilder
from .display import (display_dataframe, display_figure, display_info_box, display_markdown,
                      display_title, in_notebook)
from .reporters import (CategoricalEffectReporter, CorrelationReporter, DataReporter, ErrorAnalysisReporter,
                        ExperimentsReporter, FeatureSummaryReporter, LeaderboardReporter, MissingnessReporter,
                        PostAnalysisReporter, PreAnalysisReporter, Reporter, SubmissionReporter, TargetReporter,
                        TrainingReporter)

__all__ = [
    "BaselineBuilder", "MetricsBuilder",
    "display_dataframe", "display_figure", "display_info_box", "display_markdown", "display_title",
    "in_notebook",
    "CategoricalEffectReporter", "CorrelationReporter", "DataReporter", "ErrorAnalysisReporter",
    "ExperimentsReporter", "FeatureSummaryReporter", "LeaderboardReporter", "MissingnessReporter",
    "PostAnalysisReporter", "PreAnalysisReporter", "Reporter", "SubmissionReporter", "TargetReporter",
    "TrainingReporter",
]
