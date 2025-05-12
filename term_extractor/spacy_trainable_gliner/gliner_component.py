import torch
from torch.nn import BCEWithLogitsLoss
from thinc.api import PyTorchWrapper, Model
from spacy import registry
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.language import Language
from gliner import GLiNER

"""GLiNER spaCy component
~~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps a Hugging Face GLiNER model in a spaCy TrainablePipe so the model can be
fine‑tuned with the standard `spacy train` CLI.  The key gotcha is that
`TrainablePipe.__init__` requires exactly two positional arguments after
`self` (``name`` and ``model``).  Passing fewer will trigger the classic
``trainable_pipe __init__() takes exactly 3 positional arguments (2 given)``
traceback.  This implementation keeps the call signature correct and fixes a
few typos that caused alignment and label lookup issues in the original.
"""

# ---------------------------------------------------------------------------
# 1) Architecture registry entry
# ---------------------------------------------------------------------------
@registry.architectures("custom.GLiNERModel.v1")
def build_gliner_model(model_name: str, labels: list[str]) -> Model:  # noqa: D401
    """Return a Thinc model that wraps a GLiNER HF model."""
    hf_model = GLiNER.from_pretrained(model_name, labels=labels)
    return PyTorchWrapper(hf_model)


# ---------------------------------------------------------------------------
# 2) Trainable pipe
# ---------------------------------------------------------------------------
class GLiNERPipe(TrainablePipe):
    """spaCy pipeline component for GLiNER."""

    def __init__(
        self,
        name: str,
        model: Model,
        labels: list[str],
        threshold: float = 0.5,
    ) -> None:
        # >>>>> *DON'T BREAK THIS LINE!*  Two positional args required <<<<<
        super().__init__(name, model)

        self.labels = labels
        self.threshold = threshold
        self.loss_fn = BCEWithLogitsLoss()

    # ------------------------------
    # Inference helpers
    # ------------------------------
    def predict(self, docs):
        """Forward pass that returns raw logits (before sigmoid)."""
        return self.model.predict(docs)

    def set_annotations(self, docs, scores):
        """Convert logits to Doc.ents using the configured threshold."""
        probs = torch.sigmoid(scores).detach().cpu()
        get_span = self.model._func.get_span  # type: ignore[attr-defined]

        for doc, span_scores in zip(docs, probs):
            ents = []
            for idx, cls_scores in enumerate(span_scores):
                lid = int(cls_scores.argmax().item())
                score = float(cls_scores[lid])
                if score >= self.threshold:
                    start_char, end_char = get_span(idx)
                    span = doc.char_span(
                        start_char,
                        end_char,
                        label=self.labels[lid],
                        alignment_mode="expand",
                    )
                    if span is not None:
                        ents.append(span)
            doc.ents = ents

    # ------------------------------
    # Loss & back‑prop
    # ------------------------------
    def get_loss(self, docs, examples, scores):  # noqa: D401
        target = torch.zeros_like(scores)
        for i, ex in enumerate(examples):
            for span in ex.reference.ents:
                span_idx = self.model._func.find_span_index(
                    span.start_char, span.end_char
                )
                label_id = self.labels.index(span.label_)
                target[i, span_idx, label_id] = 1

        loss = self.loss_fn(scores, target)
        d_scores = torch.autograd.grad(loss, scores)[0]
        return loss, d_scores

    # ------------------------------
    # Initialisation
    # ------------------------------
    def initialize(self, get_examples, *args, **kwargs):  # noqa: D401
        # Ensure labels are set when training from blank pipelines
        self._require_labels()
        return {}


# ---------------------------------------------------------------------------
# 3) Factory
# ---------------------------------------------------------------------------
@Language.factory(
    "gliner",
    default_config={
        "model_name": "gliner-community/gliner_small-v2.5",
        "labels": ["TERM"],
        "threshold": 0.5,
    },
    assigns=["doc.ents"],
)
def make_gliner(
    nlp: Language,
    name: str,
    model_name: str,
    labels: list[str],
    threshold: float,
):
    """Factory that creates a GLiNERPipe."""
    model = registry.get("architectures", "custom.GLiNERModel.v1")(model_name, labels)
    return GLiNERPipe(name, model, labels, threshold)
