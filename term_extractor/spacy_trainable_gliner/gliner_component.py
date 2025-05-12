import torch
from torch.nn import BCEWithLogitsLoss
from thinc.api import PyTorchWrapper, Model
from spacy import registry
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.language import Language
from gliner import GLiNER


# 1) Architecture
@registry.architectures("custom.GLiNERModel.v1")
def build_gliner_model(model_name: str, labels: list[str]) -> Model:
    """Return a Thinc-wrapped GLiNER model ready for spaCy."""
    hf = GLiNER.from_pretrained(model_name, labels=labels)
    return PyTorchWrapper(hf)


# 2) Pipe
class GLiNERPipe(TrainablePipe):
    def __init__(
        self,
        name: str,
        model: Model,
        labels: list[str],
        threshold: float = 0.5,
    ):
        """Trainable spaCy pipe that wraps a GLiNER model.

        Args:
            name: Name of the pipeline component.
            model: The wrapped GLiNER model.
            labels: List of entity labels.
            threshold: Sigmoid score threshold for accepting entity spans.
        """
        super().__init__(name, model)
        self.labels = labels
        self.threshold = threshold
        self.loss_fn = BCEWithLogitsLoss()

    def predict(self, docs):
        # Thinc wrapper exposes .predict; GLiNER takes a list of Docs
        return self.model.predict(docs)

    def set_annotations(self, docs, scores):
        probs = torch.sigmoid(scores).cpu().detach()
        get_span = self.model._func.get_span  # type: ignore[attr-defined]

        for doc, span_scores in zip(docs, probs):
            ents = []
            for idx, cls_scores in enumerate(span_scores):
                lid = int(cls_scores.argmax().item())
                score = float(cls_scores[lid])
                if score >= self.threshold:
                    s, e = get_span(idx)
                    span = doc.char_span(
                        s,
                        e,
                        label=self.labels[lid],
                        alignment_mode="expand",
                    )
                    if span:
                        ents.append(span)
            doc.ents = ents

    def get_loss(self, docs, examples, scores):
        target = torch.zeros_like(scores)
        for i, ex in enumerate(examples):
            for span in ex.reference.ents:
                span_idx = self.model._func.find_span_index(
                    span.start_char, span.end_char
                )
                lid = self.labels.index(span.label_)
                target[i, span_idx, lid] = 1

        loss = self.loss_fn(scores, target)
        d_scores = torch.autograd.grad(loss, scores)[0]
        return loss, d_scores

    def initialize(self, get_examples, *args, **kwargs):
        # Populate labels if training from blank pipeline
        self._require_labels()
        return {}


# 3) Factory
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
    model = registry.get("architectures", "custom.GLiNERModel.v1")(
        model_name, labels
    )
    return GLiNERPipe(name, model, labels, threshold)
