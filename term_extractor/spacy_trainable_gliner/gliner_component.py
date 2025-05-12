# gliner_component.py

import torch
from torch.nn import BCEWithLogitsLoss
from thinc.api import PyTorchWrapper, Model
from spacy.language import Language
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy import registry
from gliner import GLiNER


# 1) Register the architecture so spaCy can build the Thinc model
@registry.architectures("custom.GLiNERModel.v1")
def build_gliner_model(model_name: str, labels: list[str]) -> Model:
    """
    Load GLiNER from HuggingFace and wrap it as a Thinc Model.
    """
    hf = GLiNER.from_pretrained(model_name)
    hf.set_labels(labels)  # resize classifier dimension
    return PyTorchWrapper(hf)  # makes hf a Thinc Model


# 2) Define the TrainablePipe subclass
class GLiNERPipe(TrainablePipe):
    def __init__(
        self,
        nlp,
        name: str,
        model_name: str,
        labels: list[str],
        threshold: float = 0.5,
    ):
        # Build the Thinc model from our registry entry
        thinc_model = registry.get("architectures", "custom.GLiNERModel.v1")(
            model_name, labels
        )
        super().__init__(nlp, name, model=thinc_model)
        self.labels = labels
        self.threshold = threshold
        self.loss_fn = BCEWithLogitsLoss()

    def predict(self, docs):
        # Forward pass; returns raw logits tensor of shape (B, S, L)
        return self.model(docs)

    def set_annotations(self, docs, scores):
        # Convert logits → probabilities → spans → doc.ents
        probs = torch.sigmoid(scores).detach().cpu()
        for doc, spanscores in zip(docs, probs):
            ents = []
            # GLiNER’s PyTorchWrapper stores a helper to map span idx → (char_start, char_end)
            get_span = self.model._func.get_span
            for span_idx, cls_scores in enumerate(spanscores):
                label_id = int(cls_scores.argmax().item())
                score = float(cls_scores[label_id])
                if score >= self.threshold:
                    start_char, end_char = get_span(span_idx)
                    span = doc.char_span(
                        start_char,
                        end_char,
                        label=self.labels[label_id],
                        alignment_mode="expand",
                    )
                    if span is not None:
                        ents.append(span)
            doc.ents = ents

    def get_loss(self, docs, golds, scores):
        # Build binary targets of same shape as scores, mark gold spans as 1
        target = torch.zeros_like(scores)
        for i, spans in enumerate(golds):
            for span_idx, label in spans:
                target[i, span_idx, label] = 1
        loss = self.loss_fn(scores, target)
        # Compute gradient w.r.t. scores for backprop
        d_scores = torch.autograd.grad(loss, scores)[0]
        return loss, d_scores

    def initialize(self, get_examples, *args, **kwargs):
        # Required hook; ensure labels are set correctly
        self._require_labels()
        return {}


# 3) Register the factory so spaCy can build the pipe from your config
@Language.factory(
    "gliner",  # must match [components.gliner] in your config
    default_config={
        "model_name": "urchade/gliner_small-v2.5",
        "labels": ["TERM"],
        "threshold": 0.5,
    },
    assigns=["doc.ents"],
)
def make_gliner(
    nlp: Language, name: str, model_name: str, labels: list[str], threshold: float
):
    return GLiNERPipe(nlp, name, model_name, labels, threshold)
