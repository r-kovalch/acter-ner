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
    Load GLiNER from HF *with* our label list, then wrap it.
    """
    hf = GLiNER.from_pretrained(model_name, labels=labels)
    return PyTorchWrapper(hf)  # ↩️ wraps the HF model in a trainable Thinc Model


# 2) Define the TrainablePipe subclass
class GLiNERPipe(TrainablePipe):
    def __init__(
        self,
        name: str,
        model_name: str,
        labels: list[str],
        threshold: float = 0.5,
    ):
        # Build the Thinc model from our registry entry
        thinc_model = registry.get("architectures", "custom.GLiNERModel.v1")(
            model_name, labels
        )
        # Note: TrainablePipe.__init__ signature is (self, name: str, model: Model)
        super().__init__(name, model=thinc_model)
        self.labels = labels
        self.threshold = threshold
        self.loss_fn = BCEWithLogitsLoss()

    def predict(self, docs):
        # Forward pass; returns raw logits tensor of shape (B, spans, L)
        return self.model(docs)

    def set_annotations(self, docs, scores):
        # Convert logits → probabilities → spans → doc.ents
        probs = torch.sigmoid(scores).detach().cpu()
        get_span = self.model._func.get_span  # GLiNER’s span-idx → (start,end) helper
        for doc, spanscores in zip(docs, probs):
            ents = []
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
        # Build a binary target tensor matching scores' shape
        target = torch.zeros_like(scores)
        for i, spans in enumerate(golds):
            for span_idx, label in spans:
                target[i, span_idx, label] = 1
        loss = self.loss_fn(scores, target)
        # Compute gradient w.r.t. the scores for backprop
        d_scores = torch.autograd.grad(loss, scores)[0]
        return loss, d_scores

    def initialize(self, get_examples, *args, **kwargs):
        # Ensure the labels from the config are valid
        self._require_labels()
        return {}


# 3) Register the factory so spaCy can build the pipe from your config
@Language.factory(
    "gliner",  # must match [components.gliner] in your config
    default_config={
        "model_name": "gliner-community/gliner_small-v2.5",
        "labels": ["TERM"],
        "threshold": 0.5,
    },
    assigns=["doc.ents"],
)
def make_gliner(
    nlp: Language, name: str, model_name: str, labels: list[str], threshold: float
):
    # spaCy will inject the nlp object; we only need it to access vocab etc.
    pipe = GLiNERPipe(name, model_name, labels, threshold)
    return pipe
