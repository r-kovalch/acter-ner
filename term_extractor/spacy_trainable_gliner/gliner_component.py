# gliner_component.py

import torch
from torch.nn import BCEWithLogitsLoss
from thinc.api import PyTorchWrapper, Model
from spacy.vocab import Vocab
from spacy.language import Language
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy import registry
from gliner import GLiNER


# 1) Register a custom architecture that spaCy will build via Thinc
@registry.architectures("custom.GLiNERModel.v1")
def build_gliner_model(
    vocab: Vocab,
    model_name: str,
    labels: list[str],
) -> Model:
    """
    Called as (vocab, model_name, labels) → return a Thinc Model wrapping GLiNER.
    """
    hf = GLiNER.from_pretrained(model_name, labels=labels)
    return PyTorchWrapper(hf)  # Makes it trainable by spaCy


# 2) Subclass TrainablePipe exactly matching its __init__ signature
class GLiNERPipe(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        name: str,
        model: Model,
        labels: list[str],
        threshold: float = 0.5,
    ):
        # Must pass vocab, name, model positionally
        super().__init__(vocab, name, model)  # no keywords allowed :contentReference[oaicite:1]{index=1}
        self.labels = labels
        self.threshold = threshold
        self.loss_fn = BCEWithLogitsLoss()

    def predict(self, docs):
        return self.model(docs)

    def set_annotations(self, docs, scores):
        probs = torch.sigmoid(scores).detach().cpu()
        get_span = self.model._func.get_span
        for doc, spanscores in zip(docs, probs):
            ents = []
            for idx, cls_scores in enumerate(spanscores):
                lid = int(cls_scores.argmax().item())
                score = float(cls_scores[lid])
                if score >= self.threshold:
                    s, e = get_span(idx)
                    span = doc.char_span(s, e,
                                         label=self.labels[lid],
                                         alignment_mode="expand")
                    if span:
                        ents.append(span)
            doc.ents = ents

    def get_loss(self, docs, examples, scores):
        target = torch.zeros_like(scores)
        for i, example in enumerate(examples):
            for span in example.reference.ents:
                span_idx = self.model._func.find_span_index(
                    span.start_char, span.end_char
                )
                lid = self.labels.index(span.label_)
                target[i, span_idx, lid] = 1
        loss = self.loss_fn(scores, target)
        d_scores = torch.autograd.grad(loss, scores)[0]
        return loss, d_scores

    def initialize(self, get_examples, *args, **kwargs):
        self._require_labels()
        return {}


# 3) Register factory so spaCy can resolve "gliner" in your config
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
    # Build the Model via our registered architecture
    model = registry.get("architectures", "custom.GLiNERModel.v1")(
        nlp.vocab, model_name, labels
    )
    # Instantiate the pipe with (vocab, name, model, labels, threshold)
    return GLiNERPipe(nlp.vocab, name, model, labels, threshold)
