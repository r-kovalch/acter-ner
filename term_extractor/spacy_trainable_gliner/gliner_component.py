# gliner_component.py

import torch
from torch.nn import BCEWithLogitsLoss
from thinc.api import PyTorchWrapper, Model
from spacy.vocab import Vocab
from spacy.language import Language
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy import registry
from gliner import GLiNER


# 1) Architecture: spaCy will call this with (vocab, model_name, labels)
@registry.architectures("custom.GLiNERModel.v1")
def build_gliner_model(
    vocab: Vocab,
    model_name: str,
    labels: list[str],
) -> Model:
    # load HF model with labels, wrap in Thinc
    hf = GLiNER.from_pretrained(model_name, labels=labels)
    return PyTorchWrapper(hf)


# 2) Trainable pipe: __init__ only takes (name, model)
class GLiNERPipe(TrainablePipe):
    def __init__(
        self,
        name: str,
        model: Model,
        labels: list[str],
        threshold: float = 0.5,
    ):
        super().__init__(name, model)
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
                    span.start_char, span.end_char)
                lid = self.labels.index(span.label_)
                target[i, span_idx, lid] = 1
        loss = self.loss_fn(scores, target)
        d_scores = torch.autograd.grad(loss, scores)[0]
        return loss, d_scores

    def initialize(self, get_examples, *args, **kwargs):
        self._require_labels()
        return {}


# 3) Factory: build the Thinc model, then the pipe
@Language.factory(
    "gliner",
    default_config={
        "model_name": "gliner-community/gliner_small-v2.5",
        "labels": ["TERM"],
        "threshold": 0.5,
    },
    assigns=["doc.ents"],
)
def make_gliner(nlp: Language, name: str,
                model_name: str, labels: list[str], threshold: float):
    # Build the Thinc model via our architecture
    thinc_model = registry.get("architectures", "custom.GLiNERModel.v1")(
        nlp.vocab, model_name, labels
    )
    return GLiNERPipe(name, thinc_model, labels, threshold)
