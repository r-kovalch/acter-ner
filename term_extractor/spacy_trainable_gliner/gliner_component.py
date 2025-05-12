# gliner_component.py

import torch
from torch.nn import BCEWithLogitsLoss
from thinc.api import PyTorchWrapper, Model
from spacy.vocab import Vocab
from spacy.language import Language
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy import registry
from gliner import GLiNER


@registry.architectures("custom.GLiNERModel.v1")
def build_gliner_model(
    vocab: Vocab,
    model_name: str,
    labels: list[str] | None = None,
) -> Model:
    if labels is None:
        labels = ["TERM"]
    hf = GLiNER.from_pretrained(model_name, labels=labels)
    return PyTorchWrapper(hf)


class GLiNERPipe(TrainablePipe):
    def __init__(
        self,
        nlp,
        name: str,
        model_name: str,
        labels: list[str],
        threshold: float = 0.5,
    ):
        # Build the Thinc model
        thinc_model = registry.get("architectures", "custom.GLiNERModel.v1")(
            nlp.vocab, model_name, labels
        )
        # ⚠️ Pass vocab, not nlp
        super().__init__(nlp.vocab, name, thinc_model)
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
            for span_idx, cls_scores in enumerate(spanscores):
                lid = int(cls_scores.argmax().item())
                score = float(cls_scores[lid])
                if score >= self.threshold:
                    start, end = get_span(span_idx)
                    span = doc.char_span(start, end,
                                         label=self.labels[lid],
                                         alignment_mode="expand")
                    if span:
                        ents.append(span)
            doc.ents = ents

    def get_loss(self, docs, examples, scores):
        target = torch.zeros_like(scores)
        for i, example in enumerate(examples):
            for span in example.reference.ents:
                idx = self.model._func.find_span_index(
                    span.start_char, span.end_char
                )
                lid = self.labels.index(span.label_)
                target[i, idx, lid] = 1
        loss = self.loss_fn(scores, target)
        d_scores = torch.autograd.grad(loss, scores)[0]
        return loss, d_scores

    def initialize(self, get_examples, *args, **kwargs):
        self._require_labels()
        return {}


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
    return GLiNERPipe(nlp, name, model_name, labels, threshold)
