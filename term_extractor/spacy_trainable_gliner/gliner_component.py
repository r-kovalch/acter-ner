# gliner_component.py

import torch
from torch.nn import BCEWithLogitsLoss
from thinc.api import PyTorchWrapper, Model
from spacy.vocab import Vocab
from spacy.language import Language
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy import registry
from gliner import GLiNER


# 1) Architecture registration: takes (vocab, model_name, labels?)
@registry.architectures("custom.GLiNERModel.v1")
def build_gliner_model(
    vocab: Vocab,
    model_name: str,
    labels: list[str] | None = None  # allow None during init-fill
) -> Model:
    if labels is None:
        labels = ["TERM"]
    hf = GLiNER.from_pretrained(model_name, labels=labels)
    return PyTorchWrapper(hf)


# 2) Trainable pipe: __init__ signature must be (nlp, name, model)
class GLiNERPipe(TrainablePipe):
    def __init__(
        self,
        nlp,
        name: str,
        model_name: str,
        labels: list[str],
        threshold: float = 0.5,
    ):
        # Build Thinc model via our registry entry
        thinc_model = registry.get("architectures", "custom.GLiNERModel.v1")(
            nlp.vocab, model_name, labels
        )
        super().__init__(nlp, name, thinc_model)  # positional args only :contentReference[oaicite:3]{index=3}
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
                    s, e = get_span(span_idx)
                    span = doc.char_span(s, e, label=self.labels[lid], alignment_mode="expand")
                    if span:
                        ents.append(span)
            doc.ents = ents

    def get_loss(self, docs, examples, scores):
        # examples: List[Example], use example.reference.ents
        target = torch.zeros_like(scores)
        for i, example in enumerate(examples):
            for span in example.reference.ents:
                idx = self.model._func.find_span_index(span.start_char, span.end_char)
                lid = self.labels.index(span.label_)
                target[i, idx, lid] = 1
        loss = self.loss_fn(scores, target)
        d_scores = torch.autograd.grad(loss, scores)[0]
        return loss, d_scores

    def initialize(self, get_examples, *args, **kwargs):
        self._require_labels()
        return {}


# 3) Factory registration: name must match config
@Language.factory(
    "gliner",
    default_config={
        "model_name": "gliner-community/gliner_small-v2.5",
        "labels": ["TERM"],
        "threshold": 0.5,
    },
    assigns=["doc.ents"],
)
def make_gliner(nlp: Language, name: str, model_name: str, labels: list[str], threshold: float):
    return GLiNERPipe(nlp, name, model_name, labels, threshold)
