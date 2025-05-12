import torch, spacy
from torch.nn import BCEWithLogitsLoss
from thinc.api import PyTorchWrapper, Model
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.registry import registry
from gliner import GLiNER                       # HF checkpoint


@registry.architectures("custom.GLiNERModel.v1")
def build_gliner_model(model_name: str, labels: list[str]) -> Model:
    """Return a Thinc-wrapped GLiNER model."""
    hf = GLiNER.from_pretrained(model_name)
    hf.set_labels(labels)                      # Resize final layer
    return PyTorchWrapper(hf)                  # ↩️ becomes Thinc Model


@spacy.component("gliner")
class GLiNERPipe(TrainablePipe):
    def __init__(self, nlp, name, model_name, labels, threshold: float = .5):
        thinc_model = registry.get("architectures",
                                   "custom.GLiNERModel.v1")(model_name, labels)
        super().__init__(nlp, name, model=thinc_model)
        self.labels = labels
        self.threshold = threshold
        self.loss_fn = BCEWithLogitsLoss()

    # ---------- required hooks ----------
    def predict(self, docs):
        return self.model(docs)                # Thinc runs hf.forward()

    def set_annotations(self, docs, scores):
        # scores: (B, spans, L)
        probs = torch.sigmoid(scores).detach().cpu()
        for doc, spanscores in zip(docs, probs):
            ents = []
            for span_idx, cls_scores in enumerate(spanscores):
                label_id = int(cls_scores.argmax())
                score    = cls_scores[label_id].item()
                if score >= self.threshold:
                    start, end = self.model._func.get_span(span_idx)  # GLiNER helper
                    ents.append(doc.char_span(start, end,
                                               label=self.labels[label_id],
                                               alignment_mode="expand"))
            doc.ents = [e for e in ents if e is not None]

    def get_loss(self, docs, golds, scores):
        # golds = List[List[(start,end,label_id)]]; prepare target tensor
        target = torch.zeros_like(scores)
        for b, spans in enumerate(golds):
            for s_idx, label in spans:
                target[b, s_idx, label] = 1
        loss = self.loss_fn(scores, target)
        d_scores = torch.autograd.grad(loss, scores)[0]
        return loss, d_scores

    def initialize(self, get_examples, *_):
        self._require_labels()                 # spaCy utility check
        return {}

