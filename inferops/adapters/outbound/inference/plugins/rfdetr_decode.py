"""RF-DETR ONNX decoding with sigmoid scores and normalized cxcywh boxes."""
from inferops.adapters.outbound.inference.plugins.detr_decode import DetrDecode


class RFDetrDecode(DetrDecode):
    def __init__(self, conf_threshold: float = 0.5, boxes_out: str = "dets",
                 logits_out: str = "labels", background_index: int | None = None):
        super().__init__(conf_threshold=conf_threshold, boxes_out=boxes_out,
                         logits_out=logits_out, background_index=background_index,
                         activation="sigmoid", box_format="cxcywh")
