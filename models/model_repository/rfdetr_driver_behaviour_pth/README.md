# RF-DETR checkpoint prerequisite

Put the trained detection checkpoint (`.pth` / `.pt`) here. Checkpoints and generated ONNX files are ignored by Git. Preserve the class order: driver, cigarette, food, phone, seatbelt, wheel. The existing project uses `DriverBehaviour` as its first display label; the supplied training metadata spells it `DriverBeauvoir`.

The optional converter supports RF-DETR Nano/Small/Medium/Large/Base detection architectures, detects known legacy weight signatures, and requires every learned parameter to match. Custom architectures require an exporter adapter; an arbitrary PyTorch state dictionary does not describe executable architecture by itself.

Create the isolated CPU export environment and configure `INFEROPS_EXPORT_PYTHON`, then use **Pipeline & models → Model preparation**. Upload the checkpoint or paste its absolute path inside this project. See [the conversion guide](../../../../docs/checkpoint-conversion.md).

The original ONNX remains available. Export jobs stage versioned artifacts and reports separately; they do not overwrite an active model.
