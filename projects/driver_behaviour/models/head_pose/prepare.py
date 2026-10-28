"""Download the pinned Qualcomm 6DRepNet export and embed its external tensors."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from inferops.adapters.outbound.projects.preparation import download_url

URL = ("https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/"
       "sixd_repnet/releases/v0.65.0/sixd_repnet-onnx-float.zip")


def install(target: Path) -> None:
    import onnx

    with TemporaryDirectory(prefix="sixdrepnet-") as directory:
        folder = Path(directory)
        archive = folder / "model.zip"
        download_url(URL, archive)
        with ZipFile(archive) as bundle:
            for name in ("pose_estimator.onnx", "pose_estimator.data"):
                # Extract only these known files, without trusting ZIP paths.
                (folder / name).write_bytes(bundle.read(f"sixd_repnet-onnx-float/{name}"))
        model = onnx.load(str(folder / "pose_estimator.onnx"), load_external_data=True)
        onnx.external_data_helper.convert_model_from_external_data(model)
        onnx.save(model, str(target), save_as_external_data=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    install(parser.parse_args().output)
