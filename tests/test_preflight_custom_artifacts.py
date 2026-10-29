"""Preflight validates custom artifacts and project-scoped imports without loading models."""
from typer.testing import CliRunner

from inferops.adapters.inbound.cli.__main__ import app


def test_preflight_custom_task_uses_explicit_project_and_not_onnx_validation(tmp_path):
    root = tmp_path / 'face_project'
    for directory in ('models', 'plugins', 'configs/pipelines', 'configs/fusion', 'configs/alerts'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    core = tmp_path / 'configs'
    core.mkdir()
    (root / 'project.yaml').write_text('name: face_project\npipelines: [face]\n')
    (root / 'weights.task').write_bytes(b'custom-backend-owned-artifact')
    (root / 'models/task.yaml').write_text(
        'name: task\nbackend: .plugins.runtime:TaskBackend\ninputs: []\n'
        f'artifact: {{path: "{root / "weights.task"}"}}\n')
    (root / 'configs/pipelines/face.yaml').write_text(
        'name: face\nstages: [{name: face, stage: .plugins.runtime:TaskStage, params: {model: task}}]\n')
    (root / 'plugins/runtime.py').write_text(
        'from inferops.ports.inference import InferenceBackend\n'
        'from inferops.adapters.outbound.stages.base import Stage\n'
        'class TaskBackend(InferenceBackend):\n'
        '    def __init__(self): raise AssertionError("preflight must not construct models")\n'
        '    async def load(self, spec): pass\n'
        '    async def infer(self, inputs): return {}\n'
        'class TaskStage(Stage):\n'
        '    async def process(self, item): return item\n')
    result = CliRunner().invoke(app, ['preflight', '--domain-root', str(root), '--configs-dir', str(core)])
    assert result.exit_code == 0, result.output
