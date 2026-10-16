# Repository file tree

Current consolidated source, checked 15 October 2026. Ignored model weights, environments, runtime reports, caches and generated packaging metadata are excluded. Test model files are stored outside the checkout.

Read the [documentation tour](../README.md), [architecture](../architecture.md), [lifecycle](../lifecycle.md) and [messaging guide](../messaging.md) for responsibilities and workflows.

```text
inferops/
├── .github/
│   └── workflows/
│       └── ci.yaml
├── configs/
│   └── platform.yaml
├── deploy/
│   ├── compose/
│   │   ├── auto.yaml
│   │   ├── cpu.yaml
│   │   ├── cuda.yaml
│   │   ├── deepstream.yaml
│   │   └── triton.yaml
│   ├── docker/
│   │   ├── api.Dockerfile
│   │   ├── deepstream.Dockerfile
│   │   └── minio.Dockerfile
│   ├── observability/
│   │   ├── grafana/
│   │   │   ├── dashboards/
│   │   │   │   └── inferops-overview.json
│   │   │   └── provisioning/
│   │   │       ├── dashboards/
│   │   │       │   └── inferops.yaml
│   │   │       └── datasources/
│   │   │           └── prometheus.yaml
│   │   ├── prometheus-native.yml
│   │   └── prometheus.yml
│   ├── README.md
│   └── compose.yaml
├── docs/
│   ├── architecture/
│   │   ├── 09-deployment.mmd
│   │   ├── 09-deployment.svg
│   │   ├── 10-shared-engine.mmd
│   │   ├── 10-shared-engine.svg
│   │   ├── 11-evidence-and-recovery.mmd
│   │   ├── 11-evidence-and-recovery.svg
│   │   ├── 12-package-layout.mmd
│   │   ├── 12-package-layout.svg
│   │   ├── 13-lifecycle.mmd
│   │   ├── 13-lifecycle.svg
│   │   ├── 14-redpanda.mmd
│   │   ├── 14-redpanda.svg
│   │   ├── 15-evidence-states.mmd
│   │   ├── 15-evidence-states.svg
│   │   ├── README.md
│   │   ├── index.html
│   │   ├── mermaid-config.json
│   │   └── repository-tree.md
│   ├── images/
│   │   ├── 01-dashboard-overview.png
│   │   ├── 02-core-services.png
│   │   ├── 03-runtime-snapshot.png
│   │   ├── 04-live-detection.png
│   │   ├── 05-execution-studio.png
│   │   ├── 06-activity-persistence.png
│   │   ├── 07-config-editor.png
│   │   ├── grafana-original-dashboard.png
│   │   ├── minio-original-console.png
│   │   ├── prometheus-original-ui.png
│   │   ├── workspace-desktop.png
│   │   ├── workspace-execution.png
│   │   └── workspace-mobile.png
│   ├── README.md
│   ├── adapters.md
│   ├── api.md
│   ├── architecture.md
│   ├── cleanup-verification.md
│   ├── interface-validation.md
│   ├── interfaces.md
│   ├── lifecycle.md
│   ├── messaging.md
│   ├── project-blocks.md
│   ├── python-library.md
│   ├── redesign.md
│   └── structure.md
├── inferops/
│   ├── adapters/
│   │   ├── inbound/
│   │   │   ├── cli/
│   │   │   │   ├── __init__.py
│   │   │   │   └── __main__.py
│   │   │   ├── http/
│   │   │   │   ├── static/
│   │   │   │   │   ├── dashboard.css
│   │   │   │   │   └── dashboard.js
│   │   │   │   ├── templates/
│   │   │   │   │   └── dashboard.html
│   │   │   │   ├── __init__.py
│   │   │   │   ├── app.py
│   │   │   │   ├── configs.py
│   │   │   │   ├── evidence.py
│   │   │   │   ├── main.py
│   │   │   │   ├── serialization.py
│   │   │   │   ├── system.py
│   │   │   │   └── ui.py
│   │   │   ├── workers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bootstrap.py
│   │   │   │   ├── execution.py
│   │   │   │   ├── pipeline_worker.py
│   │   │   │   └── topic_worker.py
│   │   │   ├── __init__.py
│   │   │   └── sources.py
│   │   ├── outbound/
│   │   │   ├── inference/
│   │   │   │   ├── plugins/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── detr_decode.py
│   │   │   │   │   ├── image_letterbox.py
│   │   │   │   │   ├── imagenet_normalize.py
│   │   │   │   │   ├── resize_normalize.py
│   │   │   │   │   ├── rfdetr_decode.py
│   │   │   │   │   ├── softmax_topk.py
│   │   │   │   │   └── yolo_decode.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── devices.py
│   │   │   │   ├── onnxruntime_backend.py
│   │   │   │   ├── synthetic_backend.py
│   │   │   │   ├── tensorrt_backend.py
│   │   │   │   └── triton_backend.py
│   │   │   ├── media/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── annotations.py
│   │   │   │   ├── deepstream.py
│   │   │   │   ├── encoder.py
│   │   │   │   └── spool.py
│   │   │   ├── messaging/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── delivery.py
│   │   │   │   ├── inproc.py
│   │   │   │   └── redpanda.py
│   │   │   ├── observability/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── logging.py
│   │   │   │   ├── metrics.py
│   │   │   │   └── tracing.py
│   │   │   ├── persistence/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── memory.py
│   │   │   │   └── sql.py
│   │   │   ├── projects/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── artifacts.py
│   │   │   │   ├── loader.py
│   │   │   │   ├── manifest.py
│   │   │   │   ├── model_scaffold.py
│   │   │   │   ├── models.py
│   │   │   │   ├── packages.py
│   │   │   │   ├── plugins.py
│   │   │   │   ├── preparation.py
│   │   │   │   ├── resolver.py
│   │   │   │   ├── scaffold.py
│   │   │   │   ├── settings.py
│   │   │   │   ├── validation.py
│   │   │   │   └── yaml.py
│   │   │   ├── stages/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── aggregate_window.py
│   │   │   │   ├── base.py
│   │   │   │   ├── decode_any.py
│   │   │   │   ├── decode_image.py
│   │   │   │   ├── decode_video.py
│   │   │   │   ├── factory.py
│   │   │   │   ├── infer.py
│   │   │   │   ├── postprocess_passthrough.py
│   │   │   │   ├── preprocess_generic.py
│   │   │   │   ├── registry.py
│   │   │   │   ├── route_by.py
│   │   │   │   ├── sink_file.py
│   │   │   │   ├── sink_stdout.py
│   │   │   │   ├── source_folder.py
│   │   │   │   ├── source_mjpeg.py
│   │   │   │   ├── source_rest.py
│   │   │   │   ├── source_rtsp.py
│   │   │   │   └── track.py
│   │   │   ├── storage/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── filesystem.py
│   │   │   │   └── s3.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── application/
│   │   ├── __init__.py
│   │   ├── delivery.py
│   │   ├── engine.py
│   │   ├── evidence.py
│   │   ├── execution.py
│   │   ├── incidents.py
│   │   ├── jobs.py
│   │   ├── models.py
│   │   ├── projects.py
│   │   └── sessions.py
│   ├── domain/
│   │   ├── fusion/
│   │   │   ├── __init__.py
│   │   │   ├── dsl.py
│   │   │   ├── engine.py
│   │   │   ├── signals.py
│   │   │   ├── spec.py
│   │   │   ├── state_machine.py
│   │   │   └── window.py
│   │   ├── tracking/
│   │   │   ├── __init__.py
│   │   │   ├── factory.py
│   │   │   ├── state.py
│   │   │   └── tracker.py
│   │   ├── __init__.py
│   │   ├── errors.py
│   │   ├── evidence.py
│   │   ├── incident_spec.py
│   │   ├── incidents.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── processing.py
│   │   └── types.py
│   ├── ports/
│   │   ├── __init__.py
│   │   ├── extensions.py
│   │   ├── inference.py
│   │   ├── media.py
│   │   ├── messaging.py
│   │   ├── persistence.py
│   │   ├── projects.py
│   │   └── storage.py
│   ├── sdk/
│   │   ├── __init__.py
│   │   └── client.py
│   ├── testing/
│   │   ├── configs/
│   │   │   ├── models/
│   │   │   │   └── tiny_classifier.yaml
│   │   │   └── pipelines/
│   │   │       └── tiny_classifier.yaml
│   │   ├── __init__.py
│   │   ├── fixtures.py
│   │   └── make_tiny_model.py
│   ├── __init__.py
│   ├── bootstrap.py
│   └── public.py
├── projects/
│   ├── driver_behaviour/
│   │   ├── configs/
│   │   │   ├── alerts/
│   │   │   │   └── driver.yaml
│   │   │   ├── fusion/
│   │   │   │   └── driver.yaml
│   │   │   └── pipelines/
│   │   │       ├── driver_behaviour.yaml
│   │   │       └── driver_video.yaml
│   │   ├── model_repository/
│   │   │   ├── face_mesh_onnx/
│   │   │   │   └── config.pbtxt
│   │   │   └── rfdetr_driver_behaviour_onnx/
│   │   │       └── config.pbtxt
│   │   ├── models/
│   │   │   ├── face_mesh/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── adapter.py
│   │   │   │   ├── features.py
│   │   │   │   └── model.yaml
│   │   │   ├── rfdetr_driver/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── adapter.py
│   │   │   │   └── model.yaml
│   │   │   └── __init__.py
│   │   ├── plugins/
│   │   │   ├── __init__.py
│   │   │   └── frame_annotator.py
│   │   ├── tests/
│   │   │   ├── conftest.py
│   │   │   ├── test_driver_api.py
│   │   │   ├── test_driver_behaviour.py
│   │   │   ├── test_frame_annotator.py
│   │   │   └── test_video_pipeline.py
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── project.yaml
│   │   └── pyproject.toml
│   └── README.md
├── research/
│   ├── driver_training/
│   │   ├── Dashcam shows Louisville bridge crash that left semi hanging over Ohio River.mp4
│   │   ├── RF_DETR_Train.ipynb
│   │   └── test.mp4
│   └── README.md
├── scripts/
│   ├── benchmark_redesign.py
│   ├── clean_workspace.py
│   ├── run_mode.py
│   ├── setup_container_runtime.sh
│   ├── setup_inferops.py
│   └── verify_video.py
├── tests/
│   ├── scenarios/
│   │   ├── README.md
│   │   ├── e2e_dashboard_scenario.py
│   │   └── verify_video.py
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_alerts.py
│   ├── test_api_smoke.py
│   ├── test_clean_workspace.py
│   ├── test_config_api.py
│   ├── test_config_stack.py
│   ├── test_configs_contract.py
│   ├── test_devices.py
│   ├── test_docker_startup.py
│   ├── test_domain_packs.py
│   ├── test_dsl.py
│   ├── test_event_persistence_and_ws.py
│   ├── test_execution_modes.py
│   ├── test_hexagonal_boundaries.py
│   ├── test_hexagonal_http_adapter.py
│   ├── test_infer_stage.py
│   ├── test_library.py
│   ├── test_lifecycle.py
│   ├── test_live_ws.py
│   ├── test_model_ports.py
│   ├── test_project_blocks.py
│   ├── test_project_preparation.py
│   ├── test_recoverable_evidence.py
│   ├── test_redesign_engine.py
│   ├── test_redesign_lifecycle.py
│   ├── test_redesign_persistence.py
│   ├── test_redesign_production.py
│   ├── test_redesign_projects_models.py
│   ├── test_redesign_workers.py
│   ├── test_streams_isolation.py
│   ├── test_testing_storage.py
│   ├── test_tracker.py
│   ├── test_transport_inproc.py
│   ├── test_triton_backend.py
│   ├── test_ui.py
│   ├── test_video_completion.py
│   ├── test_window.py
│   └── test_yaml_loader.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Makefile
├── README.md
├── compose.yaml
└── pyproject.toml
```
