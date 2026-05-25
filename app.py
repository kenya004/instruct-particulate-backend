from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

try:
    import spaces
except ImportError:
    spaces = None

from infer import (
    _base_metadata,
    _compute_motion_prediction_artifacts,
    _decode_face_part_ids,
    _denormalize_points,
    _joint_refit_metadata,
    _resolve_visualized_batch_link_point_prompts,
    _run_mesh_like_prediction,
    _save_articulated_mesh_outputs,
    _write_kinematic_and_overparam_visualization,
    _write_mesh_like_prediction_files,
    _write_metadata_and_summary,
    _tensor_to_numpy,
)
from instruct_particulate.model import Particulate2ArticulationModel
from instruct_particulate.utils.auto_kinematics_utils import (
    DEFAULT_AUTO_KINEMATICS_AZIMUTHS,
    call_auto_kinematics_model,
    lift_point_prompts_from_rendered_views,
    parse_auto_kinematics_response,
    prepare_lifted_prompt_records_for_saving,
    render_mesh_auto_kinematics_views,
    resolve_blender_binary,
    save_2d_point_prompt_visualizations,
    save_auto_kinematics_artifacts,
)
from instruct_particulate.utils.data_utils import (
    canonicalize_up_dir,
    load_trimesh,
    normalize_mesh,
    reorient_mesh_to_z_up,
)
from instruct_particulate.utils.export_utils import export_urdf
from instruct_particulate.utils.inference_utils import (
    build_joint_tensors,
    build_predicted_kinematic_records,
    configure_runtime_environment,
    load_model_checkpoint_for_inference,
    load_run_config,
    prepare_inference_batch_from_mesh,
    prepare_inference_batch_from_mesh_with_prompts,
    resolve_inference_sampling_config,
    run_batched_model_inference,
)
from instruct_particulate.utils.inference_visualization_utils import (
    save_predicted_point_query_rest_visualization,
)


REPO_ROOT = Path(__file__).resolve().parent
RUN_DIR = REPO_ROOT / "runs" / "FINAL-ours-final_2026-05-01T12-36-12"
CHECKPOINT_PATH = RUN_DIR / "checkpoints" / "step_0080000.pt"
CHECKPOINT_REPO_ID = os.environ.get(
    "INSTRUCT_PARTICULATE_CHECKPOINT_REPO_ID",
    "rayli/instruct-particulate-model",
)
CHECKPOINT_REPO_FILENAME = os.environ.get(
    "INSTRUCT_PARTICULATE_CHECKPOINT_REPO_FILENAME",
    "step_0080000.pt",
)
DEFAULT_OUTPUT_ROOT = Path("/data") if Path("/data").exists() else REPO_ROOT / "data"
OUTPUT_ROOT = Path(os.environ.get("INSTRUCT_PARTICULATE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
UP_DIR_CHOICES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
AUTO_KINEMATICS_MODEL_ID = "gemini-3.5-flash"
AUTO_KINEMATICS_REASONING_EFFORT = "medium"

DEFAULT_KINEMATIC_TREE = {
    "links": [
        {"id": 0, "name": "base"},
        {"id": 1, "name": "moving_part"},
    ],
    "joints": [
        {
            "parent": 0,
            "child": 1,
            "type": "revolute",
        }
    ],
}

DEMO_CSS = """
:root {
  --demo-top-row-height: 760px;
  --demo-bottom-row-height: 760px;
}
.demo-row {
  align-items: stretch !important;
  gap: 12px !important;
}
.demo-top-row {
  height: var(--demo-top-row-height);
  min-height: var(--demo-top-row-height);
  max-height: var(--demo-top-row-height);
  overflow: hidden;
}
.demo-bottom-row {
  height: var(--demo-bottom-row-height);
  min-height: var(--demo-bottom-row-height);
  max-height: var(--demo-bottom-row-height);
  overflow: hidden;
}
.demo-panel {
  height: 100%;
  min-height: 0;
  overflow: hidden;
}
.demo-panel > .form,
.demo-panel > div {
  min-height: 0;
}
.kinematic-json-sync {
  display: none !important;
}
.mesh-panel {
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding-right: 4px;
}
.mesh-panel .model3d-container,
.mesh-panel model-viewer {
  height: 360px !important;
}
.mesh-examples {
  flex: 0 0 auto;
  max-height: 105px;
  overflow-y: auto;
  border: 1px solid #d7deea;
  border-radius: 6px;
  padding: 6px;
  background: #f8fafc;
}
.mesh-examples table {
  margin: 0 !important;
}
.kin-panel {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.kin-panel .kinematic-editor-host {
  flex: 1 1 auto !important;
  min-height: 500px !important;
  height: auto !important;
  display: block !important;
  overflow: visible !important;
  opacity: 1 !important;
  visibility: visible !important;
}
.kin-panel .kinematic-editor-host > div {
  height: 100% !important;
  min-height: 500px !important;
}
#kinematic_tree_editor_html {
  display: block !important;
  min-height: 500px !important;
  opacity: 1 !important;
  visibility: visible !important;
}
.prompt-panel {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.prompt-panel .prompt-run-row {
  flex: 0 0 auto;
}
.orientation-panel {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.orientation-panel .markdown {
  flex: 0 0 auto !important;
  margin: 0 !important;
}
.orientation-panel .markdown h3,
.orientation-panel .markdown p {
  margin-bottom: 4px !important;
}
.upright-picker-grid {
  flex: 0 0 auto !important;
  min-height: 0 !important;
  height: auto !important;
  width: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 8px !important;
}
.upright-picker-row {
  flex: 0 0 auto !important;
  height: auto !important;
  min-height: 0 !important;
  gap: 8px !important;
  align-items: stretch !important;
}
.upright-picker-row > .form,
.upright-picker-row > div {
  flex: 1 1 0 !important;
  min-width: 0 !important;
  min-height: 0 !important;
}
.upright-option-image {
  position: relative !important;
  min-height: 0 !important;
  height: auto !important;
  aspect-ratio: 1 / 1 !important;
  border: 1px solid #d7deea !important;
  border-radius: 6px !important;
  background: #f8fafc !important;
  cursor: pointer;
  overflow: hidden !important;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}
.upright-option-image:hover {
  border-color: #94a3b8 !important;
}
.upright-option-image.selected {
  border-color: #2563eb !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.28) !important;
}
.upright-option-image .image-container,
.upright-option-image .image-frame,
.upright-option-image button {
  width: 100% !important;
  height: 100% !important;
  aspect-ratio: 1 / 1 !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  background: transparent !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
.upright-option-image .empty {
  height: 100% !important;
  min-height: 0 !important;
  margin: 0 !important;
}
.upright-option-image img {
  display: block !important;
  margin: auto !important;
  width: 100% !important;
  height: 100% !important;
  object-fit: cover !important;
  object-position: center center !important;
}
.upright-option-image .icon-button-wrapper {
  display: none !important;
}
.upright-option-image::after {
  content: attr(data-up-label);
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 1;
  border: 1px solid rgba(15, 23, 42, 0.16);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.88);
  color: #111827;
  font-size: 12px;
  font-weight: 600;
  line-height: 1;
  padding: 5px 7px;
  pointer-events: none;
}
.outputs-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  overflow-y: auto;
  padding-right: 4px;
}
.output-file-row {
  flex: 0 0 auto;
  align-items: center !important;
}
.urdf-export-row {
  flex: 0 0 44px;
  min-height: 44px !important;
  max-height: 44px;
  align-items: center !important;
  gap: 8px !important;
}
.output-triplet-row {
  flex: 0 0 430px;
  min-height: 0;
  max-height: 430px;
}
.output-triplet-row > .form,
.output-triplet-row > div {
  min-height: 0;
}
.capped-output {
  max-height: 430px;
  overflow: hidden;
}
.capped-output img {
  max-height: 390px !important;
  object-fit: contain !important;
}
.capped-output model-viewer,
.capped-output canvas {
  max-height: 390px !important;
}
"""

KINEMATIC_TREE_EDITOR_CSS = """
.kinematic-json-sync {
  display: none !important;
}
#kin-tree-editor,
#point-prompt-picker {
  --kin-panel-bg: #f8fafc;
  --kin-panel-border: #d7deea;
  --kin-toolbar-bg: #ffffff;
  --kin-canvas-bg: #fbfdff;
  --kin-grid: #edf1f7;
  --kin-text: #111827;
  --kin-muted: #4b5563;
  --kin-edge: #64748b;
  --kin-edge-selected: #0f172a;
  --kin-label-bg: #ffffff;
  --kin-label-border: #cbd5e1;
  --kin-control-bg: #ffffff;
  --kin-control-border: #c9d2df;
  --kin-control-text: #111827;
  --kin-primary-bg: #1f2937;
  --kin-primary-text: #ffffff;
  --kin-active-bg: #edf2ff;
  --kin-active-border: #5b78d6;
  --kin-active-text: #1f3a8a;
  --kin-focus: #2563eb;
  --kin-node-border: rgba(17, 24, 39, 0.28);
  --kin-node-shadow: rgba(15, 23, 42, 0.14);
  --kin-node-glass: rgba(255, 255, 255, 0.78);
  --kin-node-input-bg: rgba(255, 255, 255, 0.9);
  --kin-node-input-border: rgba(17, 24, 39, 0.22);
  --kin-node-text: #111827;
  --kin-close-bg: #ff5f57;
  --kin-close-border: #e0443e;
  --kin-close-cross: #555a60;
  color: var(--kin-text) !important;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  #kin-tree-editor,
  #point-prompt-picker {
    --kin-panel-bg: #111827;
    --kin-panel-border: #334155;
    --kin-toolbar-bg: #0f172a;
    --kin-canvas-bg: #0b1120;
    --kin-grid: rgba(148, 163, 184, 0.16);
    --kin-text: #f8fafc;
    --kin-muted: #cbd5e1;
    --kin-edge: #94a3b8;
    --kin-edge-selected: #f8fafc;
    --kin-label-bg: #1f2937;
    --kin-label-border: #475569;
    --kin-control-bg: #111827;
    --kin-control-border: #475569;
    --kin-control-text: #f8fafc;
    --kin-primary-bg: #e5e7eb;
    --kin-primary-text: #111827;
    --kin-active-bg: #1e3a8a;
    --kin-active-border: #93c5fd;
    --kin-active-text: #ffffff;
    --kin-focus: #60a5fa;
    --kin-node-border: rgba(255, 255, 255, 0.32);
    --kin-node-shadow: rgba(0, 0, 0, 0.35);
    --kin-node-glass: rgba(255, 255, 255, 0.78);
    --kin-node-input-bg: rgba(255, 255, 255, 0.9);
    --kin-node-input-border: rgba(17, 24, 39, 0.24);
    --kin-node-text: #111827;
    --kin-close-bg: #ff5f57;
    --kin-close-border: #e0443e;
    --kin-close-cross: #555a60;
    color-scheme: dark;
  }
}
.kin-tree-panel {
  border: 1px solid var(--kin-panel-border);
  border-radius: 8px;
  background: var(--kin-panel-bg);
  margin: 8px 0 0;
  overflow: hidden;
  flex: 1 1 auto;
  min-height: 500px;
  height: 100%;
  display: flex;
  flex-direction: column;
}
.kin-tree-header {
  padding: 12px 14px 9px;
  border-bottom: 1px solid var(--kin-panel-border);
}
.kin-tree-header h3 {
  margin: 0 0 6px;
  font-size: 16px;
  color: var(--kin-text) !important;
}
.kin-tree-header p {
  margin: 0;
  max-width: 68ch;
  color: var(--kin-muted) !important;
  font-size: 13px;
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.kin-tree-header strong {
  color: var(--kin-text) !important;
}
.kin-tree-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  padding: 10px 12px;
  border-bottom: 1px solid var(--kin-panel-border);
  background: var(--kin-toolbar-bg);
}
#kin-tree-editor .kin-tree-toolbar button,
#kin-tree-editor .kin-tree-toolbar select {
  height: 34px;
  border: 1px solid var(--kin-control-border);
  border-radius: 6px;
  background: var(--kin-control-bg);
  color: var(--kin-control-text) !important;
  font-size: 13px;
  padding: 0 10px;
}
#kin-tree-editor .kin-tree-toolbar select option {
  background: var(--kin-control-bg) !important;
  color: var(--kin-control-text) !important;
}
.kin-tree-toolbar button {
  flex: 0 0 auto;
  cursor: pointer;
}
.kin-tree-toolbar button.primary {
  background: var(--kin-primary-bg);
  color: var(--kin-primary-text) !important;
  border-color: var(--kin-primary-bg);
}
.kin-tree-toolbar button.active {
  background: var(--kin-active-bg);
  border-color: var(--kin-active-border);
  color: var(--kin-active-text) !important;
}
.kin-tree-status {
  margin-left: auto;
  color: var(--kin-muted) !important;
  font-size: 13px;
}
.kin-tree-canvas {
  position: relative;
  flex: 1 1 auto;
  min-height: 340px;
  height: 100%;
  overflow: hidden;
  background:
    linear-gradient(var(--kin-grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--kin-grid) 1px, transparent 1px),
    var(--kin-canvas-bg);
  background-size: 24px 24px;
  user-select: none;
}
.kin-tree-edge-layer {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: auto;
  z-index: 1;
}
.kin-tree-edge {
  stroke: var(--kin-edge);
  stroke-width: 2.4;
  fill: none;
  cursor: pointer;
}
.kin-tree-edge.selected {
  stroke: var(--kin-edge-selected);
  stroke-width: 3.4;
}
.kin-tree-edge-label rect {
  fill: var(--kin-label-bg);
  stroke: var(--kin-label-border);
  stroke-width: 1;
}
.kin-tree-edge-label text {
  fill: var(--kin-text);
  font-size: 12px;
  font-weight: 700;
}
.kin-tree-edge-label {
  cursor: pointer;
}
.kin-tree-joint-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 2;
}
.kin-joint-chip {
  position: absolute;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 30px;
  padding: 3px 4px 3px 6px;
  border: 1px solid var(--kin-label-border);
  border-radius: 7px;
  background: var(--kin-label-bg);
  color: var(--kin-text) !important;
  box-shadow: 0 6px 14px var(--kin-node-shadow);
  cursor: pointer;
  pointer-events: auto;
  transform: translate(-50%, -50%);
  white-space: nowrap;
}
.kin-joint-chip.selected {
  border-color: var(--kin-edge-selected);
  outline: 2px solid var(--kin-edge-selected);
  outline-offset: 1px;
}
#kin-tree-editor .kin-joint-motion {
  appearance: auto !important;
  -webkit-appearance: menulist !important;
  height: 24px;
  max-width: 92px;
  border: 1px solid #c9d2df !important;
  border-radius: 5px !important;
  background: #ffffff !important;
  background-color: #ffffff !important;
  background-image: none !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
  color-scheme: light !important;
  font-size: 12px;
  padding: 0 4px;
  opacity: 1 !important;
}
#kin-tree-editor .kin-joint-motion:hover,
#kin-tree-editor .kin-joint-motion:focus {
  background: #ffffff !important;
  background-color: #ffffff !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}
#kin-tree-editor .kin-joint-motion option {
  background: #ffffff !important;
  background-color: #ffffff !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}
.kin-joint-delete {
  flex: 0 0 auto;
}
.kin-tree-node-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 3;
}
.kin-node {
  position: absolute;
  box-sizing: border-box;
  width: 176px;
  height: 44px;
  border: 2px solid var(--kin-node-border);
  border-radius: 8px;
  box-shadow: 0 8px 18px var(--kin-node-shadow);
  cursor: grab;
  overflow: hidden;
  pointer-events: auto;
  display: flex;
  align-items: center;
}
.kin-node.dragging {
  cursor: grabbing;
  box-shadow: 0 14px 28px var(--kin-node-shadow);
}
.kin-node.selected {
  outline: 3px solid var(--kin-edge-selected);
  outline-offset: 2px;
}
.kin-node.pending-parent {
  outline: 3px solid var(--kin-focus);
  outline-offset: 2px;
}
.kin-node-row {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 6px 8px;
}
.kin-node-delete {
  flex: 0 0 auto;
}
#kin-tree-editor .kin-node-delete,
#kin-tree-editor .kin-joint-delete {
  appearance: none;
  -webkit-appearance: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  width: 18px;
  max-width: 18px;
  min-height: 18px;
  height: 18px;
  max-height: 18px;
  border: 1px solid var(--kin-close-border) !important;
  border-radius: 999px !important;
  background: var(--kin-close-bg) !important;
  background-color: var(--kin-close-bg) !important;
  background-image: none !important;
  color: var(--kin-close-cross) !important;
  -webkit-text-fill-color: var(--kin-close-cross) !important;
  cursor: pointer;
  font-size: 13px;
  font-weight: 800;
  line-height: 1;
  padding: 0;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5);
}
#kin-tree-editor .kin-node-delete:hover,
#kin-tree-editor .kin-node-delete:focus,
#kin-tree-editor .kin-joint-delete:hover,
#kin-tree-editor .kin-joint-delete:focus {
  background: var(--kin-close-bg) !important;
  background-color: var(--kin-close-bg) !important;
  background-image: none !important;
  color: var(--kin-close-cross) !important;
  -webkit-text-fill-color: var(--kin-close-cross) !important;
}
.kin-node input {
  flex: 1 1 auto;
  min-width: 0;
  height: 30px;
  border: 1px solid var(--kin-node-input-border);
  border-radius: 5px;
  background: var(--kin-node-input-bg);
  color: var(--kin-node-text) !important;
  padding: 0 7px;
  font-size: 13px;
}
.kin-node input::placeholder {
  color: #4b5563 !important;
}
.prompt-picker-panel {
  border: 1px solid var(--kin-panel-border);
  border-radius: 8px;
  background: var(--kin-panel-bg);
  margin: 0;
  overflow: hidden;
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.prompt-picker-header {
  padding: 10px 12px;
  border-bottom: 1px solid var(--kin-panel-border);
}
.prompt-picker-header h3 {
  margin: 0 0 5px;
  font-size: 15px;
  color: var(--kin-text) !important;
}
.prompt-picker-header p {
  margin: 0;
  max-width: 62ch;
  color: var(--kin-muted) !important;
  font-size: 13px;
  line-height: 1.4;
  overflow-wrap: anywhere;
}
.prompt-picker-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  padding: 9px 12px;
  border-bottom: 1px solid var(--kin-panel-border);
  background: var(--kin-toolbar-bg);
}
#point-prompt-picker .prompt-picker-toolbar button {
  flex: 0 0 auto;
  height: 32px;
  border: 1px solid var(--kin-control-border);
  border-radius: 6px;
  background: var(--kin-control-bg);
  color: var(--kin-control-text) !important;
  font-size: 13px;
  padding: 0 10px;
  cursor: pointer;
}
.prompt-picker-status {
  margin-left: auto;
  color: var(--kin-muted) !important;
  font-size: 13px;
}
.prompt-picker-canvas-wrap {
  position: relative;
  flex: 1 1 auto;
  min-height: 0;
  height: auto;
  background:
    radial-gradient(circle at 50% 42%, rgba(148, 163, 184, 0.18), transparent 62%),
    var(--kin-canvas-bg);
}
#point-prompt-canvas {
  display: block;
  width: 100%;
  height: 100%;
  cursor: crosshair;
}
.prompt-picker-empty {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 18px;
  color: var(--kin-muted) !important;
  font-size: 13px;
  text-align: center;
  pointer-events: none;
}
.kin-node.has-prompt {
  box-shadow:
    0 8px 18px var(--kin-node-shadow),
    0 0 0 3px rgba(34, 197, 94, 0.78);
}
"""

UPRIGHT_PICKER_JS = r"""
() => {
  const choices = [
    ["+X", "posX"],
    ["-X", "negX"],
    ["+Y", "posY"],
    ["-Y", "negY"],
    ["+Z", "posZ"],
    ["-Z", "negZ"]
  ];

  if (window.__instructParticulateUprightPickerReady === true) {
    return;
  }
  window.__instructParticulateUprightPickerReady = true;

  function selectedField() {
    return document.querySelector("#selected_up_dir textarea, #selected_up_dir input");
  }

  function cardForSlug(slug) {
    return document.getElementById(`upright-option-${slug}`);
  }

  function updateCards(selectedValue) {
    choices.forEach(([upDir, slug]) => {
      const card = cardForSlug(slug);
      if (!card) {
        return;
      }
      const selected = selectedValue === upDir;
      card.classList.toggle("selected", selected);
      card.setAttribute("role", "button");
      card.setAttribute("tabindex", "0");
      card.setAttribute("aria-label", `Select ${upDir} as the upright orientation`);
      card.setAttribute("aria-pressed", selected ? "true" : "false");
      card.setAttribute("data-up-label", `${upDir} up`);
      card.title = `${upDir} up`;
    });
  }

  function setSelected(upDir) {
    const field = selectedField();
    if (!field) {
      return;
    }
    field.value = upDir;
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
    updateCards(upDir);
  }

  function handleCardAction(event) {
    for (const [upDir, slug] of choices) {
      const card = cardForSlug(slug);
      if (card && card.contains(event.target)) {
        event.preventDefault();
        event.stopPropagation();
        setSelected(upDir);
        return;
      }
    }
  }

  document.addEventListener("click", handleCardAction, true);
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    handleCardAction(event);
  }, true);

  const syncFromField = () => {
    const field = selectedField();
    updateCards(field ? field.value : "");
  };
  const observer = new MutationObserver(syncFromField);
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["class", "value"]
  });
  window.setInterval(syncFromField, 500);
  syncFromField();
}
"""

KINEMATIC_TREE_EDITOR_JS = r"""
() => {
  const palette = [
    "#fca5a5", "#fdba74", "#fde047", "#86efac",
    "#67e8f9", "#93c5fd", "#c4b5fd", "#f0abfc",
    "#f9a8d4", "#a7f3d0", "#fcd34d", "#bfdbfe",
    "#ddd6fe", "#fecaca", "#bbf7d0", "#bae6fd"
  ];
  const NODE_WIDTH = 176;
  const NODE_HEIGHT = 44;
  const PROMPT_POINT_PRECISION = 6;

  function waitForEditor() {
    const root = document.getElementById("kin-tree-editor");
    const syncBox = document.querySelector("#kinematic_tree_json textarea");
    const promptMeshBox = document.querySelector("#point_prompt_mesh_data textarea");
    const promptSyncBox = document.querySelector("#point_prompt_json textarea");
    const promptCanvas = document.getElementById("point-prompt-canvas");
    if (!root || !syncBox || !promptMeshBox || !promptSyncBox || !promptCanvas) {
      window.setTimeout(waitForEditor, 200);
      return;
    }
    if (root.dataset.ready === "1") {
      return;
    }
    root.dataset.ready = "1";
    initEditor(root, syncBox);
  }

  function initEditor(root, syncBox) {
    const canvas = root.querySelector("#kin-tree-canvas");
    const nodeLayer = root.querySelector("#kin-tree-nodes");
    const edgeLayer = root.querySelector("#kin-tree-edges");
    const jointLayer = root.querySelector("#kin-tree-joints");
    const addNodeButton = root.querySelector("#kin-add-node");
    const addJointButton = root.querySelector("#kin-add-joint");
    const deleteButton = root.querySelector("#kin-delete-selected");
    const resetButton = root.querySelector("#kin-reset-tree");
    const status = root.querySelector("#kin-tree-status");
    const promptRoot = document.getElementById("point-prompt-picker");
    const promptCanvas = document.getElementById("point-prompt-canvas");
    const promptEmpty = document.getElementById("point-prompt-empty");
    const promptStatus = document.getElementById("point-prompt-status");
    const clearPromptButton = document.getElementById("point-prompt-clear-link");
    const clearAllPromptsButton = document.getElementById("point-prompt-clear-all");
    const resetPromptViewButton = document.getElementById("point-prompt-reset-view");
    const promptMeshBox = document.querySelector("#point_prompt_mesh_data textarea");
    const promptSyncBox = document.querySelector("#point_prompt_json textarea");
    const defaultTree = JSON.parse(root.dataset.defaultTree);
    const state = {
      links: [],
      joints: [],
      prompts: {},
      selectedNode: null,
      selectedEdge: null,
      connectMode: false,
      pendingParent: null,
      dragging: null,
      suppressClick: false,
      promptMesh: null,
      promptCamera: {
        yaw: -0.65,
        pitch: 0.35,
        distance: 2.6,
        target: [0, 0, 0],
        radius: 1
      },
      promptDrag: null,
      lastPromptMeshValue: null,
      lastTreeValue: null,
      lastPromptValue: null
    };

    function defaultNodePosition(index) {
      const width = canvas.clientWidth || 320;
      const height = canvas.clientHeight || 380;
      const columns = width >= 680 ? 4 : width >= 500 ? 3 : width >= 340 ? 2 : 1;
      const usableX = Math.max(0, width - NODE_WIDTH - 48);
      const usableY = Math.max(0, height - NODE_HEIGHT - 48);
      const columnGap = columns <= 1 ? 0 : usableX / Math.max(1, columns - 1);
      const rowCount = Math.max(1, Math.ceil((index + 1) / columns));
      const rowGap = rowCount <= 1 ? 0 : Math.min(118, usableY / Math.max(1, rowCount - 1));
      return {
        x: 24 + (index % columns) * columnGap,
        y: 24 + Math.floor(index / columns) * rowGap
      };
    }

    function layoutTreePositions(rawLinks, joints) {
      const width = Math.max(canvas.clientWidth || 640, NODE_WIDTH + 72);
      const height = Math.max(canvas.clientHeight || 420, NODE_HEIGHT + 72);
      const marginX = 28;
      const marginY = 28;
      const usableX = Math.max(0, width - NODE_WIDTH - marginX * 2);
      const usableY = Math.max(0, height - NODE_HEIGHT - marginY * 2);
      const linkCount = rawLinks.length;
      const childrenByParent = new Map();
      const childIds = new Set();
      joints.forEach((joint) => {
        if (
          joint.parent >= 0 &&
          joint.parent < linkCount &&
          joint.child >= 0 &&
          joint.child < linkCount
        ) {
          if (!childrenByParent.has(joint.parent)) {
            childrenByParent.set(joint.parent, []);
          }
          childrenByParent.get(joint.parent).push(joint.child);
          childIds.add(joint.child);
        }
      });
      const roots = [];
      for (let index = 0; index < linkCount; index += 1) {
        if (!childIds.has(index)) {
          roots.push(index);
        }
      }
      if (roots.length === 0 && linkCount > 0) {
        roots.push(0);
      }

      const depths = new Array(linkCount).fill(null);
      const queue = roots.map((id) => ({ id, depth: 0 }));
      while (queue.length > 0) {
        const { id, depth } = queue.shift();
        if (depths[id] !== null && depths[id] <= depth) {
          continue;
        }
        depths[id] = depth;
        (childrenByParent.get(id) || []).forEach((childId) => {
          queue.push({ id: childId, depth: depth + 1 });
        });
      }
      for (let index = 0; index < linkCount; index += 1) {
        if (depths[index] === null) {
          depths[index] = 0;
        }
      }

      const levels = new Map();
      depths.forEach((depth, index) => {
        if (!levels.has(depth)) {
          levels.set(depth, []);
        }
        levels.get(depth).push(index);
      });
      const sortedDepths = Array.from(levels.keys()).sort((a, b) => a - b);
      const maxDepth = sortedDepths.length > 0 ? Math.max(...sortedDepths) : 0;
      const positions = new Array(linkCount);
      sortedDepths.forEach((depth) => {
        const ids = levels.get(depth);
        const x = marginX + (maxDepth <= 0 ? usableX * 0.5 : usableX * depth / maxDepth);
        ids.forEach((id, levelIndex) => {
          const y = marginY + (
            ids.length <= 1
              ? usableY * 0.5
              : usableY * levelIndex / Math.max(1, ids.length - 1)
          );
          positions[id] = { x, y };
        });
      });
      return positions;
    }

    function loadTree(tree, options = {}) {
      const links = Array.isArray(tree.links) ? tree.links : [];
      const joints = Array.isArray(tree.joints)
        ? tree.joints
            .map((joint) => ({
              parent: Number(joint.parent ?? joint.parent_link_id),
              child: Number(joint.child ?? joint.child_link_id),
              type: String(joint.type || joint.joint_type || "revolute").toLowerCase()
            }))
            .filter((joint) => Number.isInteger(joint.parent) && Number.isInteger(joint.child))
        : [];
      const positions = layoutTreePositions(links, joints);
      state.links = links.map((link, index) => ({
        ...(positions[index] || defaultNodePosition(index)),
        id: index,
        name: String(link.name || `link_${index}`),
        color: palette[index % palette.length]
      }));
      state.joints = joints;
      state.selectedNode = null;
      state.selectedEdge = null;
      state.connectMode = false;
      state.pendingParent = null;
      if (!options.preservePrompts) {
        state.prompts = {};
      }
      if (options.syncPrompts !== false) {
        syncPrompts();
      }
      render();
    }

    function exportTree() {
      return {
        links: state.links.map((link, index) => ({
          id: index,
          name: link.name.trim() || `link_${index}`
        })),
        joints: state.joints.map((joint) => ({
          parent: joint.parent,
          child: joint.child,
          type: joint.type
        }))
      };
    }

    function syncTree() {
      const value = JSON.stringify(exportTree(), null, 2);
      state.lastTreeValue = value;
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value"
      ).set;
      setter.call(syncBox, value);
      syncBox.dispatchEvent(new Event("input", { bubbles: true }));
      syncBox.dispatchEvent(new Event("change", { bubbles: true }));
    }

    function roundVector(values) {
      return values.map((value) => Number(Number(value).toFixed(PROMPT_POINT_PRECISION)));
    }

    function syncPrompts() {
      if (!promptSyncBox) {
        return;
      }
      const prompts = Object.entries(state.prompts)
        .map(([linkId, prompt]) => ({
          link_id: Number(linkId),
          point: roundVector(prompt.point),
          normal: roundVector(prompt.normal)
        }))
        .sort((a, b) => a.link_id - b.link_id);
      const value = JSON.stringify({ prompts }, null, 2);
      state.lastPromptValue = value;
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value"
      ).set;
      setter.call(promptSyncBox, value);
      promptSyncBox.dispatchEvent(new Event("input", { bubbles: true }));
      promptSyncBox.dispatchEvent(new Event("change", { bubbles: true }));
    }

    function selectedLinkName() {
      const link = getLink(state.selectedNode);
      return link ? (link.name.trim() || `link_${link.id}`) : null;
    }

    function updatePromptStatus(message) {
      if (!promptStatus) {
        return;
      }
      if (message) {
        promptStatus.textContent = message;
        return;
      }
      if (!state.promptMesh) {
        promptStatus.textContent = "Upload a mesh first.";
        return;
      }
      const name = selectedLinkName();
      if (name === null) {
        promptStatus.textContent = "Select a link, then click the mesh.";
        return;
      }
      const hasPrompt = state.prompts[state.selectedNode] !== undefined;
      promptStatus.textContent = hasPrompt
        ? `Prompt set for ${name}. Click mesh to replace it.`
        : `Click mesh to set prompt for ${name}.`;
    }

    function resizePromptCanvas() {
      if (!promptCanvas) {
        return;
      }
      const rect = promptCanvas.getBoundingClientRect();
      const nextWidth = Math.max(1, Math.floor(rect.width * window.devicePixelRatio));
      const nextHeight = Math.max(1, Math.floor(rect.height * window.devicePixelRatio));
      if (promptCanvas.width !== nextWidth || promptCanvas.height !== nextHeight) {
        promptCanvas.width = nextWidth;
        promptCanvas.height = nextHeight;
      }
    }

    function resetPromptCamera() {
      if (!state.promptMesh) {
        return;
      }
      state.promptCamera.target = state.promptMesh.center.slice();
      state.promptCamera.radius = Math.max(1e-6, state.promptMesh.radius || 1);
      state.promptCamera.distance = state.promptCamera.radius * 2.9;
      state.promptCamera.yaw = -0.65;
      state.promptCamera.pitch = 0.35;
      renderPromptMesh();
    }

    function loadExternalTreeFromBox() {
      if (!syncBox) {
        return;
      }
      const value = syncBox.value || "";
      if (!value.trim() || value === state.lastTreeValue) {
        return;
      }
      try {
        const payload = JSON.parse(value);
        state.lastTreeValue = value;
        loadTree(payload, { syncPrompts: false });
        setStatus("Loaded extracted kinematic tree.");
      } catch (error) {
        setStatus(`Could not load kinematic tree: ${error.message}`);
      }
    }

    function loadExternalPromptsFromBox() {
      if (!promptSyncBox) {
        return;
      }
      const value = promptSyncBox.value || "";
      if (value === state.lastPromptValue) {
        return;
      }
      state.lastPromptValue = value;
      const nextPrompts = {};
      if (value.trim()) {
        try {
          const payload = JSON.parse(value);
          const prompts = Array.isArray(payload) ? payload : (payload.prompts || []);
          prompts.forEach((prompt) => {
            const linkId = Number(prompt.link_id ?? prompt.link);
            const point = prompt.point;
            const normal = prompt.normal;
            if (
              Number.isInteger(linkId) &&
              Array.isArray(point) &&
              point.length === 3 &&
              Array.isArray(normal) &&
              normal.length === 3
            ) {
              nextPrompts[linkId] = {
                point: roundVector(point),
                normal: roundVector(normal)
              };
            }
          });
        } catch (error) {
          updatePromptStatus(`Could not load point prompts: ${error.message}`);
          return;
        }
      }
      state.prompts = nextPrompts;
      updatePromptStatus();
      render();
      renderPromptMesh();
    }

    function loadPromptMeshFromBox() {
      if (!promptMeshBox) {
        return;
      }
      const value = promptMeshBox.value || "";
      if (value === state.lastPromptMeshValue) {
        return;
      }
      state.lastPromptMeshValue = value;
      state.prompts = {};
      syncPrompts();
      if (!value.trim()) {
        state.promptMesh = null;
        renderPromptMesh();
        updatePromptStatus("Upload a mesh first.");
        return;
      }
      try {
        const payload = JSON.parse(value);
        state.promptMesh = {
          vertices: payload.vertices || [],
          faces: payload.faces || [],
          normals: payload.normals || [],
          center: payload.center || [0, 0, 0],
          radius: Number(payload.radius || 1),
          sampled: Boolean(payload.sampled),
          displayFaces: Number(payload.display_faces || 0),
          sourceFaces: Number(payload.source_faces || 0)
        };
        resetPromptCamera();
        updatePromptStatus();
      } catch (error) {
        state.promptMesh = null;
        renderPromptMesh();
        updatePromptStatus(`Could not load prompt mesh: ${error.message}`);
      }
    }

    function addVec(a, b) {
      return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    }

    function subVec(a, b) {
      return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    }

    function scaleVec(a, value) {
      return [a[0] * value, a[1] * value, a[2] * value];
    }

    function dotVec(a, b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    function crossVec(a, b) {
      return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
      ];
    }

    function normalizeVec(a) {
      const length = Math.hypot(a[0], a[1], a[2]);
      if (length <= 1e-8) {
        return [0, 0, 0];
      }
      return [a[0] / length, a[1] / length, a[2] / length];
    }

    function promptCameraBasis() {
      const camera = state.promptCamera;
      const pitch = Math.max(-1.35, Math.min(1.35, camera.pitch));
      camera.pitch = pitch;
      const cosPitch = Math.cos(pitch);
      const offset = [
        Math.sin(camera.yaw) * cosPitch * camera.distance,
        Math.sin(pitch) * camera.distance,
        Math.cos(camera.yaw) * cosPitch * camera.distance
      ];
      const eye = addVec(camera.target, offset);
      const forward = normalizeVec(subVec(camera.target, eye));
      let right = normalizeVec(crossVec(forward, [0, 1, 0]));
      if (Math.hypot(right[0], right[1], right[2]) <= 1e-8) {
        right = [1, 0, 0];
      }
      const up = normalizeVec(crossVec(right, forward));
      return { eye, forward, right, up };
    }

    function projectPromptPoint(point, basis, width, height, focal) {
      const rel = subVec(point, basis.eye);
      const z = dotVec(rel, basis.forward);
      if (z <= 1e-5) {
        return null;
      }
      const x = dotVec(rel, basis.right);
      const y = dotVec(rel, basis.up);
      return {
        x: width * 0.5 + focal * x / z,
        y: height * 0.5 - focal * y / z,
        z
      };
    }

    function renderPromptMesh() {
      if (!promptCanvas) {
        return;
      }
      resizePromptCanvas();
      const context = promptCanvas.getContext("2d");
      const width = promptCanvas.width;
      const height = promptCanvas.height;
      context.clearRect(0, 0, width, height);
      if (!state.promptMesh || state.promptMesh.vertices.length === 0) {
        if (promptEmpty) {
          promptEmpty.style.display = "flex";
        }
        return;
      }
      if (promptEmpty) {
        promptEmpty.style.display = "none";
      }
      const basis = promptCameraBasis();
      const focal = Math.min(width, height) * 0.95;
      const light = normalizeVec([0.35, 0.65, 0.75]);
      const facesToDraw = [];
      const vertices = state.promptMesh.vertices;
      const faces = state.promptMesh.faces;
      const normals = state.promptMesh.normals;
      for (let index = 0; index < faces.length; index += 1) {
        const face = faces[index];
        const p0 = projectPromptPoint(vertices[face[0]], basis, width, height, focal);
        const p1 = projectPromptPoint(vertices[face[1]], basis, width, height, focal);
        const p2 = projectPromptPoint(vertices[face[2]], basis, width, height, focal);
        if (!p0 || !p1 || !p2) {
          continue;
        }
        const normal = normals[index] || [0, 0, 1];
        const shade = Math.max(0.18, Math.min(1.0, 0.45 + 0.55 * Math.abs(dotVec(normal, light))));
        facesToDraw.push({
          p0,
          p1,
          p2,
          depth: (p0.z + p1.z + p2.z) / 3,
          shade
        });
      }
      facesToDraw.sort((a, b) => b.depth - a.depth);
      context.lineWidth = Math.max(1, window.devicePixelRatio);
      for (const face of facesToDraw) {
        const tone = Math.round(185 * face.shade);
        context.beginPath();
        context.moveTo(face.p0.x, face.p0.y);
        context.lineTo(face.p1.x, face.p1.y);
        context.lineTo(face.p2.x, face.p2.y);
        context.closePath();
        context.fillStyle = `rgb(${tone}, ${Math.round(tone * 1.04)}, ${Math.round(tone * 1.12)})`;
        context.fill();
        context.strokeStyle = "rgba(15, 23, 42, 0.12)";
        context.stroke();
      }
      drawPromptMarkers(context, basis, width, height, focal);
    }

    function drawPromptMarkers(context, basis, width, height, focal) {
      Object.entries(state.prompts).forEach(([linkId, prompt]) => {
        const projected = projectPromptPoint(prompt.point, basis, width, height, focal);
        if (!projected) {
          return;
        }
        const link = getLink(Number(linkId));
        const color = link ? link.color : "#22c55e";
        const radius = Number(linkId) === state.selectedNode ? 7 : 5;
        context.beginPath();
        context.arc(projected.x, projected.y, radius * window.devicePixelRatio, 0, Math.PI * 2);
        context.fillStyle = color;
        context.fill();
        context.lineWidth = 2 * window.devicePixelRatio;
        context.strokeStyle = "#111827";
        context.stroke();
      });
    }

    function rayTriangleIntersection(origin, direction, v0, v1, v2) {
      const edge1 = subVec(v1, v0);
      const edge2 = subVec(v2, v0);
      const h = crossVec(direction, edge2);
      const a = dotVec(edge1, h);
      if (Math.abs(a) < 1e-8) {
        return null;
      }
      const f = 1 / a;
      const s = subVec(origin, v0);
      const u = f * dotVec(s, h);
      if (u < 0 || u > 1) {
        return null;
      }
      const q = crossVec(s, edge1);
      const v = f * dotVec(direction, q);
      if (v < 0 || u + v > 1) {
        return null;
      }
      const t = f * dotVec(edge2, q);
      if (t <= 1e-6) {
        return null;
      }
      return { t, u, v };
    }

    function pickPromptPoint(event) {
      if (!state.promptMesh || state.selectedNode === null) {
        updatePromptStatus(state.promptMesh ? "Select a link first." : "Upload a mesh first.");
        return;
      }
      const rect = promptCanvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) * window.devicePixelRatio;
      const y = (event.clientY - rect.top) * window.devicePixelRatio;
      const width = promptCanvas.width;
      const height = promptCanvas.height;
      const focal = Math.min(width, height) * 0.95;
      const basis = promptCameraBasis();
      const rayX = (x - width * 0.5) / focal;
      const rayY = -(y - height * 0.5) / focal;
      const direction = normalizeVec(addVec(addVec(basis.forward, scaleVec(basis.right, rayX)), scaleVec(basis.up, rayY)));
      let best = null;
      const vertices = state.promptMesh.vertices;
      const faces = state.promptMesh.faces;
      const normals = state.promptMesh.normals;
      for (let index = 0; index < faces.length; index += 1) {
        const face = faces[index];
        const hit = rayTriangleIntersection(
          basis.eye,
          direction,
          vertices[face[0]],
          vertices[face[1]],
          vertices[face[2]]
        );
        if (!hit || (best && hit.t >= best.t)) {
          continue;
        }
        best = { ...hit, faceIndex: index, face };
      }
      if (!best) {
        updatePromptStatus("No mesh surface was hit. Rotate or zoom and try again.");
        return;
      }
      const point = addVec(basis.eye, scaleVec(direction, best.t));
      let normal = normals[best.faceIndex] || [0, 0, 1];
      state.prompts[state.selectedNode] = {
        point: roundVector(point),
        normal: roundVector(normalizeVec(normal))
      };
      syncPrompts();
      updatePromptStatus(`Prompt set for ${selectedLinkName()}.`);
      render();
      renderPromptMesh();
    }

    function setStatus(message) {
      status.textContent = message;
    }

    function nodeCenter(link) {
      return { x: link.x + NODE_WIDTH * 0.5, y: link.y + NODE_HEIGHT * 0.5 };
    }

    function getLink(id) {
      return state.links.find((link) => link.id === id);
    }

    function edgePath(parent, child) {
      const start = nodeCenter(parent);
      const end = nodeCenter(child);
      const bend = Math.max(48, Math.abs(end.y - start.y) * 0.5);
      return `M ${start.x} ${start.y} C ${start.x + bend} ${start.y}, ${end.x - bend} ${end.y}, ${end.x} ${end.y}`;
    }

    function selectJoint(index) {
      const joint = state.joints[index];
      if (!joint) {
        return;
      }
      state.selectedEdge = index;
      state.selectedNode = null;
      state.pendingParent = null;
      state.connectMode = false;
      setStatus(`${joint.type} joint selected. Use the joint dropdown to change its type.`);
      render();
    }

    function deleteJoint(index) {
      const removed = state.joints.splice(index, 1)[0];
      if (!removed) {
        return;
      }
      state.selectedEdge = null;
      state.selectedNode = null;
      state.pendingParent = null;
      setStatus(`Deleted ${removed.type} joint`);
      render();
    }

    function updateJointType(index, type) {
      const joint = state.joints[index];
      const nextType = String(type || "").toLowerCase();
      if (!joint || !["revolute", "prismatic"].includes(nextType)) {
        return;
      }
      joint.type = nextType;
      state.selectedEdge = index;
      state.selectedNode = null;
      state.pendingParent = null;
      state.connectMode = false;
      setStatus(`Changed joint ${joint.parent} to ${joint.child} to ${nextType}.`);
      render();
    }

    function renderEdges() {
      edgeLayer.innerHTML = `
        <defs>
          <marker id="kin-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="var(--kin-edge)"></path>
          </marker>
          <marker id="kin-arrow-selected" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="var(--kin-edge-selected)"></path>
          </marker>
        </defs>
      `;
      jointLayer.innerHTML = "";
      state.joints.forEach((joint, index) => {
        const parent = getLink(joint.parent);
        const child = getLink(joint.child);
        if (!parent || !child) {
          return;
        }
        const selected = state.selectedEdge === index;
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", edgePath(parent, child));
        path.setAttribute("class", `kin-tree-edge${selected ? " selected" : ""}`);
        path.setAttribute("marker-end", selected ? "url(#kin-arrow-selected)" : "url(#kin-arrow)");
        path.addEventListener("click", (event) => {
          event.stopPropagation();
          selectJoint(index);
        });
        edgeLayer.appendChild(path);

        const start = nodeCenter(parent);
        const end = nodeCenter(child);
        const midX = (start.x + end.x) * 0.5;
        const midY = (start.y + end.y) * 0.5;
        const chip = document.createElement("div");
        chip.className = `kin-joint-chip${selected ? " selected" : ""}`;
        chip.style.left = `${midX}px`;
        chip.style.top = `${midY}px`;
        chip.title = `Joint ${joint.parent} to ${joint.child}`;
        chip.innerHTML = `
          <select class="kin-joint-motion" aria-label="Joint ${joint.parent} to ${joint.child} motion type">
            <option value="revolute">Revolute</option>
            <option value="prismatic">Prismatic</option>
          </select>
          <button class="kin-joint-delete" title="Delete joint" type="button">×</button>
        `;
        chip.addEventListener("click", (event) => {
          event.stopPropagation();
          selectJoint(index);
        });
        const chipSelect = chip.querySelector(".kin-joint-motion");
        chipSelect.value = joint.type;
        chipSelect.addEventListener("pointerdown", (event) => {
          event.stopPropagation();
        });
        chipSelect.addEventListener("click", (event) => {
          event.stopPropagation();
        });
        chipSelect.addEventListener("change", (event) => {
          event.stopPropagation();
          updateJointType(index, chipSelect.value);
        });
        const chipDelete = chip.querySelector(".kin-joint-delete");
        chipDelete.addEventListener("pointerdown", (event) => {
          event.stopPropagation();
        });
        chipDelete.addEventListener("click", (event) => {
          event.stopPropagation();
          deleteJoint(index);
        });
        jointLayer.appendChild(chip);
      });
    }

    function render() {
      addJointButton.classList.toggle("active", state.connectMode);
      renderEdges();

      nodeLayer.innerHTML = "";
      state.links.forEach((link) => {
        const node = document.createElement("div");
        node.className = "kin-node";
      if (state.selectedNode === link.id) {
          node.classList.add("selected");
        }
        if (state.pendingParent === link.id) {
          node.classList.add("pending-parent");
        }
        if (state.prompts[link.id] !== undefined) {
          node.classList.add("has-prompt");
        }
        node.style.left = `${link.x}px`;
        node.style.top = `${link.y}px`;
        node.style.background = link.color;
        node.dataset.nodeId = String(link.id);
        node.innerHTML = `
          <div class="kin-node-row">
            <input value="${escapeHtml(link.name)}" aria-label="Link name">
            <button class="kin-node-delete" title="Delete link" type="button">×</button>
          </div>
        `;
        node.querySelector(".kin-node-delete").addEventListener("click", (event) => {
          event.stopPropagation();
          deleteNode(link.id);
        });
        const input = node.querySelector("input");
        input.addEventListener("focus", () => {
          state.selectedNode = link.id;
          state.selectedEdge = null;
          state.pendingParent = null;
          state.connectMode = false;
          setStatus(`Link ${link.id} selected`);
          updatePromptStatus();
        });
        input.addEventListener("input", () => {
          link.name = input.value;
          syncTree();
          updatePromptStatus();
        });
        node.addEventListener("click", (event) => {
          if (event.target.tagName === "INPUT" || event.target.tagName === "BUTTON") {
            return;
          }
          handleNodeClick(link.id);
        });
        node.addEventListener("pointerdown", (event) => {
          if (event.target.tagName === "INPUT" || event.target.tagName === "BUTTON") {
            return;
          }
          state.dragging = {
            id: link.id,
            startX: event.clientX,
            startY: event.clientY,
            nodeX: link.x,
            nodeY: link.y,
            moved: false
          };
          node.classList.add("dragging");
          node.setPointerCapture(event.pointerId);
        });
        node.addEventListener("pointermove", (event) => {
          if (!state.dragging || state.dragging.id !== link.id) {
            return;
          }
          const dx = event.clientX - state.dragging.startX;
          const dy = event.clientY - state.dragging.startY;
          if (Math.abs(dx) + Math.abs(dy) > 3) {
            state.dragging.moved = true;
          }
          const maxX = Math.max(0, canvas.clientWidth - NODE_WIDTH - 10);
          const maxY = Math.max(0, canvas.clientHeight - NODE_HEIGHT - 10);
          link.x = Math.min(maxX, Math.max(0, state.dragging.nodeX + dx));
          link.y = Math.min(maxY, Math.max(0, state.dragging.nodeY + dy));
          node.style.left = `${link.x}px`;
          node.style.top = `${link.y}px`;
          renderEdges();
          syncTree();
        });
        node.addEventListener("pointerup", () => {
          if (state.dragging && state.dragging.moved) {
            state.suppressClick = true;
            window.setTimeout(() => {
              state.suppressClick = false;
            }, 0);
          }
          state.dragging = null;
          node.classList.remove("dragging");
        });
        nodeLayer.appendChild(node);
      });
      syncTree();
      updatePromptStatus();
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll('"', "&quot;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function handleNodeClick(nodeId) {
      if (state.suppressClick) {
        return;
      }
      state.selectedEdge = null;
      if (!state.connectMode) {
        state.selectedNode = nodeId;
        state.pendingParent = null;
        setStatus(`Link ${nodeId} selected`);
        updatePromptStatus();
        render();
        return;
      }
      if (state.pendingParent === null) {
        state.pendingParent = nodeId;
        state.selectedNode = nodeId;
        setStatus(`Parent Link ${nodeId} selected. Click a child link.`);
        render();
        return;
      }
      const parent = state.pendingParent;
      const child = nodeId;
      const result = addJoint(parent, child, "revolute");
      state.pendingParent = null;
      state.selectedNode = null;
      state.connectMode = false;
      setStatus(result);
      render();
    }

    function addNode() {
      const id = state.links.length;
      const position = defaultNodePosition(id);
      state.links.push({
        ...position,
        id,
        name: id === 0 ? "base" : `link_${id}`,
        color: palette[id % palette.length]
      });
      state.selectedNode = id;
      state.selectedEdge = null;
      setStatus(`Added Link ${id}`);
      updatePromptStatus();
      render();
    }

    function deleteNode(nodeId) {
      const oldLinks = state.links.filter((link) => link.id !== nodeId);
      const idMap = new Map();
      oldLinks.forEach((link, index) => {
        idMap.set(link.id, index);
        link.id = index;
        link.color = palette[index % palette.length];
      });
      state.links = oldLinks;
      const remappedPrompts = {};
      Object.entries(state.prompts).forEach(([rawLinkId, prompt]) => {
        const nextId = idMap.get(Number(rawLinkId));
        if (nextId !== undefined) {
          remappedPrompts[nextId] = prompt;
        }
      });
      state.prompts = remappedPrompts;
      state.joints = state.joints
        .filter((joint) => joint.parent !== nodeId && joint.child !== nodeId)
        .map((joint) => ({
          parent: idMap.get(joint.parent),
          child: idMap.get(joint.child),
          type: joint.type
        }))
        .filter((joint) => joint.parent !== undefined && joint.child !== undefined);
      state.selectedNode = null;
      state.selectedEdge = null;
      state.pendingParent = null;
      setStatus(`Deleted Link ${nodeId}`);
      syncPrompts();
      updatePromptStatus();
      render();
    }

    function addJoint(parent, child, type) {
      if (parent === child) {
        return "A joint cannot connect a link to itself.";
      }
      if (state.joints.some((joint) => joint.parent === parent && joint.child === child)) {
        return "That joint already exists.";
      }
      if (state.joints.some((joint) => joint.child === child)) {
        return `Link ${child} already has a parent.`;
      }
      if (wouldCreateCycle(parent, child)) {
        return "That joint would create a cycle.";
      }
      state.joints.push({ parent, child, type });
      return `Added ${type} joint: Link ${parent} to Link ${child}`;
    }

    function wouldCreateCycle(parent, child) {
      const stack = [child];
      const seen = new Set();
      while (stack.length > 0) {
        const current = stack.pop();
        if (current === parent) {
          return true;
        }
        if (seen.has(current)) {
          continue;
        }
        seen.add(current);
        state.joints
          .filter((joint) => joint.parent === current)
          .forEach((joint) => stack.push(joint.child));
      }
      return false;
    }

    addNodeButton.addEventListener("click", addNode);
    addJointButton.addEventListener("click", () => {
      state.connectMode = !state.connectMode;
      state.pendingParent = null;
      state.selectedEdge = null;
      setStatus(state.connectMode ? "Creating a revolute joint. Click a parent link, then a child link." : "Joint creation cancelled.");
      render();
    });
    deleteButton.addEventListener("click", () => {
      if (state.selectedNode !== null) {
        deleteNode(state.selectedNode);
        return;
      }
      if (state.selectedEdge !== null) {
        deleteJoint(state.selectedEdge);
        return;
      }
      setStatus("Select a link or joint to delete.");
    });
    resetButton.addEventListener("click", () => {
      loadTree(defaultTree);
      setStatus("Reset to the default tree.");
    });
    if (promptCanvas) {
      promptCanvas.addEventListener("pointerdown", (event) => {
        state.promptDrag = {
          startX: event.clientX,
          startY: event.clientY,
          yaw: state.promptCamera.yaw,
          pitch: state.promptCamera.pitch,
          moved: false
        };
        promptCanvas.setPointerCapture(event.pointerId);
      });
      promptCanvas.addEventListener("pointermove", (event) => {
        if (!state.promptDrag) {
          return;
        }
        const dx = event.clientX - state.promptDrag.startX;
        const dy = event.clientY - state.promptDrag.startY;
        if (Math.abs(dx) + Math.abs(dy) > 3) {
          state.promptDrag.moved = true;
        }
        if (state.promptDrag.moved) {
          state.promptCamera.yaw = state.promptDrag.yaw + dx * 0.01;
          state.promptCamera.pitch = state.promptDrag.pitch + dy * 0.01;
          renderPromptMesh();
        }
      });
      promptCanvas.addEventListener("pointerup", (event) => {
        const drag = state.promptDrag;
        state.promptDrag = null;
        if (drag && !drag.moved) {
          pickPromptPoint(event);
        }
      });
      promptCanvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        const factor = Math.exp(event.deltaY * 0.0012);
        const minDistance = Math.max(1e-5, state.promptCamera.radius * 0.35);
        const maxDistance = Math.max(1, state.promptCamera.radius * 8);
        state.promptCamera.distance = Math.max(
          minDistance,
          Math.min(maxDistance, state.promptCamera.distance * factor)
        );
        renderPromptMesh();
      }, { passive: false });
    }
    if (clearPromptButton) {
      clearPromptButton.addEventListener("click", () => {
        if (state.selectedNode === null) {
          updatePromptStatus("Select a link before clearing its prompt.");
          return;
        }
        delete state.prompts[state.selectedNode];
        syncPrompts();
        updatePromptStatus();
        render();
        renderPromptMesh();
      });
    }
    if (clearAllPromptsButton) {
      clearAllPromptsButton.addEventListener("click", () => {
        state.prompts = {};
        syncPrompts();
        updatePromptStatus("Cleared all point prompts.");
        render();
        renderPromptMesh();
      });
    }
    if (resetPromptViewButton) {
      resetPromptViewButton.addEventListener("click", resetPromptCamera);
    }
    if (promptMeshBox) {
      promptMeshBox.addEventListener("input", loadPromptMeshFromBox);
      promptMeshBox.addEventListener("change", loadPromptMeshFromBox);
    }
    syncBox.addEventListener("input", loadExternalTreeFromBox);
    syncBox.addEventListener("change", loadExternalTreeFromBox);
    if (promptSyncBox) {
      promptSyncBox.addEventListener("input", loadExternalPromptsFromBox);
      promptSyncBox.addEventListener("change", loadExternalPromptsFromBox);
    }
    canvas.addEventListener("click", (event) => {
      if (event.target === canvas || event.target === edgeLayer || event.target === nodeLayer) {
        state.selectedNode = null;
        state.selectedEdge = null;
        state.pendingParent = null;
        updatePromptStatus();
        render();
      }
    });
    window.addEventListener("resize", () => {
      render();
      renderPromptMesh();
    });
    window.setInterval(() => {
      loadPromptMeshFromBox();
      loadExternalTreeFromBox();
      loadExternalPromptsFromBox();
    }, 900);
    loadTree(defaultTree);
    loadPromptMeshFromBox();
    loadExternalTreeFromBox();
    loadExternalPromptsFromBox();
    renderPromptMesh();
    syncPrompts();
    setStatus("Drag links to arrange the tree.");
  }

  waitForEditor();
}
"""


def _extract_gradio_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw_path = value.get("path") or value.get("name")
    else:
        raw_path = value
    if raw_path is None:
        return None
    return Path(str(raw_path)).expanduser().resolve()


def _tree_to_pretty_json(tree: dict[str, Any]) -> str:
    return json.dumps(tree, indent=2)


def _load_json_object(raw_value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Kinematic tree must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Kinematic tree JSON must be an object.")
    return parsed


def _link_identifier(link: Any, fallback_id: int) -> tuple[int, str]:
    if isinstance(link, str):
        return fallback_id, link.strip()
    if isinstance(link, dict):
        raw_id = link.get("id", fallback_id)
        raw_name = link.get("name", f"link_{raw_id}")
        return int(raw_id), str(raw_name).strip()
    raise ValueError(f"Links must be strings or objects, got {type(link).__name__}.")


def _resolve_link_ref(raw_value: Any, *, name_to_id: dict[str, int], num_links: int) -> int:
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped in name_to_id:
            return int(name_to_id[stripped])
        if stripped.lstrip("+-").isdigit():
            raw_value = int(stripped)
        else:
            raise ValueError(f"Unknown link reference {raw_value!r}.")
    link_id = int(raw_value)
    if not 0 <= link_id < num_links:
        raise ValueError(f"Link ID {link_id} is outside [0, {num_links - 1}].")
    return link_id


def _joint_type_from_record(record: dict[str, Any]) -> str:
    raw_type = (
        record.get("type")
        or record.get("joint_type")
        or record.get("motion_type")
    )
    if raw_type is None:
        if bool(record.get("is_revolute", False)):
            raw_type = "revolute"
        elif bool(record.get("is_prismatic", False)):
            raw_type = "prismatic"
    joint_type = str(raw_type or "").strip().lower()
    if joint_type not in {"revolute", "prismatic"}:
        raise ValueError(
            "Each joint must specify type 'revolute' or 'prismatic'."
        )
    return joint_type


def parse_kinematic_tree(raw_value: str) -> tuple[list[str], list[tuple[int, int, str]]]:
    payload = _load_json_object(raw_value)
    raw_links = payload.get("links", payload.get("link_names"))
    if not isinstance(raw_links, list) or not raw_links:
        raise ValueError("Kinematic tree must contain a non-empty 'links' list.")

    link_records = [_link_identifier(link, idx) for idx, link in enumerate(raw_links)]
    ids = [link_id for link_id, _ in link_records]
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("Link IDs must be dense integers starting at 0.")

    link_names_by_id = {link_id: name for link_id, name in link_records}
    link_names = [link_names_by_id[idx] for idx in range(len(link_records))]
    if any(not name for name in link_names):
        raise ValueError("Link names must be non-empty.")
    name_to_id = {name: idx for idx, name in enumerate(link_names)}

    raw_joints = payload.get("joints", [])
    if not isinstance(raw_joints, list):
        raise ValueError("'joints' must be a list.")
    joint_specs: list[tuple[int, int, str]] = []
    for joint in raw_joints:
        if isinstance(joint, (list, tuple)) and len(joint) == 3:
            parent_ref, child_ref, joint_type_ref = joint
            joint_record = {
                "parent": parent_ref,
                "child": child_ref,
                "type": joint_type_ref,
            }
        elif isinstance(joint, dict):
            joint_record = joint
        else:
            raise ValueError(
                "Each joint must be an object or [parent, child, type] list."
            )

        parent_ref = joint_record.get(
            "parent",
            joint_record.get("parent_id", joint_record.get("parent_link_id")),
        )
        child_ref = joint_record.get(
            "child",
            joint_record.get("child_id", joint_record.get("child_link_id")),
        )
        if parent_ref is None or child_ref is None:
            raise ValueError("Each joint must define parent and child links.")
        parent_id = _resolve_link_ref(
            parent_ref,
            name_to_id=name_to_id,
            num_links=len(link_names),
        )
        child_id = _resolve_link_ref(
            child_ref,
            name_to_id=name_to_id,
            num_links=len(link_names),
        )
        joint_specs.append((parent_id, child_id, _joint_type_from_record(joint_record)))

    build_joint_tensors(len(link_names), joint_specs)
    return link_names, joint_specs


def _compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"))


def _prompt_mesh_payload(mesh: Any, *, max_faces: int = 12000) -> str:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Prompt picker requires a mesh with 3D vertices.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Prompt picker requires a triangular mesh.")

    face_count = int(faces.shape[0])
    sampled = False
    if face_count > max_faces:
        face_ids = np.linspace(0, face_count - 1, int(max_faces), dtype=np.int64)
        faces = faces[face_ids]
        sampled = True

    used_vertex_ids = np.unique(faces.reshape(-1))
    remap = np.full((vertices.shape[0],), -1, dtype=np.int64)
    remap[used_vertex_ids] = np.arange(len(used_vertex_ids), dtype=np.int64)
    compact_vertices = vertices[used_vertex_ids]
    compact_faces = remap[faces].astype(np.int32, copy=False)

    triangles = compact_vertices[compact_faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    ).astype(np.float32, copy=False)
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(
        normals,
        np.maximum(normal_lengths, np.float32(1e-8)),
        out=np.zeros_like(normals),
    )

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = ((bbox_min + bbox_max) * 0.5).astype(np.float32, copy=False)
    radius = float(np.linalg.norm((bbox_max - bbox_min) * 0.5))
    if radius <= 0.0:
        radius = 1.0

    payload = {
        "vertices": np.round(compact_vertices, 6).tolist(),
        "faces": compact_faces.tolist(),
        "normals": np.round(normals, 6).tolist(),
        "center": np.round(center, 6).tolist(),
        "radius": radius,
        "source_faces": face_count,
        "display_faces": int(compact_faces.shape[0]),
        "sampled": sampled,
    }
    return _compact_json(payload)


def _up_dir_slug(up_dir: str) -> str:
    return up_dir.replace("+", "pos").replace("-", "neg")


def _upright_rendering_gallery_item() -> list[tuple[str, str]]:
    from PIL import Image, ImageDraw, ImageFont

    placeholder_path = OUTPUT_ROOT / "_ui" / "upright_orientation_rendering.png"
    if not placeholder_path.exists():
        placeholder_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (768, 384), (248, 250, 252))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 38)
            small_font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        title = "Rendering"
        subtitle = "Preparing six upright orientation previews..."
        title_box = draw.textbbox((0, 0), title, font=font)
        subtitle_box = draw.textbbox((0, 0), subtitle, font=small_font)
        draw.text(
            ((768 - (title_box[2] - title_box[0])) / 2, 150),
            title,
            fill=(17, 24, 39),
            font=font,
        )
        draw.text(
            ((768 - (subtitle_box[2] - subtitle_box[0])) / 2, 204),
            subtitle,
            fill=(75, 85, 99),
            font=small_font,
        )
        image.save(placeholder_path)
    return [(str(placeholder_path), "Rendering")]


def _upright_rendering_preview_paths() -> list[str]:
    placeholder_path = _upright_rendering_gallery_item()[0][0]
    return [placeholder_path for _ in UP_DIR_CHOICES]


def _upright_preview_paths(gallery_items: list[tuple[str, str]]) -> list[str | None]:
    paths: list[str | None] = [path for path, _caption in gallery_items]
    if len(paths) < len(UP_DIR_CHOICES):
        paths.extend([None] * (len(UP_DIR_CHOICES) - len(paths)))
    return paths[: len(UP_DIR_CHOICES)]


def _set_preview_axes_equal(ax: Any, vertices: np.ndarray) -> None:
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    radius = float(np.max(bbox_max - bbox_min) * 0.55)
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))
    ax.set_box_aspect((1, 1, 1))


def _render_up_direction_preview_image(
    mesh: Any,
    *,
    up_dir: str,
    output_path: Path,
    max_faces: int = 3000,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reoriented_mesh, _ = reorient_mesh_to_z_up(mesh, up_dir)
    normalized_mesh, _, _ = normalize_mesh(reoriented_mesh)
    vertices = np.asarray(normalized_mesh.vertices, dtype=np.float32)
    faces = np.asarray(normalized_mesh.faces, dtype=np.int64)
    if faces.shape[0] > max_faces:
        face_ids = np.linspace(0, faces.shape[0] - 1, int(max_faces), dtype=np.int64)
        faces = faces[face_ids]

    fig = plt.figure(figsize=(3.2, 3.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color="#cbd5e1",
        edgecolor="#64748b",
        linewidth=0.025,
        antialiased=True,
        shade=True,
    )
    _set_preview_axes_equal(ax, vertices)
    ax.view_init(elev=20, azim=235)
    ax.set_axis_off()
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _render_up_direction_previews(
    *,
    mesh_path: Path,
    mesh: Any,
) -> list[tuple[str, str]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    preview_dir = OUTPUT_ROOT / "up_direction_previews" / f"{mesh_path.stem}_{timestamp}"
    try:
        return _render_up_direction_previews_with_blender(
            mesh_path=mesh_path,
            output_dir=preview_dir,
        )
    except Exception:
        traceback.print_exc()

    gallery_items: list[tuple[str, str]] = []
    for up_dir in UP_DIR_CHOICES:
        output_path = preview_dir / f"up_{_up_dir_slug(up_dir)}.png"
        _render_up_direction_preview_image(
            mesh,
            up_dir=up_dir,
            output_path=output_path,
        )
        gallery_items.append((str(output_path), f"{up_dir} up"))
    return gallery_items


def _render_up_direction_previews_with_blender(
    *,
    mesh_path: Path,
    output_dir: Path,
) -> list[tuple[str, str]]:
    import subprocess

    blender_path = resolve_blender_binary(os.environ.get("BLENDER_BIN") or None)
    blender_script_path = REPO_ROOT / "scripts" / "render_upright_previews_blender.py"
    if not blender_script_path.exists():
        raise FileNotFoundError(f"Upright preview Blender script is missing: {blender_script_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_manager = tempfile.TemporaryDirectory(prefix="upright_previews_")
    render_output_dir = Path(temp_dir_manager.name)
    command = [
        str(blender_path),
        "-b",
        "-P",
        str(blender_script_path),
        "--",
        "--mesh-path",
        str(mesh_path),
        "--output-dir",
        str(render_output_dir),
        "--resolution",
        "384",
        "--up-dirs",
        ",".join(UP_DIR_CHOICES),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            if result.stdout:
                print(result.stdout)
            raise subprocess.CalledProcessError(
                result.returncode,
                command,
                output=result.stdout,
            )

        gallery_items: list[tuple[str, str]] = []
        for up_dir in UP_DIR_CHOICES:
            temp_output_path = render_output_dir / f"up_{_up_dir_slug(up_dir)}.png"
            output_path = output_dir / temp_output_path.name
            if not temp_output_path.exists() or temp_output_path.stat().st_size <= 0:
                stdout_tail = "\n".join((result.stdout or "").splitlines()[-40:])
                raise FileNotFoundError(
                    "Blender did not write upright preview: "
                    f"{temp_output_path}\nBlender output:\n{stdout_tail}"
                )
            shutil.copy2(temp_output_path, output_path)
            gallery_items.append((str(output_path), f"{up_dir} up"))
        return gallery_items
    finally:
        temp_dir_manager.cleanup()


def _load_point_prompt_payload(raw_value: str | None) -> dict[str, Any]:
    if raw_value is None or not str(raw_value).strip():
        return {"prompts": []}
    try:
        payload = json.loads(str(raw_value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Point prompt JSON must be valid JSON: {exc}") from exc
    if isinstance(payload, list):
        return {"prompts": payload}
    if not isinstance(payload, dict):
        raise ValueError("Point prompt JSON must be an object or a list.")
    return payload


def _parse_point_prompt_arrays(
    raw_value: str | None,
    *,
    num_links: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    payload = _load_point_prompt_payload(raw_value)
    raw_prompts = payload.get("prompts", [])
    if raw_prompts is None:
        raw_prompts = []
    if not isinstance(raw_prompts, list):
        raise ValueError("'prompts' must be a list.")

    points = np.zeros((num_links, 3), dtype=np.float32)
    normals = np.zeros((num_links, 3), dtype=np.float32)
    has_prompt = np.zeros((num_links,), dtype=np.bool_)

    for prompt in raw_prompts:
        if not isinstance(prompt, dict):
            raise ValueError("Each point prompt must be an object.")
        link_id = int(prompt.get("link_id", prompt.get("link", -1)))
        if not 0 <= link_id < num_links:
            raise ValueError(f"Point prompt link_id {link_id} is outside [0, {num_links - 1}].")
        point = np.asarray(prompt.get("point"), dtype=np.float32)
        normal = np.asarray(prompt.get("normal"), dtype=np.float32)
        if point.shape != (3,):
            raise ValueError(f"Point prompt for link {link_id} must have a 3D point.")
        if normal.shape != (3,):
            raise ValueError(f"Point prompt for link {link_id} must have a 3D normal.")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-8:
            raise ValueError(f"Point prompt for link {link_id} has a zero normal.")
        points[link_id] = point
        normals[link_id] = normal / np.float32(normal_norm)
        has_prompt[link_id] = True

    if not bool(has_prompt.any()):
        return None
    return points, normals, has_prompt


def _normalize_point_prompt_arrays(
    *,
    points: np.ndarray,
    normals: np.ndarray,
    mesh_geometry: Any,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(mesh_geometry.up_dir_rotation, dtype=np.float32)
    rotated_points = np.asarray(points, dtype=np.float32) @ rotation.T
    rotated_normals = np.asarray(normals, dtype=np.float32) @ rotation.T
    normal_lengths = np.linalg.norm(rotated_normals, axis=1, keepdims=True)
    rotated_normals = np.divide(
        rotated_normals,
        np.maximum(normal_lengths, np.float32(1e-8)),
        out=np.zeros_like(rotated_normals),
    )
    normalized_points = (
        (rotated_points - np.asarray(mesh_geometry.center, dtype=np.float32))
        * np.float32(mesh_geometry.scale)
    )
    return (
        normalized_points.astype(np.float32, copy=False),
        rotated_normals.astype(np.float32, copy=False),
    )


def _duplicate_link_prompt_warning(
    link_names: list[str],
    point_prompt_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> str | None:
    has_prompt = (
        np.zeros((len(link_names),), dtype=np.bool_)
        if point_prompt_arrays is None
        else np.asarray(point_prompt_arrays[2], dtype=np.bool_)
    )
    name_to_link_ids: dict[str, list[int]] = {}
    for link_id, link_name in enumerate(link_names):
        normalized_name = str(link_name).strip()
        name_to_link_ids.setdefault(normalized_name, []).append(int(link_id))

    ambiguous_groups: list[str] = []
    for link_name, link_ids in name_to_link_ids.items():
        if len(link_ids) <= 1:
            continue
        missing_prompt_ids = [link_id for link_id in link_ids if not bool(has_prompt[link_id])]
        if missing_prompt_ids:
            ambiguous_groups.append(
                f"{link_name!r} links {link_ids} missing prompts for {missing_prompt_ids}"
            )
    if not ambiguous_groups:
        return None
    return (
        "Duplicate link names need point prompts to disambiguate them. "
        "Rename the duplicate links or add point prompts for every link in each duplicate-name group: "
        + "; ".join(ambiguous_groups)
    )


def _auto_tree_from_parsed_response(parsed_response: dict[str, Any]) -> dict[str, Any]:
    return {
        "links": [
            {
                "id": int(link["link_id"]),
                "name": str(link["name"]).strip() or f"link_{int(link['link_id'])}",
            }
            for link in parsed_response["links"]
        ],
        "joints": [
            {
                "parent": int(joint["parent_link_id"]),
                "child": int(joint["child_link_id"]),
                "type": str(joint["joint_type"]).strip().lower(),
            }
            for joint in parsed_response["joints"]
        ],
    }


def _point_prompt_json_from_normalized_prompts(
    *,
    normalized_points: np.ndarray,
    normalized_normals: np.ndarray,
    mesh_geometry: Any,
) -> str:
    rotation = np.asarray(mesh_geometry.up_dir_rotation, dtype=np.float32)
    points = np.asarray(normalized_points, dtype=np.float32)
    normals = np.asarray(normalized_normals, dtype=np.float32)
    model_points = points / np.float32(mesh_geometry.scale) + np.asarray(
        mesh_geometry.center,
        dtype=np.float32,
    )
    raw_points = model_points @ rotation
    raw_normals = normals @ rotation
    normal_lengths = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    raw_normals = np.divide(
        raw_normals,
        np.maximum(normal_lengths, np.float32(1e-8)),
        out=np.zeros_like(raw_normals),
    )
    prompts = [
        {
            "link_id": int(link_id),
            "point": np.round(raw_points[link_id], 6).astype(float).tolist(),
            "normal": np.round(raw_normals[link_id], 6).astype(float).tolist(),
        }
        for link_id in range(int(points.shape[0]))
    ]
    return json.dumps({"prompts": prompts}, indent=2)


def _auto_render_gallery_items(
    rendered_views: list[Any],
    overlay_paths: list[Path] | None = None,
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = [
        (
            str(view.image_path),
            f"Render {int(view.image_id)} ({float(view.azimuth_deg):.0f} deg)",
        )
        for view in rendered_views
    ]
    for overlay_path in overlay_paths or []:
        items.append((str(overlay_path), f"Prompt overlay {overlay_path.stem}"))
    return items

def _make_inference_args(
    *,
    output_dir: Path,
    num_query_points: int,
    num_query_points_per_face_for_seg: int | None,
    query_batch_size: int,
    animation_frames: int,
    strict_face_postprocess: bool,
    enforce_connectivity_per_part: bool,
    joint_decoding_confidence_temperature: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=output_dir,
        num_query_points=int(num_query_points),
        num_query_points_per_face_for_seg=num_query_points_per_face_for_seg,
        query_batch_size=int(query_batch_size),
        animation_frames=int(animation_frames),
        strict_face_postprocess=bool(strict_face_postprocess),
        enforce_connectivity_per_part=bool(enforce_connectivity_per_part),
        no_point_prompt=True,
        joint_decoding_confidence_temperature=float(
            joint_decoding_confidence_temperature
        ),
        export_urdf=False,
    )


def _zip_directory(directory: Path) -> Path:
    zip_path = directory.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zip_file.write(path, path.relative_to(directory))
    return zip_path


def _ensure_instruct_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Install huggingface_hub or "
            f"provide the checkpoint locally."
        ) from exc

    downloaded_path = hf_hub_download(
        repo_id=CHECKPOINT_REPO_ID,
        filename=CHECKPOINT_REPO_FILENAME,
        local_dir=str(checkpoint_path.parent),
        local_dir_use_symlinks=False,
    )
    downloaded_path = Path(downloaded_path)
    if downloaded_path != checkpoint_path:
        shutil.copy2(downloaded_path, checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not download checkpoint to {checkpoint_path}")
    return checkpoint_path


def _spaces_gpu(fn):
    if spaces is None:
        return fn
    duration = min(int(os.environ.get("SPACES_GPU_DURATION", "120")), 120)
    return spaces.GPU(duration=duration)(fn)


class InstructParticulateApp:
    def __init__(
        self,
        *,
        run_dir: Path = RUN_DIR,
        checkpoint_path: Path = CHECKPOINT_PATH,
        output_root: Path = OUTPUT_ROOT,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.checkpoint_path = checkpoint_path.resolve()
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = _ensure_instruct_checkpoint(self.checkpoint_path)
        self.config = load_run_config(self.run_dir)
        configure_runtime_environment(self.config)
        self.num_shape_points, self.default_num_query_points, self.sharp_point_ratio = (
            resolve_inference_sampling_config(self.config)
        )
        self.device: torch.device | None = None
        self.model: Particulate2ArticulationModel | None = None
        self._model_lock = threading.Lock()

    def _ensure_model_loaded(self) -> tuple[Particulate2ArticulationModel, torch.device]:
        if self.model is not None and self.device is not None:
            return self.model, self.device
        with self._model_lock:
            if self.model is not None and self.device is not None:
                return self.model, self.device
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this demo but is not available.")
            device = torch.device("cuda")
            model = Particulate2ArticulationModel(**self.config["model"])
            load_model_checkpoint_for_inference(
                model,
                self.checkpoint_path,
                device=device,
            )
            if model.encoder.use_text_conditioning:
                model.encoder.compute_link_text_embeddings_on_the_fly = True
            self.model = model
            self.device = device
            return model, device

    def register_mesh(
        self,
        mesh_value: Any,
    ):
        empty_orientation_previews = [None] * len(UP_DIR_CHOICES)
        mesh_path = _extract_gradio_path(mesh_value)
        if mesh_path is None:
            yield (
                None,
                "Upload a mesh to run inference.",
                "",
                *empty_orientation_previews,
                "",
            )
            return
        try:
            mesh = load_trimesh(mesh_path)
        except Exception as exc:
            yield (
                None,
                f"Could not load mesh: {exc}",
                "",
                *empty_orientation_previews,
                "",
            )
            return

        rendering_orientation_previews = _upright_rendering_preview_paths()
        yield (
            str(mesh_path),
            f"Loaded {mesh_path.name}: {len(mesh.faces)} faces. Rendering upright orientation previews...",
            "",
            *rendering_orientation_previews,
            "",
        )

        try:
            prompt_mesh_data = _prompt_mesh_payload(mesh)
        except Exception as exc:
            yield (
                str(mesh_path),
                f"Loaded {mesh_path.name}: {len(mesh.faces)} faces. Could not prepare point picker mesh: {exc}",
                "",
                *rendering_orientation_previews,
                "",
            )
            return

        yield (
            str(mesh_path),
            f"Loaded {mesh_path.name}: {len(mesh.faces)} faces. Point picker ready. Rendering upright orientation previews...",
            prompt_mesh_data,
            *rendering_orientation_previews,
            "",
        )

        try:
            upright_previews = _render_up_direction_previews(
                mesh_path=mesh_path,
                mesh=mesh,
            )
        except Exception as exc:
            traceback.print_exc()
            yield (
                str(mesh_path),
                f"Loaded {mesh_path.name}: {len(mesh.faces)} faces. Upright-view previews could not be rendered: {exc}",
                prompt_mesh_data,
                *empty_orientation_previews,
                "",
            )
            return

        yield (
            str(mesh_path),
            f"Loaded {mesh_path.name}: {len(mesh.faces)} faces. Select the upright orientation before inference.",
            prompt_mesh_data,
            *_upright_preview_paths(upright_previews),
            "",
        )

    def extract_kinematic_structure(
        self,
        mesh_path_value: Any,
        current_kinematic_tree_json: str,
        current_point_prompt_json: str,
        up_dir: str,
    ):
        current_tree = current_kinematic_tree_json or _tree_to_pretty_json(
            DEFAULT_KINEMATIC_TREE
        )
        current_prompts = current_point_prompt_json or '{"prompts":[]}'
        mesh_path = _extract_gradio_path(mesh_path_value)
        if mesh_path is None:
            yield current_tree, current_prompts, [], "Upload a mesh first."
            return
        if not mesh_path.exists():
            yield current_tree, current_prompts, [], f"Mesh file does not exist: {mesh_path}"
            return
        if not up_dir:
            yield (
                current_tree,
                current_prompts,
                [],
                "Select the upright orientation image before extracting the kinematic structure.",
            )
            return

        gallery_items: list[tuple[str, str]] = []
        try:
            canonical_up = canonicalize_up_dir(up_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_root / f"{mesh_path.stem}_auto_{timestamp}"
            auto_output_dir = output_dir / "auto_kinematics"
            renders_dir = auto_output_dir / "renders"
            auto_output_dir.mkdir(parents=True, exist_ok=True)

            mesh_geometry = self._prepare_geometry(mesh_path, canonical_up)

            yield (
                current_tree,
                current_prompts,
                gallery_items,
                "Rendering",
            )
            rendered_views = render_mesh_auto_kinematics_views(
                mesh_geometry.render_mesh,
                output_dir=renders_dir,
                azimuths_deg=DEFAULT_AUTO_KINEMATICS_AZIMUTHS,
                blender_bin=os.environ.get("BLENDER_BIN") or None,
            )
            gallery_items = _auto_render_gallery_items(rendered_views)
            yield (
                current_tree,
                current_prompts,
                gallery_items,
                "Identifying and localizing kinematic parts",
            )

            raw_response = call_auto_kinematics_model(
                model_id=AUTO_KINEMATICS_MODEL_ID,
                system_prompt_path=REPO_ROOT / "kinematic-inference-system-prompt.md",
                rendered_views=rendered_views,
                reasoning_effort=AUTO_KINEMATICS_REASONING_EFFORT,
            )
            parsed_response = parse_auto_kinematics_response(raw_response)
            yield (
                current_tree,
                current_prompts,
                gallery_items,
                "Picking 3D point prompts",
            )

            lifted = lift_point_prompts_from_rendered_views(
                links=parsed_response["links"],
                rendered_views=rendered_views,
            )
            prepared_prompts = prepare_lifted_prompt_records_for_saving(
                normalized_mesh=mesh_geometry.normalized_mesh,
                lifted=lifted,
                center=np.asarray(mesh_geometry.center, dtype=np.float32),
                scale=float(mesh_geometry.scale),
                render_to_model_rotation=mesh_geometry.render_to_model_rotation,
            )
            save_auto_kinematics_artifacts(
                output_dir=auto_output_dir,
                raw_response=raw_response,
                parsed_response=parsed_response,
                lifted_prompt_records=prepared_prompts["records"],
                center=np.asarray(mesh_geometry.center, dtype=np.float32),
                scale=float(mesh_geometry.scale),
            )
            overlay_paths = save_2d_point_prompt_visualizations(
                rendered_views=rendered_views,
                lifted_prompt_records=prepared_prompts["records"],
                output_dir=auto_output_dir / "point_prompt_overlays",
            )
            gallery_items = _auto_render_gallery_items(rendered_views, overlay_paths)

            extracted_tree = _auto_tree_from_parsed_response(parsed_response)
            extracted_tree_json = _tree_to_pretty_json(extracted_tree)
            extracted_prompt_json = _point_prompt_json_from_normalized_prompts(
                normalized_points=np.asarray(
                    prepared_prompts["point_prompts"],
                    dtype=np.float32,
                ),
                normalized_normals=np.asarray(
                    prepared_prompts["point_prompt_normals"],
                    dtype=np.float32,
                ),
                mesh_geometry=mesh_geometry,
            )
            (auto_output_dir / "demo_kinematic_tree.json").write_text(
                extracted_tree_json + "\n",
                encoding="utf-8",
            )
            (auto_output_dir / "demo_point_prompts.json").write_text(
                extracted_prompt_json + "\n",
                encoding="utf-8",
            )

            yield (
                extracted_tree_json,
                extracted_prompt_json,
                gallery_items,
                "Success",
            )
        except Exception as exc:
            traceback.print_exc()
            yield (
                current_tree,
                current_prompts,
                gallery_items,
                f"Auto-kinematic extraction error: {exc}",
            )
        finally:
            torch.cuda.empty_cache()

    def predict(
        self,
        mesh_path_value: Any,
        kinematic_tree_json: str,
        point_prompt_json: str,
        up_dir: str,
        num_query_points: int,
        num_query_points_per_face_for_seg: int,
        query_batch_size: int,
        animation_frames: int,
        strict_face_postprocess: bool,
        enforce_connectivity_per_part: bool,
        joint_decoding_confidence_temperature: float,
    ):
        mesh_path = _extract_gradio_path(mesh_path_value)
        if mesh_path is None:
            yield None, None, None, None, None, "Upload a mesh first."
            return
        if not mesh_path.exists():
            yield None, None, None, None, None, f"Mesh file does not exist: {mesh_path}"
            return
        if not up_dir:
            yield (
                None,
                None,
                None,
                None,
                None,
                "Select the upright orientation image before running inference.",
            )
            return

        early_visualization_path: Path | None = None
        try:
            link_names, joint_specs = parse_kinematic_tree(kinematic_tree_json)
            point_prompt_arrays = _parse_point_prompt_arrays(
                point_prompt_json,
                num_links=len(link_names),
            )
            duplicate_prompt_warning = _duplicate_link_prompt_warning(
                link_names,
                point_prompt_arrays,
            )
            if duplicate_prompt_warning is not None:
                yield None, None, None, None, None, duplicate_prompt_warning
                return
            canonical_up = canonicalize_up_dir(up_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_root / f"{mesh_path.stem}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            args = _make_inference_args(
                output_dir=output_dir,
                num_query_points=int(num_query_points),
                num_query_points_per_face_for_seg=(
                    None
                    if int(num_query_points_per_face_for_seg) <= 0
                    else int(num_query_points_per_face_for_seg)
                ),
                query_batch_size=int(query_batch_size),
                animation_frames=int(animation_frames),
                strict_face_postprocess=bool(strict_face_postprocess),
                enforce_connectivity_per_part=bool(enforce_connectivity_per_part),
                joint_decoding_confidence_temperature=float(
                    joint_decoding_confidence_temperature
                ),
            )

            model, device = self._ensure_model_loaded()
            mesh_geometry = self._prepare_geometry(mesh_path, canonical_up)
            segmentation_num_query_points = self._segmentation_num_query_points(
                args,
                mesh_geometry.normalized_mesh,
            )
            if point_prompt_arrays is None:
                batch, query_face_indices = prepare_inference_batch_from_mesh(
                    mesh_geometry.normalized_mesh,
                    num_shape_points=self.num_shape_points,
                    num_query_points=segmentation_num_query_points,
                    num_query_points_per_face_for_seg=args.num_query_points_per_face_for_seg,
                    sharp_point_ratio=self.sharp_point_ratio,
                    link_names=link_names,
                    joint_specs=joint_specs,
                    device=device,
                )
            else:
                raw_prompt_points, raw_prompt_normals, has_point_prompt = point_prompt_arrays
                normalized_prompt_points, normalized_prompt_normals = _normalize_point_prompt_arrays(
                    points=raw_prompt_points,
                    normals=raw_prompt_normals,
                    mesh_geometry=mesh_geometry,
                )
                batch, query_face_indices = prepare_inference_batch_from_mesh_with_prompts(
                    mesh_geometry.normalized_mesh,
                    num_shape_points=self.num_shape_points,
                    num_query_points=segmentation_num_query_points,
                    num_query_points_per_face_for_seg=args.num_query_points_per_face_for_seg,
                    sharp_point_ratio=self.sharp_point_ratio,
                    link_names=link_names,
                    joint_specs=joint_specs,
                    device=device,
                    link_point_prompts=torch.from_numpy(normalized_prompt_points).float(),
                    link_point_prompt_normals=torch.from_numpy(normalized_prompt_normals).float(),
                    require_unique_link_names=False,
                )
                no_prompt_mask = torch.from_numpy(~has_point_prompt).bool().unsqueeze(0).to(
                    device
                )
                batch["link_point_prompt_dropout_eligible"] = no_prompt_mask

            output = run_batched_model_inference(
                model,
                query_batch_size=int(args.query_batch_size),
                no_point_prompt_for_unique_text=bool(args.no_point_prompt),
                decode_joint_parameters=False,
                **batch,
            )
            point_part_probabilities = (
                F.softmax(output["segmentation_logits"], dim=-1)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            point_part_ids = (
                output["segmentation_logits"]
                .argmax(dim=-1)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32)
            )
            query_points = _tensor_to_numpy(batch["query_points"][0], dtype=np.float32)
            query_normals = _tensor_to_numpy(
                batch["query_point_normals"][0],
                dtype=np.float32,
            )
            links_for_query_visualization = [
                {"link_id": int(link_id), "name": str(link_name)}
                for link_id, link_name in enumerate(link_names)
            ]
            visualization_link_point_prompts_world, visualization_link_point_prompt_ids = (
                _resolve_visualized_batch_link_point_prompts(
                    args=args,
                    batch=batch,
                    links=links_for_query_visualization,
                    center=mesh_geometry.center,
                    scale=mesh_geometry.scale,
                )
            )
            early_visualization_path = save_predicted_point_query_rest_visualization(
                output_dir,
                query_points=_denormalize_points(
                    query_points,
                    center=mesh_geometry.center,
                    scale=mesh_geometry.scale,
                ),
                query_normals=query_normals,
                predicted_part_ids=point_part_ids,
                link_point_prompts=visualization_link_point_prompts_world,
                link_point_prompt_ids=visualization_link_point_prompt_ids,
                links=links_for_query_visualization,
            )
            yield (
                None,
                None,
                str(early_visualization_path),
                None,
                None,
                "Point query visualization ready. Running face postprocessing and articulation export...",
            )

            face_part_ids, face_part_ids_unrefined = _decode_face_part_ids(
                mesh_geometry.normalized_mesh,
                point_part_ids=point_part_ids,
                point_part_probabilities=point_part_probabilities,
                query_face_indices=query_face_indices,
                input_part_ids=np.arange(len(link_names), dtype=np.int32),
                strict=bool(args.strict_face_postprocess),
                enforce_connectivity_per_part=bool(args.enforce_connectivity_per_part),
            )
            motion_artifacts = _compute_motion_prediction_artifacts(
                args,
                model=model,
                batch=batch,
                normalized_mesh=mesh_geometry.normalized_mesh,
                face_part_ids=face_part_ids,
                joint_refit_num_query_points=int(args.num_query_points),
                num_links=len(link_names),
                center=mesh_geometry.center,
                scale=mesh_geometry.scale,
            )
            prediction = {
                "query_points": query_points,
                "query_normals": query_normals,
                "point_part_ids": point_part_ids,
                "face_part_ids": face_part_ids,
                "face_part_ids_unrefined": face_part_ids_unrefined,
                **motion_artifacts,
            }
            self._write_outputs(
                args=args,
                mesh_path=mesh_path,
                up_dir=canonical_up,
                output_dir=output_dir,
                mesh_geometry=mesh_geometry,
                batch=batch,
                query_face_indices=query_face_indices,
                link_names=link_names,
                joint_specs=joint_specs,
                prediction=prediction,
                segmentation_num_query_points=segmentation_num_query_points,
                visualization_path=early_visualization_path,
            )
            zip_path = _zip_directory(output_dir)
            yield (
                str(output_dir / "animated_textured.glb"),
                str(output_dir / "mesh_parts_with_axes.glb"),
                gr.update(),
                str(zip_path),
                str(output_dir),
                f"Success using input up direction {canonical_up}. Wrote outputs to {output_dir}",
            )
        except Exception as exc:
            traceback.print_exc()
            yield (
                None,
                None,
                gr.update()
                if early_visualization_path is not None
                and early_visualization_path.exists()
                else None,
                None,
                None,
                f"Error: {exc}",
            )
        finally:
            torch.cuda.empty_cache()

    def _prepare_geometry(self, mesh_path: Path, up_dir: str):
        from infer import _prepare_mesh_geometry

        return _prepare_mesh_geometry(input_path=mesh_path, up_dir=up_dir)

    def _segmentation_num_query_points(
        self,
        args: argparse.Namespace,
        normalized_mesh: Any,
    ) -> int:
        if args.num_query_points_per_face_for_seg is None:
            return int(args.num_query_points)
        num_faces = int(np.asarray(normalized_mesh.faces).shape[0])
        if num_faces <= 0:
            raise ValueError("Cannot sample query points for a mesh with no faces.")
        return int(args.num_query_points_per_face_for_seg) * num_faces

    def export_urdf_package(self, output_dir_value: Any) -> tuple[Any, str]:
        if not output_dir_value:
            return (
                gr.update(value=None, interactive=False),
                "Run inference before exporting URDF.",
            )
        output_dir = Path(str(output_dir_value)).expanduser().resolve()
        payload_path = output_dir / "urdf_export_payload.npz"
        metadata_path = output_dir / "urdf_export_metadata.json"
        if not payload_path.exists() or not metadata_path.exists():
            return (
                gr.update(value=None, interactive=False),
                f"URDF export payload is missing for {output_dir}.",
            )

        try:
            import trimesh

            payload = np.load(payload_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            vertices = np.asarray(payload["vertices"], dtype=np.float32)
            faces = np.asarray(payload["faces"], dtype=np.int64)
            face_part_ids = np.asarray(payload["face_part_ids"], dtype=np.int32)
            if faces.shape[0] != face_part_ids.shape[0]:
                raise ValueError(
                    "URDF export payload has mismatched face and part-label counts: "
                    f"{faces.shape[0]} faces vs {face_part_ids.shape[0]} labels."
                )
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            unique_part_ids = np.unique(face_part_ids).astype(np.int32, copy=False)
            mesh_parts_original = [
                mesh.submesh([face_part_ids == part_id], append=True)
                for part_id in unique_part_ids
            ]
            urdf_dir = output_dir / "urdf"
            if urdf_dir.exists():
                shutil.rmtree(urdf_dir)
            export_urdf(
                mesh_parts_original,
                unique_part_ids,
                [tuple(map(int, row)) for row in np.asarray(payload["motion_hierarchy"])],
                np.asarray(payload["is_part_revolute"], dtype=np.bool_),
                np.asarray(payload["is_part_prismatic"], dtype=np.bool_),
                np.asarray(payload["revolute_plucker"], dtype=np.float32),
                np.asarray(payload["revolute_range"], dtype=np.float32),
                np.asarray(payload["prismatic_axis"], dtype=np.float32),
                np.asarray(payload["prismatic_range"], dtype=np.float32),
                output_path=str(urdf_dir / "model.urdf"),
                name=str(metadata.get("urdf_name", output_dir.name)),
                link_names=list(metadata.get("link_names", [])),
            )
            urdf_zip = _zip_directory(urdf_dir)
            return (
                gr.update(value=str(urdf_zip), interactive=True),
                f"URDF exported to {urdf_dir}",
            )
        except Exception as exc:
            traceback.print_exc()
            return (
                gr.update(value=None, interactive=False),
                f"URDF export error: {exc}",
            )

    def _write_urdf_export_payload(
        self,
        *,
        output_dir: Path,
        mesh: Any,
        face_part_ids: np.ndarray,
        motion_hierarchy: list[tuple[int, int]],
        is_part_revolute: np.ndarray,
        is_part_prismatic: np.ndarray,
        revolute_plucker: np.ndarray,
        revolute_range: np.ndarray,
        prismatic_axis: np.ndarray,
        prismatic_range: np.ndarray,
        urdf_name: str,
        link_names: list[str],
    ) -> None:
        np.savez_compressed(
            output_dir / "urdf_export_payload.npz",
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            face_part_ids=np.asarray(face_part_ids, dtype=np.int32),
            motion_hierarchy=np.asarray(motion_hierarchy, dtype=np.int32),
            is_part_revolute=np.asarray(is_part_revolute, dtype=np.bool_),
            is_part_prismatic=np.asarray(is_part_prismatic, dtype=np.bool_),
            revolute_plucker=np.asarray(revolute_plucker, dtype=np.float32),
            revolute_range=np.asarray(revolute_range, dtype=np.float32),
            prismatic_axis=np.asarray(prismatic_axis, dtype=np.float32),
            prismatic_range=np.asarray(prismatic_range, dtype=np.float32),
        )
        (output_dir / "urdf_export_metadata.json").write_text(
            json.dumps(
                {
                    "urdf_name": str(urdf_name),
                    "link_names": [str(name) for name in link_names],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_outputs(
        self,
        *,
        args: argparse.Namespace,
        mesh_path: Path,
        up_dir: str,
        output_dir: Path,
        mesh_geometry: Any,
        batch: dict[str, Any],
        query_face_indices: np.ndarray,
        link_names: list[str],
        joint_specs: list[tuple[int, int, str]],
        prediction: dict[str, Any],
        segmentation_num_query_points: int,
        visualization_path: Path,
    ) -> None:
        motion_output = prediction["motion_output"]
        joint_refit_sampling = prediction["joint_refit_sampling"]
        motion_arrays_normalized = prediction["motion_arrays_normalized"]
        motion_arrays_world = prediction["motion_arrays_world"]
        prismatic_axis_world = prediction["prismatic_axis_world"]
        point_part_ids = prediction["point_part_ids"]
        face_part_ids = prediction["face_part_ids"]
        query_points = prediction["query_points"]

        _write_mesh_like_prediction_files(
            output_dir,
            face_part_ids=face_part_ids,
            query_points=query_points,
            query_face_indices=query_face_indices,
            point_part_ids=point_part_ids,
            center=mesh_geometry.center,
            scale=mesh_geometry.scale,
            joint_refit_sampling=joint_refit_sampling,
            query_normals=prediction["query_normals"],
        )

        motion_hierarchy = [
            (parent_link_id, child_link_id)
            for parent_link_id, child_link_id, _ in joint_specs
        ]
        predicted_kinematic = build_predicted_kinematic_records(
            link_names,
            joint_specs,
            revolute_plucker=motion_arrays_world["revolute_plucker"],
            prismatic_plucker=motion_arrays_world["prismatic_plucker"],
            prismatic_axis=prismatic_axis_world,
            revolute_range=motion_arrays_world["revolute_range"],
            prismatic_range=motion_arrays_world["prismatic_range"],
            revolute_parameter_valid=motion_arrays_normalized["revolute_parameter_valid"],
            prismatic_parameter_valid=motion_arrays_normalized["prismatic_parameter_valid"],
        )
        overparam_visualization_path = _write_kinematic_and_overparam_visualization(
            output_dir,
            kinematic_records=predicted_kinematic,
            visualization_records=None,
            motion_output=motion_output,
            query_points=query_points,
            point_part_ids=point_part_ids,
            joint_refit_sampling=joint_refit_sampling,
            center=mesh_geometry.center,
            scale=mesh_geometry.scale,
        )
        unique_part_ids = _save_articulated_mesh_outputs(
            args,
            output_dir=output_dir,
            original_mesh=mesh_geometry.original_mesh,
            face_part_ids=face_part_ids,
            motion_hierarchy=motion_hierarchy,
            is_part_revolute=motion_arrays_normalized["is_part_revolute"],
            is_part_prismatic=motion_arrays_normalized["is_part_prismatic"],
            revolute_plucker=motion_arrays_world["revolute_plucker"],
            revolute_range=motion_arrays_world["revolute_range"],
            prismatic_axis=prismatic_axis_world,
            prismatic_range=motion_arrays_world["prismatic_range"],
            urdf_name=mesh_path.stem,
            link_names=link_names,
        )
        self._write_urdf_export_payload(
            output_dir=output_dir,
            mesh=mesh_geometry.original_mesh,
            face_part_ids=face_part_ids,
            motion_hierarchy=motion_hierarchy,
            is_part_revolute=motion_arrays_normalized["is_part_revolute"],
            is_part_prismatic=motion_arrays_normalized["is_part_prismatic"],
            revolute_plucker=motion_arrays_world["revolute_plucker"],
            revolute_range=motion_arrays_world["revolute_range"],
            prismatic_axis=prismatic_axis_world,
            prismatic_range=motion_arrays_world["prismatic_range"],
            urdf_name=mesh_path.stem,
            link_names=link_names,
        )

        metadata = _base_metadata(
            args,
            mode="mesh",
            input_path=mesh_path,
            run_dir=self.run_dir,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            num_shape_points=self.num_shape_points,
            segmentation_num_query_points=segmentation_num_query_points,
            joint_refit_num_query_points=int(args.num_query_points),
            sharp_point_ratio=self.sharp_point_ratio,
        ) | {
            "link_names": link_names,
            "num_input_parts": int(len(link_names)),
            "joint_specs": [
                {
                    "parent_link_id": int(parent_link_id),
                    "child_link_id": int(child_link_id),
                    "joint_type": joint_type,
                }
                for parent_link_id, child_link_id, joint_type in joint_specs
            ],
            "unique_part_ids_in_segmentation": unique_part_ids.astype(int).tolist(),
            "visualization_path": str(visualization_path),
            "normalization": {
                "center": mesh_geometry.center.tolist(),
                "scale": mesh_geometry.scale,
            },
            "mesh_orientation": {
                "input_up_dir": str(up_dir),
                "canonical_up_dir": "+Z",
                "reoriented_to_canonical_z_up": bool(up_dir != "+Z"),
                "rotation_matrix": mesh_geometry.up_dir_rotation.tolist(),
            },
        } | _joint_refit_metadata(
            joint_refit_sampling,
            num_links=len(link_names),
        )
        _write_metadata_and_summary(
            output_dir=output_dir,
            metadata=metadata,
            checkpoint_path=self.checkpoint_path,
            unique_part_ids=unique_part_ids,
            visualization_path=visualization_path,
            overparam_visualization_path=overparam_visualization_path,
            optional_paths={
                "overparam_visualization_path": overparam_visualization_path,
                "urdf_export_payload_path": output_dir / "urdf_export_payload.npz",
            },
        )


_ACTIVE_APP: InstructParticulateApp | None = None


def _get_active_app() -> InstructParticulateApp:
    global _ACTIVE_APP
    if _ACTIVE_APP is None:
        _ACTIVE_APP = InstructParticulateApp()
    return _ACTIVE_APP


@_spaces_gpu
def run_predict_on_gpu(
    mesh_path_value: Any,
    kinematic_tree_json: str,
    point_prompt_json: str,
    up_dir: str,
    num_query_points: int,
    num_query_points_per_face_for_seg: int,
    query_batch_size: int,
    animation_frames: int,
    strict_face_postprocess: bool,
    enforce_connectivity_per_part: bool,
    joint_decoding_confidence_temperature: float,
):
    yield from _get_active_app().predict(
        mesh_path_value,
        kinematic_tree_json,
        point_prompt_json,
        up_dir,
        num_query_points,
        num_query_points_per_face_for_seg,
        query_batch_size,
        animation_frames,
        strict_face_postprocess,
        enforce_connectivity_per_part,
        joint_decoding_confidence_temperature,
    )


def _example_meshes() -> list[list[str]]:
    examples_dir = REPO_ROOT / "examples"
    if not examples_dir.exists():
        return []
    return [[str(path)] for path in sorted(examples_dir.glob("*.glb"))]


def _html_attr_json(value: dict[str, Any]) -> str:
    return (
        json.dumps(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _kinematic_tree_editor_html() -> str:
    return f"""
    <style>
    {KINEMATIC_TREE_EDITOR_CSS}
    </style>
    <div id="kin-tree-editor" class="kin-tree-panel" data-default-tree="{_html_attr_json(DEFAULT_KINEMATIC_TREE)}">
      <div class="kin-tree-header">
        <h3>Kinematic Tree Editor</h3>
        <p>Drag colored link rectangles to arrange the tree. Add links with <strong>Add Link</strong>. To add a joint, press <strong>Add Joint</strong>, then click the parent link followed by the child link. New joints start as revolute; use each joint's dropdown to change it to prismatic. Link names are editable inside each rectangle.</p>
      </div>
      <div class="kin-tree-toolbar">
        <button id="kin-add-node" class="primary" type="button">Add Link</button>
        <button id="kin-add-joint" type="button">Add Joint</button>
        <button id="kin-delete-selected" type="button">Delete Selected</button>
        <button id="kin-reset-tree" type="button">Reset</button>
        <span id="kin-tree-status" class="kin-tree-status"></span>
      </div>
      <div id="kin-tree-canvas" class="kin-tree-canvas">
        <svg id="kin-tree-edges" class="kin-tree-edge-layer"></svg>
        <div id="kin-tree-joints" class="kin-tree-joint-layer"></div>
        <div id="kin-tree-nodes" class="kin-tree-node-layer">
          <div class="kin-node" style="left: 28px; top: 42px; background: #fca5a5;">
            <div class="kin-node-row">
              <input value="base" aria-label="Link name" />
              <button class="kin-node-delete" title="Delete link" type="button">×</button>
            </div>
          </div>
          <div class="kin-node" style="left: 220px; top: 180px; background: #fdba74;">
            <div class="kin-node-row">
              <input value="moving_part" aria-label="Link name" />
              <button class="kin-node-delete" title="Delete link" type="button">×</button>
            </div>
          </div>
        </div>
      </div>
    </div>
    """


def _point_prompt_picker_html() -> str:
    return """
    <div id="point-prompt-picker" class="prompt-picker-panel">
      <div class="prompt-picker-header">
        <h3>Point Prompt Picker</h3>
        <p>Select a link in the tree, then click the mesh below to set an optional point prompt for that link. Drag to rotate and scroll to zoom.</p>
      </div>
      <div class="prompt-picker-toolbar">
        <button id="point-prompt-clear-link" type="button">Clear Link Prompt</button>
        <button id="point-prompt-clear-all" type="button">Clear All Prompts</button>
        <button id="point-prompt-reset-view" type="button">Reset View</button>
        <span id="point-prompt-status" class="prompt-picker-status"></span>
      </div>
      <div class="prompt-picker-canvas-wrap">
        <canvas id="point-prompt-canvas"></canvas>
        <div id="point-prompt-empty" class="prompt-picker-empty">Upload a mesh to enable point picking.</div>
      </div>
    </div>
    """


def create_gradio_app(app: InstructParticulateApp) -> gr.Blocks:
    global _ACTIVE_APP
    _ACTIVE_APP = app
    with gr.Blocks(title="Instruct Particulate Demo") as demo:
        gr.HTML(
            f"""
            <style>
            {DEMO_CSS}
            {KINEMATIC_TREE_EDITOR_CSS}
            </style>
            <h1>Instruct Particulate</h1>
            <p>Upload one object mesh and provide a kinematic tree. The demo predicts part segmentation and joint parameters, then exports animated GLB and URDF assets.</p>
            """
        )

        loaded_mesh_path = gr.State(None)
        selected_up_dir = gr.Textbox(
            value="",
            label="Selected Upright Direction",
            elem_id="selected_up_dir",
            elem_classes=["kinematic-json-sync"],
        )
        latest_output_dir = gr.State(None)

        with gr.Row(equal_height=True, elem_classes=["demo-row", "demo-top-row"]):
            with gr.Column(scale=1, elem_classes=["demo-panel", "mesh-panel"]):
                input_mesh = gr.Model3D(label="Input Mesh", interactive=True, height=360)
                examples = _example_meshes()
                examples_container = None
                if examples:
                    with gr.Column(elem_classes=["mesh-examples"]) as examples_container:
                        pass
                mesh_status = gr.Textbox(
                    label="Mesh Status",
                    interactive=False,
                    visible=False,
                    elem_classes=["kinematic-json-sync"],
                )
                with gr.Accordion("Inference Parameters", open=True):
                    with gr.Row():
                        num_query_points = gr.Number(
                            value=102400,
                            label="Joint Refit Query Points",
                            precision=0,
                            minimum=512,
                        )
                        per_face_queries = gr.Number(
                            value=3,
                            label="Segmentation Queries Per Face",
                            precision=0,
                            minimum=0,
                        )
                    with gr.Row():
                        query_batch_size = gr.Number(
                            value=8192,
                            label="Query Batch Size",
                            precision=0,
                            minimum=256,
                        )
                        animation_frames = gr.Number(
                            value=50,
                            label="Animation Frames",
                            precision=0,
                            minimum=2,
                        )
                    with gr.Row():
                        strict = gr.Checkbox(
                            label="Strict Face Postprocess",
                            value=True,
                        )
                        connectivity = gr.Checkbox(
                            label="Enforce Connectivity Per Part",
                            value=False,
                        )
                    confidence_temperature = gr.Number(
                        value=1.0,
                        label="Joint Confidence Temperature",
                        minimum=0.01,
                    )

            with gr.Column(scale=1, elem_classes=["demo-panel", "kin-panel"]):
                extract_button = gr.Button("Extract Kinematic Structure")
                auto_status = gr.Textbox(
                    label="Kinematic Extraction Status",
                    interactive=False,
                )
                gr.HTML(
                    _kinematic_tree_editor_html(),
                    elem_id="kinematic_tree_editor_html",
                    elem_classes=["kinematic-editor-host"],
                    min_height=500,
                    container=False,
                    padding=False,
                )
                with gr.Accordion("Auto-Kinematic Rendered Images", open=False):
                    auto_render_gallery = gr.Gallery(
                        label="Auto-Kinematic Rendered Images",
                        columns=3,
                        height=160,
                        object_fit="contain",
                        elem_classes=["capped-output"],
                    )

            with gr.Column(scale=1, elem_classes=["demo-panel", "prompt-panel"]):
                gr.HTML(_point_prompt_picker_html())
                with gr.Row(elem_classes=["prompt-run-row"]):
                    run_button = gr.Button("Run Inference", variant="primary")

        kinematic_tree = gr.Textbox(
            label="Kinematic Tree JSON Sync",
            value=_tree_to_pretty_json(DEFAULT_KINEMATIC_TREE),
            lines=8,
            elem_id="kinematic_tree_json",
            elem_classes=["kinematic-json-sync"],
        )
        point_prompt_mesh_data = gr.Textbox(
            label="Point Prompt Mesh Data",
            value="",
            lines=1,
            elem_id="point_prompt_mesh_data",
            elem_classes=["kinematic-json-sync"],
        )
        point_prompts = gr.Textbox(
            label="Point Prompt JSON Sync",
            value='{"prompts":[]}',
            lines=4,
            elem_id="point_prompt_json",
            elem_classes=["kinematic-json-sync"],
        )

        with gr.Row(equal_height=True, elem_classes=["demo-row", "demo-bottom-row"]):
            with gr.Column(scale=1, elem_classes=["demo-panel", "orientation-panel"]):
                gr.Markdown(
                    "### Upright Orientation\n"
                    "After uploading or selecting a mesh, click the preview where the object is upright. "
                    "That choice is used for both kinematic extraction and inference."
                )
                upright_preview_images: list[gr.Image] = []
                with gr.Column(elem_classes=["upright-picker-grid"]):
                    for up_dir_row in (UP_DIR_CHOICES[:3], UP_DIR_CHOICES[3:]):
                        with gr.Row(equal_height=True, elem_classes=["upright-picker-row"]):
                            for up_dir in up_dir_row:
                                preview_image = gr.Image(
                                    type="filepath",
                                    show_label=False,
                                    container=False,
                                    interactive=False,
                                    buttons=[],
                                    elem_id=f"upright-option-{_up_dir_slug(up_dir)}",
                                    elem_classes=["upright-option-image"],
                                )
                                preview_image.buttons = []
                                upright_preview_images.append(preview_image)

                if examples and examples_container is not None:
                    with examples_container:
                        gr.Examples(
                            examples=examples,
                            inputs=[input_mesh],
                            outputs=[
                                loaded_mesh_path,
                                mesh_status,
                                point_prompt_mesh_data,
                                *upright_preview_images,
                                selected_up_dir,
                            ],
                            fn=app.register_mesh,
                            run_on_click=True,
                            cache_examples=False,
                            label="Examples",
                        )

            with gr.Column(scale=2, elem_classes=["demo-panel", "outputs-panel"]):
                with gr.Row(equal_height=True, elem_classes=["output-triplet-row"]):
                    query_visualization = gr.Image(
                        label="Point Query Visualization",
                        height=400,
                        scale=1,
                        elem_classes=["capped-output"],
                    )
                    animated_model = gr.Model3D(
                        label="Animated Articulated Model",
                        height=400,
                        scale=1,
                        elem_classes=["capped-output"],
                    )
                    prediction_model = gr.Model3D(
                        label="Predicted Parts and Axes",
                        height=400,
                        scale=1,
                        elem_classes=["capped-output"],
                    )
                output_zip = gr.File(
                    label="All Outputs",
                    visible=False,
                    elem_classes=["kinematic-json-sync"],
                )
                with gr.Row(elem_classes=["urdf-export-row"]):
                    export_urdf_button = gr.Button("Export URDF")
                    urdf_zip = gr.DownloadButton(
                        "Download URDF ZIP",
                        value=None,
                        interactive=False,
                    )
                urdf_status = gr.Textbox(
                    label="URDF Export Status",
                    interactive=False,
                    visible=False,
                    elem_classes=["kinematic-json-sync"],
                )
                status = gr.Textbox(label="Status", interactive=False)

        input_mesh.change(
            fn=app.register_mesh,
            inputs=[input_mesh],
            outputs=[
                loaded_mesh_path,
                mesh_status,
                point_prompt_mesh_data,
                *upright_preview_images,
                selected_up_dir,
            ],
        )
        extract_button.click(
            fn=app.extract_kinematic_structure,
            inputs=[
                input_mesh,
                kinematic_tree,
                point_prompts,
                selected_up_dir,
            ],
            outputs=[
                kinematic_tree,
                point_prompts,
                auto_render_gallery,
                auto_status,
            ],
        )
        run_button.click(
            fn=run_predict_on_gpu,
            inputs=[
                input_mesh,
                kinematic_tree,
                point_prompts,
                selected_up_dir,
                num_query_points,
                per_face_queries,
                query_batch_size,
                animation_frames,
                strict,
                connectivity,
                confidence_temperature,
            ],
            outputs=[
                animated_model,
                prediction_model,
                query_visualization,
                output_zip,
                latest_output_dir,
                status,
            ],
        )
        export_urdf_button.click(
            fn=app.export_urdf_package,
            inputs=[latest_output_dir],
            outputs=[urdf_zip, urdf_status],
        )
        demo.load(fn=None, js=UPRIGHT_PICKER_JS)
        demo.load(fn=None, js=KINEMATIC_TREE_EDITOR_JS)
        demo.queue()
    return demo


if __name__ == "__main__":
    print("Initializing Instruct Particulate demo...")
    demo_app = InstructParticulateApp()
    demo = create_gradio_app(demo_app)
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        allowed_paths=[str(demo_app.output_root.resolve())],
    )
