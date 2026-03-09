# -*- coding: utf-8 -*-
"""Unit tests for ModelRegistry."""

from __future__ import annotations

import pytest

from src.registry.model_registry import ModelRegistry, ModelStage


class TestModelRegistry:
    """Tests for the model registry."""

    def test_register_model(self, model_registry: ModelRegistry):
        """Registering a model should create version 1."""
        v = model_registry.register_model(
            name="my_model",
            source_run_id="run_abc",
            description="First version",
            metrics={"accuracy": 0.95},
        )
        assert v.name == "my_model"
        assert v.version == 1
        assert v.source_run_id == "run_abc"
        assert v.metrics["accuracy"] == 0.95
        assert v.stage == ModelStage.NONE

    def test_auto_increment_version(self, model_registry: ModelRegistry):
        """Registering the same model name should auto-increment versions."""
        model_registry.register_model(name="inc_model", source_run_id="r1")
        v2 = model_registry.register_model(name="inc_model", source_run_id="r2")
        assert v2.version == 2

    def test_get_model_latest(self, model_registry: ModelRegistry):
        """Getting a model without specifying version returns the latest."""
        model_registry.register_model(name="latest_test", source_run_id="r1")
        model_registry.register_model(name="latest_test", source_run_id="r2")

        latest = model_registry.get_model("latest_test")
        assert latest is not None
        assert latest.version == 2

    def test_get_model_specific_version(self, model_registry: ModelRegistry):
        """Getting a specific version should return that exact version."""
        model_registry.register_model(name="ver_test", source_run_id="r1")
        model_registry.register_model(name="ver_test", source_run_id="r2")

        v1 = model_registry.get_model("ver_test", version=1)
        assert v1 is not None
        assert v1.version == 1
        assert v1.source_run_id == "r1"

    def test_transition_stage(self, model_registry: ModelRegistry):
        """Stage transitions should update the model's stage field."""
        model_registry.register_model(name="stage_test", source_run_id="r1")

        v = model_registry.transition_stage("stage_test", 1, ModelStage.STAGING)
        assert v is not None
        assert v.stage == ModelStage.STAGING

        v = model_registry.transition_stage("stage_test", 1, ModelStage.PRODUCTION)
        assert v.stage == ModelStage.PRODUCTION

    def test_production_archives_previous(self, model_registry: ModelRegistry):
        """Promoting to Production should archive the previous production version."""
        model_registry.register_model(name="arch_test", source_run_id="r1")
        model_registry.register_model(name="arch_test", source_run_id="r2")

        model_registry.transition_stage("arch_test", 1, ModelStage.PRODUCTION)
        model_registry.transition_stage("arch_test", 2, ModelStage.PRODUCTION)

        v1 = model_registry.get_model("arch_test", version=1)
        assert v1.stage == ModelStage.ARCHIVED

        v2 = model_registry.get_model("arch_test", version=2)
        assert v2.stage == ModelStage.PRODUCTION

    def test_list_versions(self, model_registry: ModelRegistry):
        """Listing versions should return all versions in order."""
        model_registry.register_model(name="list_test", source_run_id="r1")
        model_registry.register_model(name="list_test", source_run_id="r2")
        model_registry.register_model(name="list_test", source_run_id="r3")

        versions = model_registry.list_versions("list_test")
        assert len(versions) == 3
        assert [v.version for v in versions] == [1, 2, 3]

    def test_get_model_not_found(self, model_registry: ModelRegistry):
        """Getting a nonexistent model should return None."""
        assert model_registry.get_model("no_such_model") is None

    def test_delete_model(self, model_registry: ModelRegistry):
        """Deleting a model should remove it and all versions."""
        model_registry.register_model(name="del_test", source_run_id="r1")
        model_registry.register_model(name="del_test", source_run_id="r2")

        assert model_registry.delete_model("del_test") is True
        assert model_registry.get_model("del_test") is None
        assert model_registry.list_versions("del_test") == []
