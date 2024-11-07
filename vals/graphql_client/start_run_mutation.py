# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import Optional

from pydantic import Field

from .base_model import BaseModel


class StartRunMutation(BaseModel):
    start_run: Optional["StartRunMutationStartRun"] = Field(alias="startRun")


class StartRunMutationStartRun(BaseModel):
    run_id: str = Field(alias="runId")


StartRunMutation.model_rebuild()
