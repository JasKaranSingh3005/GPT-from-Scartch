from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    """Base schema. Older pydantic v2 releases warn on fields starting with 'model_'."""

    model_config = ConfigDict(protected_namespaces=())