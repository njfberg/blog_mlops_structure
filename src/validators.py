from pydantic import BaseModel, validator


class InputData(BaseModel):
    """Pydantic class to validate input data"""

    id: int
    feature1: float
    feature2: float

    @validator("id")
    def id_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("id must be a positive integer")
        return v


class OutputData(BaseModel):
    """Pydantic class to validate output data"""

    id: int
    prediction: int

    @validator("id")
    def id_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("id must be a positive integer")
        return v
