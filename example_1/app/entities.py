from pydantic import BaseModel

class PayloadInstance(BaseModel):
    feat1: float
    feat2: float
    feat3: float
    feat4: float
    feat5: float
    feat6: float
    feat7: float
    feat8: float
    feat9: float
    feat10: float
    feat11: float
    feat12: float
    feat13: float
    feat14: float
    feat15: float
    feat16: float
    feat17: float
    feat18: float
    feat19: float
    feat20: float
    feat21: float
    feat22: float
    feat23: float
    feat24: float
    feat25: float
    feat26: float

feat_ls = list(PayloadInstance.model_fields)


class PredictionRequest(BaseModel):
    instances: list[PayloadInstance]
