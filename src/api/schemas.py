from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE = {
    "duration_months": 24,
    "credit_amount": 3500.0,
    "installment_rate": 3,
    "residence_since": 2,
    "age": 35,
    "existing_credits": 1,
    "num_dependents": 1,
    "checking_status": "A12",
    "credit_history": "A32",
    "purpose": "A43",
    "savings": "A61",
    "employment_since": "A73",
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "property": "A121",
    "other_installment_plans": "A143",
    "housing": "A152",
    "job": "A173",
    "telephone": "A192",
    "foreign_worker": "A201",
}


class CreditApplication(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    duration_months: int = Field(ge=1, le=120, description="Durée du crédit en mois")
    credit_amount: float = Field(gt=0, le=1_000_000, description="Montant du crédit")
    installment_rate: int = Field(ge=1, le=4, description="Taux d'effort en % du revenu (1 à 4)")
    residence_since: int = Field(ge=1, le=4, description="Ancienneté de résidence (1 à 4)")
    age: int = Field(ge=18, le=100)
    existing_credits: int = Field(ge=1, le=10, description="Crédits existants dans la banque")
    num_dependents: int = Field(ge=0, le=10, description="Personnes à charge")

    checking_status: Literal["A11", "A12", "A13", "A14"]
    credit_history: Literal["A30", "A31", "A32", "A33", "A34"]
    purpose: Literal["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"]
    savings: Literal["A61", "A62", "A63", "A64", "A65"]
    employment_since: Literal["A71", "A72", "A73", "A74", "A75"]
    personal_status_sex: Literal["A91", "A92", "A93", "A94"]
    other_debtors: Literal["A101", "A102", "A103"]
    property: Literal["A121", "A122", "A123", "A124"]
    other_installment_plans: Literal["A141", "A142", "A143"]
    housing: Literal["A151", "A152", "A153"]
    job: Literal["A171", "A172", "A173", "A174"]
    telephone: Literal["A191", "A192"]
    foreign_worker: Literal["A201", "A202"]


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    default_probability: float = Field(ge=0, le=1, description="Probabilité de défaut de paiement")
    prediction: Literal[0, 1] = Field(description="1 = défaut prévu, 0 = bon payeur")
    model_version: str


class BatchRequest(BaseModel):
    applications: list[CreditApplication] = Field(min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: list[Prediction]
    model_version: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    alias: str
    model_version: str
    run_id: str
    decision_threshold: float
