from enum import Enum
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import os

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH")

app = FastAPI()

class Molecule(BaseModel):
    id: int
    smile: str
    is_binded: bool = None
    binding_affinity: float = None

class Protein(str, Enum):
    seh = 'sEH'
    brd4 = 'BRD4'
    hsa = 'HSA'

@app.get("/test")
def test_api_connection():
    return {"version": "0.0.1-rc.0"}

@app.post("/predict/{protein_code}")
def predict_small_molecule_protein_binding_affinity(protein_code: Protein, molecules: List[Molecule]) -> List[Molecule]:
    from app.smp_binding_affinity import model as smpba_model, datatypes

    model_molecules = [datatypes.Molecule(id=molecule.id, smile=molecule.smile) for molecule in molecules]
    model_protein = datatypes.Protein(acronym=protein_code.value)

    model = smpba_model.SMPBindingAffinityModel(
        gcs_bucket_name=GCS_BUCKET_NAME,
        gcs_model_path=f"{GCS_MODEL_PATH}/{protein_code.value}_model.pt"
    )

    predicted_molecules = model.predict(
        protein=model_protein,
        molecules=model_molecules
    )

    return {
        "data": [{"molecule": datatypes.Molecule(id=molecule.id, smile=molecule.smile), "protein": protein_code, "binding_affinity": molecule.binding_affinity} for molecule in predicted_molecules]
    }