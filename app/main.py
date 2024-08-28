from enum import Enum
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from app.external_services.smp_binding_affinity import SMPBindingAffinityModel as Model, Molecule as ModelMolecule, Protein as ModelProtein
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

@app.get("/status")
def test_api_connection():
    return {"status": "successful"}

@app.post("/predict/{protein_code}")
def predict_small_molecule_protein_binding_affinity(protein_code: Protein, molecules: List[Molecule]) -> List[Molecule]:
    model_molecules = [ModelMolecule(id=molecule.id, smile=molecule.smile) for molecule in molecules]
    model_protein = ModelProtein(acronym=protein_code.value)

    model = Model(
        gcs_bucket_name=GCS_BUCKET_NAME,
        gcs_model_path=f"{GCS_MODEL_PATH}/{protein_code.value}_model.pt"
    )

    predicted_molecules = model.predict(
        protein=model_protein,
        molecules=model_molecules
    )

    return {
        "data": [{"molecule": ModelMolecule(id=molecule.id, smile=molecule.smile), "protein": protein_code, "binding_affinity": molecule.binding_affinity} for molecule in predicted_molecules]
    }