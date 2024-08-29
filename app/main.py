from enum import Enum
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.external_services.smp_binding_affinity import SMPBindingAffinityModel as Model, Molecule as ModelMolecule, Protein as ModelProtein
import os

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

class Molecule(BaseModel):
    id: int
    smiles: str
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
    model_molecules = [ModelMolecule(molecule_id=molecule.id, smile=molecule.smiles) for molecule in molecules]
    model_protein = ModelProtein(acronym=protein_code.value)

    model = Model(
        gcs_bucket_name=GCS_BUCKET_NAME,
        gcs_model_path=f"{GCS_MODEL_PATH}/{protein_code.value}_model.pt"
    )

    predicted_molecules = model.predict(
        protein=model_protein,
        molecules=model_molecules
    )

    print(predicted_molecules)

    final_molecules = []
    for molecule in predicted_molecules:
        final_molecules.append(Molecule(
                id=molecule.molecule_id, 
                smiles=molecule.smile,
                binding_affinity=molecule.binding_affinity,
                is_binded=molecule.is_binded))

    return final_molecules