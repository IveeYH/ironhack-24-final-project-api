from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel

app = FastAPI()

class Molecule(BaseModel):
    molecule_id: int
    smiles: str

class Protein(BaseModel):
    protein_id: int
    sequence: str

@app.get("/test")
def test_api_connection():
    return {"project": "small-molecule-protein-binding-affinity"}

@app.post("/predict")
def predict_small_molecule_protein_binding_affinity(molecule: Molecule, protein: Protein):
    return {
        "molecule_id": molecule.molecule_id,
        "protein_id": protein.protein_id,
        "predicted_affinity": 0.9232302
    }