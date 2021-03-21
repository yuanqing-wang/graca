import graca


def moonshot():
    
    from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer
    import pandas as pd
    import os
    df = pd.read_csv(os.path.dirname(graca.data.collections.__file__) + "/covid_submissions_all_info.csv")
    df = df.dropna(subset=["f_avg_pIC50"])

    from rdkit import Chem
    from rdkit.Chem import MCS

    ds = []
    for idx0, row0 in df.iterrows():
        smiles0 = row0["SMILES"]
        mol0 = Chem.MolFromSmiles(smiles0)
        for idx1, row1 in df.iloc[idx0+1:].iterrows():
            smiles1 = row1["SMILES"]
            mol1 = Chem.MolFromSmiles(smiles1)
            res = MCS.FindMCS([mol0, mol1])
            if res.numAtoms > 15:
                ds.append(
                    (
                        mol_to_bigraph(mol1, node_featurizer=CanonicalAtomFeaturizer(atom_data_field='feat')),
                        mol_to_bigraph(mol0, node_featurizer=CanonicalAtomFeaturizer(atom_data_field='feat')),
                        row1["f_avg_pIC50"],
                        row0["f_avg_pIC50"],
                    )
                )
                
    ds_tr = ds[:500]
    ds_te = ds[500:]
            
    return ds_tr, ds_te
