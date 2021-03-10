def moonshot():
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
    import pandas as pd
    df = pd.read_csv("covid_submissions_all_info.csv")
    df = df.dropna(subset=["f_avg_pIC50"])
    pairs = []
    for idx, row in df.dropna(subset=['f_avg_pIC50']).iterrows():
        candidates = row['inspired_by']
        if not isinstance(candidates, str):
            continue
        candidates = candidates.split(",")

        for candidate in candidates:
            if len(df.where(df['CID']==candidate).dropna(subset=['CID']).index) > 0:
                pairs.append((idx, df.where(df['CID']==candidate).dropna(subset=['CID']).index.item()))


    pairs_tr = pairs[:100]
    pairs_te = pairs[100:]

    _pairs_tr = list(zip(*pairs_tr))
    _pairs_te = list(zip(*pairs_te))

    _ds_tr = [
        (
            smiles_to_bigraph(df.loc[x]["SMILES"], node_featurizer=CanonicalAtomFeaturizer(atom_data_field='feat')),
            df.loc[x]["f_avg_pIC50"],
        )
        for x in set(_pairs_tr[0])
    ]

    _ds_te = [
        (
            smiles_to_bigraph(df.loc[x]["SMILES"], node_featurizer=CanonicalAtomFeaturizer(atom_data_field='feat')),
            df.loc[x]["f_avg_pIC50"],
        )
        for x in set(_pairs_te[0])
    ]

    ds_tr = [
          (
              smiles_to_bigraph(df.loc[x]["SMILES"], node_featurizer=CanonicalAtomFeaturizer(atom_data_field='feat')),
              smiles_to_bigraph(df.loc[y]["SMILES"], node_featurizer=CanonicalAtomFeaturizer(atom_data_field='feat')),
              df.loc[x]["f_avg_pIC50"],
              df.loc[y]["f_avg_pIC50"]
          )
          for x, y in pairs_tr
    ]

    ds_te = [
          (
              smiles_to_bigraph(df.loc[x]["SMILES"], node_featurizer=CanonicalAtomFeaturizer(atom_data_field='feat')),
              smiles_to_bigraph(df.loc[y]["SMILES"], node_featurizer=CanonicalAtomFeaturizer(atom_data_field='feat')),
              df.loc[x]["f_avg_pIC50"],
              df.loc[y]["f_avg_pIC50"]
          )
          for x, y in pairs_te
    ]

    return ds_tr, ds_te
