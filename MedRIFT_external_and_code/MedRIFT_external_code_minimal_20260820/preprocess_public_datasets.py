# 此文件更新于 2026-08-20 15:26（北京时间）
"""Build leakage-safe MedRIFT tables from the three downloaded public datasets.

The generated CSV files intentionally retain cleaned, unstandardized values.
Imputation, clipping, categorical encoding, and standardization must be fitted
inside each training fold by the experiment code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


UPDATED_AT = "2026-07-16 15:15 Asia/Tokyo"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "source_raw"
OUT = ROOT / "data"
MANIFEST_OUT = ROOT / "resources" / "manifests"
GRAPH_OUT = ROOT / "resources" / "graphs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps({"_updated_at": UPDATED_AT, **payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def graph_from_sources(
    features: list[str],
    panels: dict[str, list[str]],
    groups: dict[str, list[str]],
    loinc: dict[str, Optional[str]],
) -> dict:
    feature_set = set(features)
    if set(loinc) != feature_set:
        raise ValueError("LOINC mapping keys must exactly match the feature list")
    panel_members = {member for members in panels.values() for member in members}
    unmapped_panel_members = sorted(member for member in panel_members if not loinc.get(member))
    if unmapped_panel_members:
        raise ValueError(f"Every E_med panel member requires a LOINC mapping: {unmapped_panel_members}")

    def edges_from_sets(source_sets: dict[str, list[str]]) -> list[list[str]]:
        edges: set[tuple[str, str]] = set()
        for members in source_sets.values():
            active = sorted(set(members) & feature_set)
            for j, left in enumerate(active):
                for right in active[j + 1 :]:
                    edges.add((left, right))
        return [list(edge) for edge in sorted(edges)]

    medical_edges = edges_from_sets(panels)
    group_edges = edges_from_sets(groups)
    union = sorted({tuple(edge) for edge in medical_edges + group_edges})
    return {
        "edge_definition": "E = E_med union E_group; no train-correlation edges",
        "nodes": features,
        "loinc_mapping": loinc,
        "loinc_panels": panels,
        "clinical_groups": groups,
        "E_med": medical_edges,
        "E_group": group_edges,
        "E": [list(edge) for edge in union],
        "counts": {
            "nodes": len(features),
            "E_med": len(medical_edges),
            "E_group": len(group_edges),
            "E_union": len(union),
        },
    }


def build_nhanes() -> tuple[pd.DataFrame, dict, dict]:
    folder = RAW / "nhanes"

    def read(name: str) -> pd.DataFrame:
        return pd.read_sas(folder / f"{name}.xpt", format="xport")

    demo = read("DEMO_J")[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]]
    target = read("DIQ_J")[["SEQN", "DIQ010"]]
    bmx = read("BMX_J")[["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]]
    cbc = read("CBC_J")[[
        "SEQN", "LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT",
        "LBXEOPCT", "LBXBAPCT", "LBXRBCSI", "LBXHGB", "LBXHCT",
        "LBXMCVSI", "LBXMCHSI", "LBXMC", "LBXRDW", "LBXPLTSI", "LBXMPSI",
    ]]
    biopro = read("BIOPRO_J")[[
        "SEQN", "LBXSATSI", "LBXSAL", "LBXSAPSI", "LBXSASSI", "LBXSBU",
        "LBXSCLSI", "LBXSCK", "LBXSCR", "LBXSGB", "LBXSGL", "LBXSGTSI",
        "LBXSIR", "LBXSLDSI", "LBXSOSSI", "LBXSPH", "LBXSKSI", "LBXSNASI",
        "LBXSTB", "LBXSCA", "LBXSCH", "LBXSTP", "LBXSTR", "LBXSUA",
    ]]
    hdl = read("HDL_J")[["SEQN", "LBDHDD"]]
    ghb = read("GHB_J")[["SEQN", "LBXGH"]]

    merged = demo.merge(target, on="SEQN", how="left", validate="one_to_one")
    for table in (bmx, cbc, biopro, hdl, ghb):
        merged = merged.merge(table, on="SEQN", how="left", validate="one_to_one")

    merged = merged.loc[
        (merged["RIDAGEYR"] >= 20) & merged["DIQ010"].isin([1.0, 2.0])
    ].copy()

    rename = {
        "SEQN": "patient_id", "RIDAGEYR": "age_years", "RIAGENDR": "sex",
        "RIDRETH3": "race_ethnicity", "BMXWT": "weight_kg", "BMXHT": "height_cm",
        "BMXBMI": "bmi_kg_m2", "BMXWAIST": "waist_cm", "LBXWBCSI": "wbc_10e9_l",
        "LBXLYPCT": "lymphocyte_pct", "LBXMOPCT": "monocyte_pct",
        "LBXNEPCT": "neutrophil_pct", "LBXEOPCT": "eosinophil_pct",
        "LBXBAPCT": "basophil_pct", "LBXRBCSI": "rbc_10e12_l", "LBXHGB": "hemoglobin_g_dl",
        "LBXHCT": "hematocrit_pct", "LBXMCVSI": "mcv_fl", "LBXMCHSI": "mch_pg",
        "LBXMC": "mchc_g_dl", "LBXRDW": "rdw_pct", "LBXPLTSI": "platelets_10e9_l",
        "LBXMPSI": "mpv_fl", "LBXSATSI": "alt_u_l", "LBXSAL": "albumin_g_dl",
        "LBXSAPSI": "alp_u_l", "LBXSASSI": "ast_u_l", "LBXSBU": "bun_mg_dl",
        "LBXSCLSI": "chloride_mmol_l", "LBXSCK": "ck_u_l", "LBXSCR": "creatinine_mg_dl",
        "LBXSGB": "globulin_g_dl", "LBXSGL": "glucose_mg_dl", "LBXSGTSI": "ggt_u_l",
        "LBXSIR": "iron_ug_dl", "LBXSLDSI": "ldh_u_l", "LBXSOSSI": "osmolality_mOsm_kg",
        "LBXSPH": "phosphorus_mg_dl", "LBXSKSI": "potassium_mmol_l",
        "LBXSNASI": "sodium_mmol_l", "LBXSTB": "bilirubin_total_mg_dl",
        "LBXSCA": "calcium_mg_dl", "LBXSCH": "cholesterol_total_mg_dl",
        "LBXSTP": "protein_total_g_dl", "LBXSTR": "triglycerides_mg_dl",
        "LBXSUA": "uric_acid_mg_dl", "LBDHDD": "hdl_cholesterol_mg_dl",
        "LBXGH": "hba1c_pct",
    }
    df = merged.rename(columns=rename)
    df["patient_id"] = df["patient_id"].astype("Int64").astype(str).radd("NHANES_")
    df["sex"] = df["sex"].map({1.0: "male", 2.0: "female"})
    df["race_ethnicity"] = df["race_ethnicity"].map({
        1.0: "mexican_american", 2.0: "other_hispanic", 3.0: "non_hispanic_white",
        4.0: "non_hispanic_black", 6.0: "non_hispanic_asian", 7.0: "other_or_multiracial",
    })
    df["target"] = (merged["DIQ010"] == 1.0).astype("int8").to_numpy()
    df = df.drop(columns=["DIQ010"])

    features = [column for column in df.columns if column not in {"patient_id", "target"}]
    panels = {
        "LOINC_85353-1_vital_signs_anthropometry": ["weight_kg", "height_cm", "bmi_kg_m2"],
        "LOINC_57021-8_CBC_with_auto_differential": [
            "wbc_10e9_l", "lymphocyte_pct", "monocyte_pct", "neutrophil_pct",
            "eosinophil_pct", "basophil_pct", "rbc_10e12_l", "hemoglobin_g_dl",
            "hematocrit_pct", "mcv_fl", "mch_pg", "mchc_g_dl", "rdw_pct",
            "platelets_10e9_l", "mpv_fl",
        ],
        "LOINC_24323-8_comprehensive_metabolic_panel": [
            "alt_u_l", "albumin_g_dl", "alp_u_l", "ast_u_l", "bun_mg_dl",
            "chloride_mmol_l", "creatinine_mg_dl", "glucose_mg_dl",
            "potassium_mmol_l", "sodium_mmol_l", "bilirubin_total_mg_dl",
            "calcium_mg_dl", "protein_total_g_dl",
        ],
        "LOINC_57698-3_lipid_panel": [
            "cholesterol_total_mg_dl", "triglycerides_mg_dl", "hdl_cholesterol_mg_dl"
        ],
    }
    groups = {
        "demographics": ["age_years", "sex", "race_ethnicity"],
        "anthropometry": ["weight_kg", "height_cm", "bmi_kg_m2", "waist_cm"],
        "white_cell_differential": [
            "wbc_10e9_l", "lymphocyte_pct", "monocyte_pct", "neutrophil_pct",
            "eosinophil_pct", "basophil_pct",
        ],
        "red_cell_indices": [
            "rbc_10e12_l", "hemoglobin_g_dl", "hematocrit_pct", "mcv_fl",
            "mch_pg", "mchc_g_dl", "rdw_pct",
        ],
        "platelet_indices": ["platelets_10e9_l", "mpv_fl"],
        "hepatic": [
            "alt_u_l", "ast_u_l", "alp_u_l", "ggt_u_l", "albumin_g_dl",
            "globulin_g_dl", "bilirubin_total_mg_dl", "protein_total_g_dl", "ldh_u_l",
        ],
        "renal_electrolyte": [
            "bun_mg_dl", "creatinine_mg_dl", "uric_acid_mg_dl", "sodium_mmol_l",
            "potassium_mmol_l", "chloride_mmol_l", "calcium_mg_dl",
            "phosphorus_mg_dl", "osmolality_mOsm_kg",
        ],
        "lipid": ["cholesterol_total_mg_dl", "triglycerides_mg_dl", "hdl_cholesterol_mg_dl"],
        "glycemic": ["glucose_mg_dl", "hba1c_pct"],
        "tissue_enzyme_iron": ["ck_u_l", "ldh_u_l", "iron_ug_dl"],
    }
    loinc = {feature: None for feature in features}
    loinc.update({
        "age_years": "30525-0", "sex": "46098-0", "weight_kg": "29463-7",
        "height_cm": "8302-2", "bmi_kg_m2": "39156-5", "waist_cm": "8280-0",
        "wbc_10e9_l": "6690-2", "lymphocyte_pct": "736-9", "monocyte_pct": "5905-5",
        "neutrophil_pct": "770-8", "eosinophil_pct": "713-8", "basophil_pct": "706-2",
        "rbc_10e12_l": "789-8", "hemoglobin_g_dl": "718-7",
        "hematocrit_pct": "4544-3", "mcv_fl": "787-2", "mch_pg": "785-6",
        "mchc_g_dl": "786-4", "rdw_pct": "788-0", "platelets_10e9_l": "777-3",
        "mpv_fl": "32623-1",
        "alt_u_l": "1742-6", "albumin_g_dl": "1751-7", "alp_u_l": "6768-6",
        "ast_u_l": "1920-8", "bun_mg_dl": "3094-0", "chloride_mmol_l": "2075-0",
        "creatinine_mg_dl": "2160-0", "glucose_mg_dl": "2345-7", "ggt_u_l": "2324-2",
        "iron_ug_dl": "2498-4", "ldh_u_l": "14804-9", "phosphorus_mg_dl": "2777-1",
        "potassium_mmol_l": "2823-3", "sodium_mmol_l": "2951-2",
        "bilirubin_total_mg_dl": "1975-2", "calcium_mg_dl": "17861-6",
        "cholesterol_total_mg_dl": "2093-3", "protein_total_g_dl": "2885-2",
        "triglycerides_mg_dl": "2571-8", "uric_acid_mg_dl": "3084-1",
        "hdl_cholesterol_mg_dl": "2085-9", "hba1c_pct": "4548-4",
    })
    graph = graph_from_sources(features, panels, groups, loinc)
    manifest = {
        "dataset": "NHANES_2017_2018_diabetes",
        "task": "binary physician-diagnosed diabetes status classification",
        "id_column": "patient_id", "target_column": "target",
        "positive_definition": "DIQ010 == 1 (Yes)",
        "negative_definition": "DIQ010 == 2 (No)",
        "excluded_labels": "DIQ010 in {3 borderline, 7 refused, 9 don't know} or missing",
        "cohort": "age >= 20 years",
        "categorical_features": ["sex", "race_ethnicity"],
        "continuous_features": [f for f in features if f not in {"sex", "race_ethnicity"}],
        "never_use_as_features": ["patient_id", "SEQN", "DIQ010", "all_other_DIQ_variables", "survey_weights"],
        "preprocessing_rule": "fit imputation, encoding, clipping, and scaling within each training fold only",
    }
    return df[["patient_id", *features, "target"]], manifest, graph


def build_mi() -> tuple[pd.DataFrame, dict, dict]:
    folder = RAW / "mi"
    metadata = json.loads((folder / "uci_metadata_579.json").read_text(encoding="utf-8"))["data"]
    names = [variable["name"] for variable in metadata["variables"]]
    raw = pd.read_csv(folder / "MI.data", header=None, names=names, na_values="?")

    # Strict admission-time snapshot: demographics/history, admission signs and ECG,
    # admission labs, and onset-to-hospital time. All treatments and post-admission
    # measurements are removed even when the source documentation permits some of them.
    selected = [names[index - 1] for index in [*range(2, 76), *range(83, 93)]]
    selected = [
        feature for feature in selected
        if feature not in {"IBS_NASL", "KFK_BLOOD", "GIPO_K", "GIPER_NA"}
    ]
    df = raw[["ID", *selected, "LET_IS"]].copy()
    df = df.rename(columns={"ID": "patient_id"})
    df["patient_id"] = df["patient_id"].astype("Int64").astype(str).radd("MI_")
    df["target"] = (df["LET_IS"] != 0).astype("int8")
    df = df.drop(columns="LET_IS")

    # Zero blood pressure is physiologically impossible and represents an invalid record.
    for column in ["S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT"]:
        df.loc[df[column] <= 0, column] = np.nan

    features = [column for column in df.columns if column not in {"patient_id", "target"}]
    panels = {
        "LOINC_85353-1_vital_signs": ["S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT"],
        "LOINC_24320-4_basic_metabolic_panel_components": ["K_BLOOD", "NA_BLOOD"],
        "LOINC_24325-3_hepatic_function_panel_components": ["ALT_BLOOD", "AST_BLOOD"],
    }
    groups = {
        "demographics": ["AGE", "SEX"],
        "cardiovascular_history": [
            "INF_ANAM", "STENOK_AN", "FK_STENOK", "IBS_POST", "GB", "SIM_GIPERT",
            "DLIT_AG", "ZSN_A",
        ],
        "arrhythmia_history": [f"nr_{suffix}" for suffix in ["11", "01", "02", "03", "04", "07", "08"]],
        "conduction_history": [f"np_{suffix}" for suffix in ["01", "04", "05", "07", "08", "09", "10"]],
        "endocrine_history": ["endocr_01", "endocr_02", "endocr_03"],
        "pulmonary_history": ["zab_leg_01", "zab_leg_02", "zab_leg_03", "zab_leg_04", "zab_leg_06"],
        "admission_blood_pressure": ["S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT"],
        "admission_acute_state": ["O_L_POST", "K_SH_POST", "MP_TP_POST", "SVT_POST", "GT_POST", "FIB_G_POST"],
        "mi_location": ["ant_im", "lat_im", "inf_im", "post_im", "IM_PG_P"],
        "admission_ecg_rhythm": [f"ritm_ecg_p_{suffix}" for suffix in ["01", "02", "04", "06", "07", "08"]],
        "admission_ecg_arrhythmia": [f"n_r_ecg_p_{suffix}" for suffix in ["01", "02", "03", "04", "05", "06", "08", "09", "10"]],
        "admission_ecg_conduction": [f"n_p_ecg_p_{suffix}" for suffix in ["01", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]],
        "electrolytes": ["K_BLOOD", "NA_BLOOD"],
        "hepatic_enzymes": ["ALT_BLOOD", "AST_BLOOD"],
        "inflammation_hematology": ["L_BLOOD", "ROE"],
    }
    loinc = {feature: None for feature in features}
    loinc.update({
        "AGE": "30525-0", "SEX": "46098-0", "S_AD_KBRIG": "8480-6",
        "D_AD_KBRIG": "8462-4", "S_AD_ORIT": "8480-6", "D_AD_ORIT": "8462-4",
        "K_BLOOD": "2823-3", "NA_BLOOD": "2951-2", "ALT_BLOOD": "1742-6",
        "AST_BLOOD": "1920-8", "L_BLOOD": "6690-2", "ROE": "4537-7",
    })
    graph = graph_from_sources(features, panels, groups, loinc)
    manifest = {
        "dataset": "UCI_myocardial_infarction_complications",
        "task": "binary lethal-outcome classification after myocardial infarction",
        "id_column": "patient_id", "target_column": "target",
        "positive_definition": "LET_IS in {1,...,7}", "negative_definition": "LET_IS == 0",
        "prediction_time": "hospital/ICU admission",
        "continuous_features": [
            "AGE", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT",
            "K_BLOOD", "NA_BLOOD", "ALT_BLOOD", "AST_BLOOD", "L_BLOOD", "ROE",
        ],
        "categorical_features": [
            feature for feature in features
            if feature not in {
                "AGE", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT",
                "K_BLOOD", "NA_BLOOD", "ALT_BLOOD", "AST_BLOOD", "L_BLOOD", "ROE",
            }
        ],
        "dropped_for_extreme_missingness": ["IBS_NASL", "KFK_BLOOD"],
        "dropped_as_deterministic_lab_flags": ["GIPO_K", "GIPER_NA"],
        "never_use_as_features": [
            "patient_id", "ID", "LET_IS", "FIBR_PREDS", "PREDS_TAH", "JELUD_TAH",
            "FIBR_JELUD", "A_V_BLOK", "OTEK_LANC", "RAZRIV", "DRESSLER", "ZSN",
            "REC_IM", "P_IM_STEN", "all_treatment_columns_76_to_82_and_93_to_112",
        ],
        "preprocessing_rule": "fit imputation, encoding, clipping, and scaling within each training fold only",
    }
    return df[["patient_id", *features, "target"]], manifest, graph


def build_support2() -> tuple[pd.DataFrame, dict, dict]:
    raw = pd.read_csv(RAW / "support2" / "support2_raw.csv")
    features = [
        "age", "sex", "dzgroup", "num.co", "race", "hday", "diabetes", "dementia", "ca",
        "meanbp", "wblc", "hrt", "resp", "temp", "pafi", "alb", "bili", "crea", "sod",
        "ph", "glucose", "bun", "urine", "adlp", "adls",
    ]
    df = raw[["id", *features, "hospdead"]].copy()
    df = df.rename(columns={"id": "patient_id", "hospdead": "target"})
    df["patient_id"] = df["patient_id"].astype("Int64").astype(str).radd("SUPPORT2_")
    df["target"] = df["target"].astype("int8")
    for column in ["sex", "dzgroup", "race", "ca"]:
        df[column] = df[column].astype("string").str.strip().str.lower().replace({"missing": pd.NA})

    panels = {
        "LOINC_85353-1_vital_signs": ["meanbp", "hrt", "resp", "temp"],
        "LOINC_24323-8_comprehensive_metabolic_panel_components": [
            "alb", "bili", "crea", "sod", "glucose", "bun"
        ],
        "LOINC_24325-3_hepatic_function_panel_components": ["alb", "bili"],
    }
    groups = {
        "demographics": ["age", "sex", "race"],
        "disease_burden": ["dzgroup", "num.co", "diabetes", "dementia", "ca"],
        "hemodynamics": ["meanbp", "hrt"],
        "respiratory": ["resp", "pafi", "ph"],
        "hepatic": ["alb", "bili"],
        "renal_metabolic": ["crea", "sod", "glucose", "bun", "urine"],
        "functional_status": ["adlp", "adls"],
    }
    loinc = {feature: None for feature in features}
    loinc.update({
        "age": "30525-0", "sex": "46098-0", "meanbp": "8478-0", "wblc": "6690-2",
        "hrt": "8867-4", "resp": "9279-1", "temp": "8310-5", "alb": "1751-7",
        "bili": "1975-2", "crea": "2160-0", "sod": "2951-2", "ph": "2744-1",
        "glucose": "2345-7", "bun": "3094-0",
    })
    graph = graph_from_sources(features, panels, groups, loinc)
    manifest = {
        "dataset": "UCI_SUPPORT2",
        "task": "binary in-hospital mortality classification",
        "id_column": "patient_id", "target_column": "target",
        "positive_definition": "hospdead == 1", "negative_definition": "hospdead == 0",
        "prediction_time": "SUPPORT study entry/day-3 clinical snapshot",
        "categorical_features": ["sex", "dzgroup", "race", "ca"],
        "continuous_features": [f for f in features if f not in {"sex", "dzgroup", "race", "ca"}],
        "never_use_as_features": [
            "patient_id", "id", "death", "hospdead", "slos", "d.time", "sfdm2",
            "sps", "aps", "surv2m", "surv6m", "prg2m", "prg6m", "scoma",
            "charges", "totcst", "totmcst", "avtisst", "dnr", "dnrday",
        ],
        "preprocessing_rule": "fit imputation, encoding, clipping, and scaling within each training fold only",
    }
    return df[["patient_id", *features, "target"]], manifest, graph


def validate_and_summarize(name: str, df: pd.DataFrame, manifest: dict, graph: dict) -> dict:
    id_column = manifest["id_column"]
    target_column = manifest["target_column"]
    features = [column for column in df.columns if column not in {id_column, target_column}]
    assert df[id_column].notna().all()
    assert not df[id_column].duplicated().any()
    assert set(df[target_column].unique()) <= {0, 1}
    assert graph["nodes"] == features
    assert not set(manifest["never_use_as_features"]) & set(features)
    missing = df[features].isna().mean().sort_values(ascending=False)
    return {
        "dataset": name,
        "rows": int(len(df)), "features": len(features),
        "positive": int(df[target_column].sum()),
        "negative": int((df[target_column] == 0).sum()),
        "positive_rate": float(df[target_column].mean()),
        "duplicate_ids": int(df[id_column].duplicated().sum()),
        "duplicate_rows_excluding_id": int(df.drop(columns=id_column).duplicated().sum()),
        "max_feature_missing_rate": float(missing.iloc[0]),
        "features_over_50pct_missing": missing[missing > 0.5].index.tolist(),
        "all_missing_features": missing[missing == 1.0].index.tolist(),
        "graph_nodes": graph["counts"]["nodes"],
        "graph_edges": graph["counts"]["E_union"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.mkdir(parents=True, exist_ok=True)
    GRAPH_OUT.mkdir(parents=True, exist_ok=True)
    builders = {
        "nhanes_diabetes": build_nhanes,
        "mi_mortality": build_mi,
        "support2_mortality": build_support2,
    }
    summaries = []
    for name, builder in builders.items():
        df, manifest, graph = builder()
        csv_path = OUT / f"{name}.csv"
        manifest_path = MANIFEST_OUT / f"{name}_manifest.json"
        graph_path = GRAPH_OUT / f"{name}_graph.json"
        df.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
        write_json(manifest_path, manifest)
        write_json(graph_path, graph)
        summary = validate_and_summarize(name, df, manifest, graph)
        summary["csv_sha256"] = sha256(csv_path)
        summaries.append(summary)

    raw_files = sorted(path for path in RAW.rglob("*") if path.is_file())
    write_json(OUT / "dataset_validation_summary.json", {
        "standardization_status": "not globally standardized; fit within each training fold",
        "datasets": summaries,
        "raw_file_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in raw_files},
    })
    for summary in summaries:
        print(
            f"{summary['dataset']}: rows={summary['rows']}, features={summary['features']}, "
            f"positive={summary['positive']} ({summary['positive_rate']:.2%}), "
            f"graph_edges={summary['graph_edges']}"
        )


if __name__ == "__main__":
    main()
