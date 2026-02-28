import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def find_local_dataset(candidates=None):
    if candidates is None:
        candidates = ['hospital_deterioration_ml_ready.csv',]
    from pathlib import Path
    for fn in candidates:
        p = Path(fn)
        if p.exists():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description='Dataset suitability checker')
    parser.add_argument('--path', '-p', help='Path to CSV dataset', default=None)
    parser.add_argument('--target', '-t', help='Target column name (optional)', default=None)
    parser.add_argument('--patient-col', '-c', help='Patient ID column name (optional)', default=None)
    args = parser.parse_args()

    file_path = args.path or find_local_dataset()
    if file_path is None:
        print('ERROR: No dataset path provided and no local candidate found. Set --path or place a known CSV file in the folder.')
        sys.exit(1)

    df = pd.read_csv(file_path)

    print(f"\n✅ Dataset Loaded Successfully: {file_path}")
    print('Shape:', df.shape)

    print('\n📌 Dataset Info:')
    print(df.info())

    # missing value report
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_percent
    })
    print('\n📌 Missing Value Report:')
    print(missing_report[missing_report['Missing Count'] > 0])

    # detect target
    requested_target = args.target or 'deterioration_next_12h'
    target_candidates = [requested_target, 'SepsisLabel', 'sepsis', 'label', 'deterioration_next_12h']
    target = None
    for c in target_candidates:
        if c in df.columns:
            target = c
            break

    if target is None:
        print('\nERROR: No target column found. Looked for:', target_candidates)
        print('Please provide the correct target via --target or rename the column.')
        sys.exit(1)

    print(f"\n📌 Using target column: '{target}'")

    # Target distribution
    print('\n📌 Target Variable Distribution:')
    try:
        print(df[target].value_counts(dropna=False))
        print('\n📌 Target Percentage:')
        print(df[target].value_counts(normalize=True, dropna=False) * 100)
    except Exception as e:
        print('Could not compute target distribution:', e)

    # numeric vs categorical
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude target from features list
    feature_cols = [c for c in num_features if c != target]
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print('\n✅ Numerical Features:', feature_cols)
    print('\n✅ Categorical Features:', cat_features)

    # detect patient id column
    patient_col = args.patient_col
    if patient_col is None:
        patient_candidates = ['Patient_ID', 'patient_id', 'subject_id', 'id', 'ID', 'PatientId']
        for pc in patient_candidates:
            if pc in df.columns:
                patient_col = pc
                break

    if patient_col is None:
        print('\n⚠ WARNING: No patient identifier column detected. Pass --patient-col to specify one.')
    else:
        print(f"\n📌 Using patient id column: '{patient_col}'")

        # compute unique sepsis patients and related stats
        try:
            # create boolean mask for positive label
            if pd.api.types.is_numeric_dtype(df[target]):
                positive_mask = df[target] == 1
            else:
                positive_mask = df[target].astype(str).str.lower().isin(['1', 'true', 'yes', 'y', 'positive'])

            total_unique_patients = df[patient_col].nunique()
            sepsis_unique_patients = df.loc[positive_mask, patient_col].nunique()
            sepsis_rows = int(positive_mask.sum())
            total_rows = len(df)

            print('\n📌 Per-patient Sepsis Summary:')
            print(f'Total unique patients: {total_unique_patients}')
            print(f'Unique sepsis-positive patients: {sepsis_unique_patients}')
            if total_unique_patients > 0:
                print(f'Sepsis patient percentage: {sepsis_unique_patients / total_unique_patients * 100:.2f}%')
            print(f'Total rows: {total_rows}')
            print(f'Sepsis rows: {sepsis_rows} ({sepsis_rows / total_rows * 100:.2f}% of rows)')
        except Exception as e:
            print('Could not compute per-patient sepsis stats:', e)

    # correlation with target (only if target numeric)
    print('\n📌 Feature Correlation with Target:')
    if pd.api.types.is_numeric_dtype(df[target]):
        # compute correlation of numeric features with target
        if len(feature_cols) == 0:
            print('No numeric features available to compute correlations.')
            target_corr = pd.Series(dtype=float)
        else:
            try:
                target_corr = df[feature_cols].corrwith(df[target]).sort_values(ascending=False)
                print(target_corr)
            except Exception as e:
                print('Could not compute correlations:', e)
                target_corr = pd.Series(dtype=float)
    else:
        print('Target is not numeric; skipping numeric correlation analysis.')
        target_corr = pd.Series(dtype=float)

    # data leakage check
    suspicious_features = ['sepsis_risk_score']
    print('\n⚠ Data Leakage Check:')
    for col in suspicious_features:
        if col in df.columns:
            print(f'⚠ WARNING: {col} may cause data leakage')

    # final suitability score
    suitability_score = 0
    if df.shape[0] > 5000:
        suitability_score += 2
    if len(feature_cols) >= 10:
        suitability_score += 2
    # binary target check
    try:
        if df[target].nunique() == 2:
            suitability_score += 2
    except Exception:
        pass
    if missing.max() < 0.25 * len(df):
        suitability_score += 2
    try:
        if not target_corr.empty and target_corr.abs().max() > 0.1:
            suitability_score += 2
    except Exception:
        pass

    print('\n✅ FINAL SUITABILITY SCORE (OUT OF 10):', suitability_score)
    if suitability_score >= 8:
        print('🔥 EXCELLENT DATASET FOR ML TRAINING')
    elif suitability_score >= 6:
        print('✅ GOOD DATASET WITH MINOR CLEANING NEEDED')
    else:
        print('⚠ DATASET NEEDS MAJOR IMPROVEMENTS')

    # optional: class balance plot (safely)
    try:
        df[target].value_counts().plot(kind='bar')
        plt.title('Target Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.show()
    except Exception:
        pass
if __name__ == '__main__':
    main()
