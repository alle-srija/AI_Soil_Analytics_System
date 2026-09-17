import json
from pathlib import Path

import pandas as pd

base = Path(r'c:\Users\A.NAVYAMANI\Desktop\AI Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory')

dataset_path = base / 'Samples EJP Probefield.xlsx'
report_path = base / 'milestone1_exact_report.json'

if not dataset_path.exists():
    raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

# Load dataset
soil_df = pd.read_excel(dataset_path)

# Validate dataset for soil nutrient analytics
numeric_df = soil_df.select_dtypes(include='number')
missing_values = soil_df.isna().sum()
missing_total = int(missing_values.sum())

report = {
    'project_name': 'AI-Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory',
    'milestone': 'Milestone 1',
    'dataset_file': str(dataset_path),
    'rows': int(soil_df.shape[0]),
    'columns': int(soil_df.shape[1]),
    'column_names': soil_df.columns.tolist(),
    'numeric_feature_count': int(numeric_df.shape[1]),
    'missing_total': missing_total,
    'missing_values_by_column': {str(k): int(v) for k, v in missing_values.items()},
    'sample_rows': soil_df.head(5).to_dict(orient='records'),
    'dataset_fit_for_milestone': 'Suitable' if numeric_df.shape[1] > 0 else 'Not suitable',
    'milestone_goal': 'Validate the dataset and confirm it is usable for nutrient assessment and crop advisory analysis.'
}

report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

print(json.dumps({
    'project_name': report['project_name'],
    'milestone': report['milestone'],
    'rows': report['rows'],
    'columns': report['columns'],
    'numeric_feature_count': report['numeric_feature_count'],
    'missing_total': report['missing_total'],
    'dataset_fit_for_milestone': report['dataset_fit_for_milestone']
}, indent=2))
print('\nReport saved to:', report_path)
