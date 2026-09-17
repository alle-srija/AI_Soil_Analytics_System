import json
from pathlib import Path

import pandas as pd

base = Path(r'c:\Users\A.NAVYAMANI\Desktop\AI Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory')
dataset_path = base / 'Samples EJP Probefield.xlsx'
summary_path = base / 'milestone1_summary.json'

if not dataset_path.exists():
    raise FileNotFoundError(f'Dataset not found: {dataset_path}')

df = pd.read_excel(dataset_path)

# Basic validation checks
numeric_cols = df.select_dtypes(include='number').columns.tolist()
missing = df.isna().sum()
summary = {
    'project': 'AI-Powered Soil Analytics System for Nutrient Assessment and Intelligent Crop Advisory',
    'milestone': 'Milestone 1',
    'dataset_path': str(dataset_path),
    'row_count': int(df.shape[0]),
    'column_count': int(df.shape[1]),
    'columns': df.columns.tolist(),
    'numeric_columns': numeric_cols,
    'missing_values': {str(k): int(v) for k, v in missing.items()},
    'dataset_fit': 'Suitable' if len(numeric_cols) > 0 else 'Not suitable',
    'head': df.head(10).to_dict(orient='records'),
}

# Create a small exploratory analysis report
report = {
    'data_quality': {
        'rows': summary['row_count'],
        'columns': summary['column_count'],
        'missing_entries': int(missing.sum()),
        'numeric_feature_count': len(numeric_cols),
    },
    'suitability': {
        'status': summary['dataset_fit'],
        'reason': 'The dataset contains numeric agronomic/soil variables needed for nutrient analysis and crop advisory modeling.'
    },
    'top_columns': summary['columns'][:10],
    'sample_preview': summary['head'][:5],
}

summary_path.write_text(json.dumps({'summary': summary, 'report': report}, indent=2), encoding='utf-8')

print('Milestone 1 analysis complete.')
print(json.dumps({
    'row_count': summary['row_count'],
    'column_count': summary['column_count'],
    'numeric_feature_count': len(numeric_cols),
    'dataset_fit': summary['dataset_fit'],
    'missing_entries': int(missing.sum())
}, indent=2))
print('\nFirst rows of the dataset:')
print(df.head(10).to_string(index=False))
