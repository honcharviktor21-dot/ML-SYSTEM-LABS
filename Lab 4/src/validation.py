import os

def validate_data(df, config):
    print("--- 2. Validation & Label-Quality Step ---")
    ranges = config['pipeline']['schema']['ranges']
    report_lines = ["=== LAB 4 DATA VALIDATION REPORT ===\n"]

    violations = 0
    for col, bounds in ranges.items():
        out_of_bounds = df[(df[col] < bounds[0]) | (df[col] > bounds[1])]
        if not out_of_bounds.empty:
            report_lines.append(f"Range violation in {col}: {len(out_of_bounds)} rows out of bounds {bounds}")
            violations += 1
            
    if violations == 0:
        report_lines.append("Schema Range Checks: ALL PASSED.")

    feature_cols = df.columns.drop('income')
    duplicates = df[df.duplicated(subset=feature_cols, keep=False)]
    conflicts = duplicates.groupby(list(feature_cols))['income'].nunique()
    conflict_count = (conflicts > 1).sum()
    report_lines.append(f"\nLabel Quality Audit: Found {conflict_count} duplicate feature rows with conflicting labels.")

    report_str = "\n".join(report_lines)
    print(report_str + "\n")

    os.makedirs(os.path.dirname(config['pipeline']['paths']['validation_report']), exist_ok=True)
    with open(config['pipeline']['paths']['validation_report'], 'w') as f:
        f.write(report_str)