# generate_report.py
# Run after evaluate_model.py to make tables for the report

import json
import os
import glob
from datetime import datetime

RESULTS_DIR = "./results/"


def load_all_results():
    """Load all json result files"""
    results = []
    for f in glob.glob(RESULTS_DIR + "*_results.json"):
        with open(f, 'r') as file:
            results.append(json.load(file))
    return results


def make_latex_table(results):
    """Generate latex table"""
    
    latex = """
\\begin{table}[h]
\\centering
\\caption{Model Performance on Test Set}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
Model & Accuracy & Precision & Recall & F1 \\\\
\\hline
"""
    
    for r in results:
        macro = r["classification_report"]["macro avg"]
        name = r['model_name'].replace('_', ' ')
        latex += f"{name} & {r['test_accuracy']:.4f} & "
        latex += f"{macro['precision']:.4f} & {macro['recall']:.4f} & "
        latex += f"{macro['f1-score']:.4f} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    return latex


def make_markdown_report(results):
    """Make markdown version"""
    
    md = "# Sentiment Analysis Results\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    # Main table
    md += "## Overall Results\n\n"
    md += "| Model | Accuracy | Precision | Recall | F1 |\n"
    md += "|-------|----------|-----------|--------|----|\n"
    
    for r in results:
        macro = r["classification_report"]["macro avg"]
        md += f"| {r['model_name']} | {r['test_accuracy']:.4f} | "
        md += f"{macro['precision']:.4f} | {macro['recall']:.4f} | "
        md += f"{macro['f1-score']:.4f} |\n"
    
    md += "\n"
    
    # Per class breakdown
    md += "## Per-Class Results\n\n"
    for r in results:
        md += f"### {r['model_name']}\n\n"
        md += "| Class | Precision | Recall | F1 |\n"
        md += "|-------|-----------|--------|----|\n"
        
        for cls in ['Negative', 'Neutral', 'Positive']:
            if cls in r["classification_report"]:
                c = r["classification_report"][cls]
                md += f"| {cls} | {c['precision']:.4f} | "
                md += f"{c['recall']:.4f} | {c['f1-score']:.4f} |\n"
        md += "\n"
    
    # Confusion matrices
    md += "## Confusion Matrices\n\n"
    for r in results:
        md += f"### {r['model_name']}\n\n"
        md += "```\n"
        md += "            Pred\n"
        md += "         Neg Neu Pos\n"
        cm = r["confusion_matrix"]
        md += f"True Neg {cm[0][0]:4} {cm[0][1]:4} {cm[0][2]:4}\n"
        md += f"     Neu {cm[1][0]:4} {cm[1][1]:4} {cm[1][2]:4}\n"
        md += f"     Pos {cm[2][0]:4} {cm[2][1]:4} {cm[2][2]:4}\n"
        md += "```\n\n"
    
    # Training info
    md += "## Dataset Info\n\n"
    if results:
        r = results[0]
        md += f"- Training samples: {r['train_samples']}\n"
        md += f"- Test samples: {r['test_samples']}\n"
        md += f"- Validation samples: {r['val_samples']}\n"
        md += f"- Epochs: {r['num_epochs']}\n"
    
    return md


def main():
    print("Generating report tables...")
    
    results = load_all_results()
    
    if not results:
        print("No results found! Run evaluate_model.py first.")
        return
    
    print(f"Found {len(results)} models")
    
    # Make latex
    latex = make_latex_table(results)
    with open(RESULTS_DIR + "table.tex", 'w') as f:
        f.write(latex)
    print(f"Saved LaTeX table to {RESULTS_DIR}table.tex")
    
    # Make markdown
    md = make_markdown_report(results)
    with open(RESULTS_DIR + "REPORT.md", 'w') as f:
        f.write(md)
    print(f"Saved markdown report to {RESULTS_DIR}REPORT.md")
    
    # Print quick summary
    print("\n--- Quick Summary ---")
    best = max(results, key=lambda x: x['test_accuracy'])
    print(f"Best model: {best['model_name']} with {best['test_accuracy']:.4f} accuracy")
    
    for r in results:
        print(f"  {r['model_name']}: {r['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
