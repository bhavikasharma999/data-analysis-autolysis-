# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "os",
# ]
# ///

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Configure OpenAI API token
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

openai.api_key = AIPROXY_TOKEN

def load_data(filename):
    """Load and validate the dataset."""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

def generate_visualizations(df, output_dir):
    """Generate up to 3 relevant visualizations based on the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn")

    # Generic Visualization 1: Correlation heatmap for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

    # Visualization 2: Distribution of numeric columns
    for col in numeric_cols[:3]:  # Limit to first 3 columns for brevity
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], bins=20, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"distribution_{col}.png"))
        plt.close()

    # Visualization 3: Bar plot for top categories in a categorical column
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if categorical_cols.any():
        top_cat_col = categorical_cols[0]
        top_categories = df[top_cat_col].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_categories.values, y=top_categories.index)
        plt.title(f"Top 10 Categories in {top_cat_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"top_categories_{top_cat_col}.png"))
        plt.close()

def analyze_with_llm(df, filename):
    """Leverage GPT-4o-Mini for dataset analysis."""
    data_summary = {
        "filename": filename,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": df.head(3).to_dict(),
    }

    prompt = f"""
    The following dataset has been provided for analysis:
    {json.dumps(data_summary)}

    Based on this summary, provide:
    1. An overview of the dataset.
    2. Key insights derived from the data.
    3. Recommendations for further analysis or actions.

    Respond in Markdown format.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        sys.exit(1)

def save_analysis(content, output_path):
    """Save the LLM-generated analysis to README.md."""
    with open(output_path, "w") as f:
        f.write(content)

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = os.path.splitext(input_file)[0]
    readme_path = os.path.join(output_dir, "README.md")

    print("Loading dataset...")
    df = load_data(input_file)

    print("Generating visualizations...")
    generate_visualizations(df, output_dir)

    print("Analyzing data with GPT-4o-Mini...")
    analysis = analyze_with_llm(df, input_file)

    print("Saving analysis to README.md...")
    save_analysis(analysis, readme_path)

    print(f"Analysis complete. Output saved in {output_dir}/")

if __name__ == "__main__":
    main()

