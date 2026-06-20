import gradio as gr
import subprocess
import os

def rank_candidates(file_path):
    if not file_path:
        return None, "Please upload a candidates.jsonl file."
    
    out_csv = "artifacts/sandbox_submission.csv"
    out_features = "artifacts/sandbox_features.parquet"
    
    # Run the pipeline
    try:
        # 1. Feature Engineering
        subprocess.run([
            "python", "src/precompute/engineer_features.py",
            "--candidates", file_path,
            "--taxonomy", "artifacts/skill_taxonomy.json",
            "--jd", "job_requirements.yaml",
            "--out", out_features
        ], check=True, capture_output=True, text=True)
        
        # 2. Ranking
        rank_process = subprocess.run([
            "python", "src/rank.py",
            "--candidates", file_path,
            "--features", out_features,
            "--out", out_csv
        ], check=True, capture_output=True, text=True)
        
        if os.path.exists(out_csv):
            return out_csv, "Ranking complete!\n\nLogs:\n" + rank_process.stdout
        else:
            return None, "Failed to generate submission.csv"
            
    except subprocess.CalledProcessError as e:
        return None, f"Pipeline failed:\n{e.stderr}\n{e.stdout}"

demo = gr.Interface(
    fn=rank_candidates,
    inputs=gr.File(label="Upload candidates.jsonl", file_types=[".jsonl"]),
    outputs=[
        gr.File(label="Download Ranked submission.csv"),
        gr.Textbox(label="Execution Logs", lines=15)
    ],
    title="Redrob Ranker Sandbox",
    description="Upload a small sample of candidates.jsonl (≤100) to receive a ranked submission.csv."
)

if __name__ == "__main__":
    demo.launch(share=True)
