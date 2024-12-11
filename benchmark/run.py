import subprocess
import os

# Define the benchmark folder
benchmark_folder = os.path.dirname(os.path.abspath(__file__))

# Define the project directory (parent of the benchmark folder)
project_directory = os.path.dirname(benchmark_folder)

# List of scripts to run (located in the benchmark folder)
scripts = [
    os.path.join(benchmark_folder, "benchmark331Single.py"),
    os.path.join(benchmark_folder, "benchmark331Multiple.py"),
    os.path.join(benchmark_folder, "benchmark331Combined.py"),
    os.path.join(benchmark_folder, "benchmark311Single.py"),
    os.path.join(benchmark_folder, "benchmark311Multiple.py"),
    os.path.join(benchmark_folder, "benchmark311Combined.py")
]

# Data path to pass to each script (located in project_directory)
data_path = os.path.join(project_directory, "data")

# Model path to pass to each script
model_path = "OpenGVLab/InternVL2-1B"

# Result directory (inside the benchmark folder)
result_directory = os.path.join(benchmark_folder, "result")
os.makedirs(result_directory, exist_ok=True)  # Create the result directory if it doesn't exist

# Log and result file paths
log_file_path = os.path.join(result_directory, "log.txt")
result_file_path = os.path.join(result_directory, "result.txt")

# Keywords to identify result lines
result_keywords = [
    "True Positives:",
    "False Positives:",
    "True Negatives:",
    "False Negatives:",
    "Accuracy:",
    "Precision:",
    "Recall:",
    "F1 Score:"
]

# Run each script sequentially
with open(log_file_path, "w") as log_file, open(result_file_path, "w") as result_file:
    for script in scripts:
        # Header for the current script section
        header = f"Dataset: {data_path}\nModel: {model_path}\nScript: {script}\n"

        print(f"Running {script} with data path: {data_path} and model path: {model_path}...")
        log_file.write(f"Running {script} with data path: {data_path} and model path: {model_path}...\n")

        try:
            # Pass the data path and model path as arguments to the script
            result = subprocess.run(
                ["python", script, data_path, model_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Write full output and errors to the log file
            log_file.write(result.stdout)
            if result.stderr:
                log_file.write(f"Errors from {script}:\n{result.stderr}")

            # Extract and save specific results to the result file
            results_found = False
            result_file.write(header)
            for line in result.stdout.splitlines():
                if any(keyword in line for keyword in result_keywords):
                    result_file.write(line + "\n")
                    results_found = True

            # Add a separator if results were found
            if results_found:
                result_file.write("\n" + "-" * 50 + "\n\n")

        except Exception as e:
            error_message = f"Failed to run {script}: {e}\n"
            log_file.write(error_message)
            print(error_message)

# Inform the user where logs and results are saved
print(f"Logs saved to {log_file_path}")
print(f"Results saved to {result_file_path}")
