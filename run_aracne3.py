
import subprocess
import os 
#ARACNe3 C++ Code
def run_aracne3_app(exp_mat, output_folder, regulators, subnets):
    #Run the ARACNe3 executable.
    """
    Parameters:
        base_dir (str): Base directory where ARACNe3 is installed.
        output_folder (str): The folder where ARACNe3 should write its output.
        regulators (str): Path to a txt file containing all the regulators.
        subnets (str): Number of subnets you want to generate.
    """

    # Since the build directory is one level up from the current folder, build the path accordingly:
    script_dir = "/shares/vasciaveo_lab/programs/ARACNe3"
    app_exe = os.path.join(script_dir, "build", "src", "app", "ARACNe3_app_release")

    cmd = [
        app_exe,
        "-e", exp_mat,
        "-r", regulators,
        "-o", output_folder,
        "-x", subnets,
        "--alpha", "0.05",
        "--threads", "32" #Change threads 
    ]
    print("Running ARACNe3 app:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

        #Run the C++ ARACNe3 executable

regulator_path = "/shares/vasciaveo_lab/aarulselvan/arachne/human_tf_cotf_plus_sig_surf.txt"

network_path = "/shares/vasciaveo_lab/aarulselvan/arachne/NEWPAtables/lung1sample1/"
os.makedirs(network_path, exist_ok=True)

run_aracne3_app("/shares/vasciaveo_lab/data/adhiban_scRNAseq_datasets/lung/lung_reimagined/epithelial/lung_epithelial_metacells_FINAL_updated.tsv", network_path, regulator_path, "100")

print()
