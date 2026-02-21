from biomarker import main as biomarker_main
from mri import main as mri_main

def main(): # main operator to run multimodal or single modality

    print("Select the function to run:")
    print("1. MRI Classification")
    print("2. Biomarker Risk Prediction")
    
    choice = input("Enter 1 or 2: ")
    
    if choice == '1':
        mri_main()
    elif choice == '2':
        iterations = int(input("Enter number of iterations: "))
        runs = int(input("Enter number of runs per iteration: "))
        biomarker_main(iterations, runs)
    else:
        print("Invalid choice. Please enter 1 or 2.")
