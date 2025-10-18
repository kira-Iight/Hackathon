# main.py
"""
Main entry point for training plant species classifier.
"""

from pipeline import train_pipeline


if __name__ == "__main__":
    # === CONFIGURE YOUR RUN HERE ===
    labels_csv = "data/tree_labels.csv"
    plant_type = "tree"

    print(f"Launching training for '{plant_type}' using labels from '{labels_csv}'")
    model, history = train_pipeline(labels_csv, plant_type)

    print("\nTraining complete.")
    print("You can now inspect the saved checkpoint in ./checkpoints/")
