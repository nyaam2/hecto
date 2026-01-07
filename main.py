# main.py
from config import (
    DEVICE,
    MODEL_ID,
    NUM_FRAMES,
    TARGET_SIZE,
    TEST_DIR,
    OUT_CSV,
    SAMPLE_SUBMISSION,
    ensure_dirs,
    set_seed,
)
from model_utils import load_model
from predict import predict_directory
from make_submission import make_submission, save_submission


def main():
    ensure_dirs()
    set_seed()

    print(f"Device: {DEVICE}")
    print("Loading model...")
    model, processor = load_model(MODEL_ID, DEVICE)

    print(f"Model loaded: {MODEL_ID}")
    print(f"Model config: num_labels={model.config.num_labels}")
    if hasattr(model.config, "id2label"):
        print(f"id2label: {model.config.id2label}")

    results = predict_directory(
        test_dir=TEST_DIR,
        num_frames=NUM_FRAMES,
        target_size=TARGET_SIZE,
        model=model,
        processor=processor,
        device=DEVICE,
    )

    submission_df = make_submission(SAMPLE_SUBMISSION, results)
    save_submission(submission_df, OUT_CSV)


if __name__ == "__main__":
    main()
