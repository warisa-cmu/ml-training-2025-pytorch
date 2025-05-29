import cv2
import os

# Change this
CLASS_LIST = ["class_1", "class_2", "class_3"]
BASE_PATH = "src/T04_cnn_models/dataset"


def get_user_choice(options, prompt):
    print(f"\n{prompt}")
    for idx, option in enumerate(options):
        print(f" {idx + 1}. {option}")
    while True:
        choice = input("Enter your choice number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print("Invalid choice. Try again.")


def main():
    print("=== Webcam Image Collector ===")
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    while True:
        # User chooses class
        CLASS = get_user_choice(CLASS_LIST, "Select class:")

        # Setup
        SAVE_DIR_TRAIN = os.path.join(BASE_PATH, "train", CLASS)
        SAVE_DIR_VAL = os.path.join(BASE_PATH, "val", CLASS)
        os.makedirs(SAVE_DIR_TRAIN, exist_ok=True)
        os.makedirs(SAVE_DIR_VAL, exist_ok=True)

        # Start video capture
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Cannot access camera. Try restarting or checking permissions.")
            return

        # Automatically index filenames to avoid overwriting
        existing_train = [f for f in os.listdir(SAVE_DIR_TRAIN) if f.endswith(".jpg")]
        count_train = len(existing_train)
        existing_val = [f for f in os.listdir(SAVE_DIR_VAL) if f.endswith(".jpg")]
        count_val = len(existing_val)

        print(f"\nCollecting images for: {CLASS}")
        print("Press SPACE to capture, 'n' for new class, 'q' to quit.")

        idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            cv2.imshow("Webcam - Space=Capture, n=New class, q=Quit", frame)
            key = cv2.waitKey(1)

            if key == ord(" "):  # SPACE pressed
                if idx % 3 > 0:  # Training data
                    img_name = f"{CLASS}_{count_train:04d}.jpg"
                    img_path = os.path.join(SAVE_DIR_TRAIN, img_name)
                    cv2.imwrite(img_path, frame)
                    print(f"Saved Train Data: {img_path}")
                    count_train += 1
                else:  # Validation data
                    img_name = f"{CLASS}_{count_val:04d}.jpg"
                    img_path = os.path.join(SAVE_DIR_VAL, img_name)
                    cv2.imwrite(img_path, frame)
                    print(f"Saved Val Data: {img_path}")
                    count_val += 1
                idx += 1

            elif key == ord("n"):  # Move to new class/split
                break
            elif key == ord("q"):  # Quit collecting
                cap.release()
                cv2.destroyAllWindows()
                print("Exiting...")
                return

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
