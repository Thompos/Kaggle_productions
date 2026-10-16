from data import load_data
from data_prep import prepare_data


def main():
    train, test = load_data()

    print("Training data shape:", train.shape)
    print("Test data shape:", test.shape)

    print("\nTraining data preview:")
    print(train.head(4))

    train_clean, test_clean = prepare_data(train, test)

    print("\nCleaned training data shape:", train_clean.shape)
    print("Cleaned test data shape:", test_clean.shape)


if __name__ == "__main__":
    main()
    