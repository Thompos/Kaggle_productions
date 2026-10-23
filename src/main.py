from operator import truediv
from data import (
    load_data,
    save_processed_data,
    load_processed_data,
)
from data_prep import prepare_data1, prepare_data2

REBUILD_DATA = False


def main():
    
    temp = REBUILD_DATA
    
    if not temp:
        try:
            train_clean, test_clean, clean_ALL = load_processed_data()
            print("Loaded processed datasets from disk.")
        except:
            print("Data needs to be rebuilt!")
            temp = True
        
    if temp:
        train, test = load_data()
        print("Training data shape:", train.shape)
        print("Test data shape:", test.shape)

        train_clean, test_clean = prepare_data1(train, test)
        clean_ALL = prepare_data2(train_clean, test_clean)

        save_processed_data(train_clean, test_clean, clean_ALL)

        
    print("\nCleaned training data shape:", train_clean.shape)
    print("Cleaned test data shape:", test_clean.shape)
    print("\nCleaned combined data shape:", clean_ALL.shape)
    print(clean_ALL.head(4))
    
    return clean_ALL


if __name__ == "__main__":
    output = main()
