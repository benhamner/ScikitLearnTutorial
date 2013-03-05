import data_io
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Reading in the training data")
    train, train_labels = data_io.get_train()

    print("Extracting features and training model")
    classifier = RandomForestClassifier(n_estimators = 500,
                                        min_samples_leaf = 1)
    classifier.fit(train, train_labels)

    print("Saving the classifier")
    data_io.save_model(classifier)
    
if __name__=="__main__":
    main()