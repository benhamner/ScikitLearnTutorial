import data_io
import pickle

def main():
    print("Loading the classifier")
    classifier = data_io.load_model()
    
    print("Making predictions") 
    test = data_io.get_test()
    predictions = classifier.predict(test)  

    print("Writing predictions to file")
    data_io.write_submission(predictions)

if __name__=="__main__":
    main()