# importing packages
import pandas as pd
import pickle
import json
import argparse
from IPython.display import display

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define module')
    parser.add_argument('--module', help='1 or 2', required=True)
    parser.add_argument('--file_path', help='file path for test data')
    parser.set_defaults(file_path='data/test.csv')
    
    args = parser.parse_args()
    print(args.module)
    print(args.file_path)
    
    # importing test data
    test_df = pd.read_csv('data/test.csv')
    
    if args.module == '1':
        # reading config file
        config_file = open('trained_models/stay_duration_pred.json',)
        # load trained model
        loaded_model = pickle.load(open('trained_models/stay_duration_pred.sav', 'rb'))
    
    elif args.module == '2':
        # reading config file
        config_file = open('trained_models/transaction_pred.json',)
        # load trained model
        loaded_model = pickle.load(open('trained_models/transaction_pred.sav', 'rb'))
    
    else:
        raise ValueError('No such module.')

    data = json.load(config_file)
    columns = data['columns']
    result = pd.DataFrame({'results':loaded_model.predict(test_df[columns])})

    display(result)
    result.to_csv('result_{}'.format(args.module))