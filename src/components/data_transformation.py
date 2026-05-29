import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path= os.path.join('artifacts' , 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_obj(self):
        try:
            num_feature = ['writing_score' , 'reading_score']
            cat_feature = ['gender' , 'race_ethnicity' ,'parental_level_of_education' , 'lunch' , 'test_preparation_course'] 

            num_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler(with_mean=False)) ])
            cat_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')) ,('encoder',OneHotEncoder(handle_unknown='ignore'))])

            logging.info('Numerical columns completed')
            logging.info('Categorical columns completed')

            preprocessor = ColumnTransformer(transformers=[('num',num_transformer,num_feature) , ('cat',cat_transformer,cat_feature)])

            return preprocessor

        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info('Read train and test data')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_transformer_obj()

            target_column_name = 'math_score'
            num_feature = ['writing_score' , 'reading_score']

            X_train = train_df.drop(columns=[target_column_name],axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name] , axis=1)
            y_test = test_df[target_column_name]

            logging.info('Applying preprocessing object on tain and test dataframe')

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info('Saved preprocessing object')

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path,

            )

        except Exception as e:
            raise CustomException(e,sys)
            
        
