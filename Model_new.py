import pandas as pd
import numpy as np
from sklearn import metrics
import configparser as cp
import pyodbc as pyd
import logging as lg
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine,event
from urllib.parse import quote_plus


class workForceModel:

    def __init__(self, configfilepath):
        self.__exceptionFlag = False
        try:
            self.__config = cp.ConfigParser()
            self.__config.read(configfilepath)
            self.logfile = self.__config['logs']['logspath'] + "log_WF_analytics-{}.txt".format(
                dt.datetime.utcnow().date().strftime('%Y-%m-%d'))
            self.tablepath = self.__config['logs']['tablepath']
            lg.basicConfig(filename=self.logfile, format='%(asctime)s %(message)s')
            self.logger = lg.getLogger()
            self.logger.setLevel(lg.DEBUG)
        except Exception as ex:
            self.__exceptionFlag = True
            self.logger.error("INITIALISATION FAILED ERRMSG:{}".format(ex))

    def data_loading(self):
        try:
            self.__cnxn = pyd.connect(self.__config['devDbConnDetails']['conn'])
            quoted = quote_plus(self.__config['devDbConnDetails']['conn'])
            self.engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted), fast_executemany = True)
            self.__cnxn.setdecoding(pyd.SQL_CHAR, encoding='utf-8')
            self.__cnxn.setdecoding(pyd.SQL_WCHAR, encoding='utf-8')
            self.__cnxn.setencoding(encoding='utf-8')
            self.__cursor = self.__cnxn.cursor()
            self.logger.debug("CONNECTED TO DATABASE")
        except Exception as ex:
            self.__exceptionFlag = True
            self.logger.error("DATABASE CONNECTION FAILED ERRMSG:{}".format(ex))

        try:
            self.logger.debug("DATA LOADING STARTED")
            self.__data = pd.read_sql(self.__config['queries']['loadData'], self.__cnxn)
            print(self.__data.shape)
            if self.__exceptionFlag == False:
                self.logger.debug("DATA LOADING COMPLETED")
            else:
                self.logger.debug("DATA LOADING FAILED")
        except Exception as ex:
            self.__exceptionFlag = True
            self.logger.error("QUERY EXECUTION FAILED FAILED ERRMSG:{}".format(ex))

    def datapreprocessing(self):
        try:
            self.logger.debug("DATA PREPROCESSING STARTED")
            print(self.__data.dtypes)

            # removing all the null rows.
            self.data = self.__data.dropna(axis=0, how='any')
            self.logger.debug("NULL VALUES DROPPED")

            self.data['EMP_STATUS'] = self.data['EMP_STATUS'].astype('int')
            self.data['GENDER_CODE'] = self.data['GENDER_CODE'].astype('int')
            print("number of input columns " + str(len(self.data.columns)))
            print(self.data['EMP_STATUS'].value_counts())
            self.final_table = self.data

            # Adding Target variable
            self.user_input.append('EMP_STATUS')
            self.emp_id_list = self.data[['EMP_WIN']]
            self.user_data = self.data.filter(self.user_input)

            print(self.user_data.columns)
            print("number of input columns "+ str(len(self.user_data.columns)))
            # Filling the dummies for Categorical data columns
            cate_main_list = ['MANAGER_INDIVIDUAL_CONT', 'GLOBAL_JOB_LEVEL', 'FLSA_CODE', 'SEG_DESC', 'COUNTRY_DESC',
                         'SO_LVL1_DESC', 'CUST_LVL1_DESC', 'AGE_GROUP','CONDUENT_EXP_GROUP']
            new_cate_list = [x for x in self.user_input if x in cate_main_list]
            cate_data = pd.get_dummies(data=self.user_data, columns=new_cate_list)
            #print(cate_data.columns)
            print("dummies ready!!!!!!")
            # Preparing the data for Model Training
            # getting the unbiased dataset for Training
            active = cate_data[cate_data.EMP_STATUS == 0]
            terminate = cate_data[cate_data.EMP_STATUS == 1]

            active_percent = len(active) / len(cate_data) * 100
            print("Active percent : "+str(active_percent))
            term_percent = len(terminate) / len(cate_data) * 100
            print("Term percent : " + str(term_percent))
            if term_percent <= 30 or active_percent <= 30:
                if term_percent <= 30 and active_percent >= 70:
                    sample_size = (70 / 100 * len(terminate) / 30) * 70
                    sample = active.sample(int(sample_size), random_state=12)
                    resample = pd.concat([terminate, sample])
                elif active_percent <= 30 and term_percent >= 70:
                    sample_size = (len(active) / 30) * 70
                    sample = terminate.sample(int(sample_size), random_state=12)
                    resample = pd.concat([active, sample])
            else:
                t_sample_size = 70 / 100 * len(terminate)
                t_sample = terminate.sample(int(t_sample_size), random_state=12)
                a_sample_size = 70 / 100 * len(active)
                a_sample = active.sample(int(a_sample_size), random_state=12)
                resample = pd.concat([a_sample, t_sample])
                print(resample.shape)
                print(resample.dtypes)
            self.np_train_labels = np.array(resample[['EMP_STATUS']])
            train_features = resample.drop(['EMP_STATUS'], axis=1)
            self.np_train_features = np.array(train_features)
            self.features_list = train_features.columns

            # Creating test features, test labels
            self.np_test_labels = np.array(cate_data[['EMP_STATUS']])
            self.np_test_features = np.array(cate_data.drop(['EMP_STATUS'], axis=1))

            self.logger.debug("SAMPLING AND FEATURE SELECTION COMPLETED")
            self.logger.debug("SELECTED FEATURES : " + str(self.user_input))
            if self.__exceptionFlag == False:
                self.logger.debug("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
            else:
                self.logger.debug("DATA PREPROCESSING FAILED")

        except Exception as ex:
            self.__exceptionFlag = True
            self.logger.error("DATA PRE-PROCESSING FAILED ERRMSG:{}".format(ex))

    def model_building(self):
        try:
            self.acc_time = pd.DataFrame()
            self.acc_time['APP_USER_NAME'] = [self.__config['user']['user_name'],self.__config['user']['user_name']]
            self.acc_time['MODEL_NAME'] = ['Linear Regression','Random Forest']
            self.logger.debug("MODEL BUILDING STARTED")
            # instantiate the model (using the default parameters)
            start_time = dt.datetime.utcnow()
            self.logreg = LogisticRegression()
            self.logreg.fit(self.np_train_features, self.np_train_labels)
            end_time = dt.datetime.utcnow()
            self.lr_model_time = end_time - start_time
            self.logger.debug("LOGISTIC REGRESSION MODEL COMPLETED")

            # Random Forest Model Building
            start_time = dt.datetime.utcnow()
            rfc = RandomForestClassifier(n_estimators=110, random_state=42)
            self.rfc = rfc.fit(self.np_train_features, self.np_train_labels)
            end_time = dt.datetime.utcnow()
            self.rf_model_time = end_time - start_time
            self.acc_time['MODEL_TIME_SEC'] = [self.lr_model_time.seconds,self.rf_model_time.seconds]

            self.logger.debug("RANDOM FOREST MODEL COMPLETED")
            self.feature_importances = pd.DataFrame(rfc.feature_importances_, index=self.features_list,
                                                    columns=['importance']).sort_values('importance', ascending=False)
            self.feature_importances = self.feature_importances.rename_axis("Features").reset_index()
            if self.__exceptionFlag == False:
                self.logger.debug("MODEL BUILD SUCCESSFULLY !!!!!!")
            else:
                self.logger.debug("MODEL BUILD FAILED!!!!!!")
        except Exception as ex:
            self.__exceptionFlag = True
            self.logger.error("MODEL BUILDING FAILED ERRMSG:{}".format(ex))

    def prediction(self):
        try:
            self.logger.debug("PREDICTION PROCESS STARTED")
            # Results to be saved
            # 1. Prediction result from LR
            # 2. Accuracy and Model Building time
            # 3. LR's Co-efficients and RF's Feature Importance for input features

            # Logistic Regression Prediction
            y_pred = self.logreg.predict(self.np_test_features)

            # Combining the Original data with the prediction result
            self.final_table['ACTUAL_STATUS'] = self.final_table['EMP_STATUS']
            self.final_table['PREDICTED_STATUS'] = y_pred

            # Logistic Regression Accuracy Calculation
            y_pred_proba = self.logreg.predict_proba(self.np_test_features)[::, 1]
            lr_auc = metrics.roc_auc_score(self.np_test_labels, y_pred_proba)
            self.logger.debug("LOGISTIC REGRESSION PREDICTION COMPLETED")


            # Random Forest Prediction
            rf_preds = self.rfc.predict(self.np_test_features)
            rf_auc = metrics.roc_auc_score(self.np_test_labels, rf_preds)

            # features with co-eff and importance
            feature_df = pd.DataFrame({'Features': self.features_list})
            self.logger.debug("RANDOM FOREST PREDICTION COMPLETED")

            self.acc_time['ACCURACY'] = [lr_auc, rf_auc]
            self.acc_time['USER_FEATURES_LIST'] = [str(self.user_input),str(self.user_input)]

            # Getting LR co-efficients and merging with RF Feature Importance
            co_eff = self.logreg.coef_
            coeff_list = co_eff.tolist()
            for item in coeff_list:
                coeff_item = item

            feature_df['coefficients'] = coeff_item
            self.feat_coeff_df = pd.merge(self.feature_importances, feature_df, on = 'Features')
            #self.feat_coeff_df['App_User_Name'] = [self.__config['user']['user_name']]*len(self.features_list)

            self.logger.debug("PREDICTION COMPLETED!!!!!")

            # Writing the results into Tables
            #self.final_table.to_csv(self.tablepath+"Final_Table1.csv")

            self.acc_time.to_sql('WF_ANALYTICS_ACCURACY_TIME', con = self.engine, schema='HR', if_exists='append', index=False)
            #self.acc_time.to_csv(self.tablepath+"Accuracy_Time.csv")
            self.logger.debug("Accuracy and Model Building time saved!!!!!")
            #self.feat_coeff_df.to_csv(self.tablepath+"Feature_importance_coeff.csv")
            self.feat_coeff_df.to_sql('WF_ANALYTICS_FEATURE_IMP_COEFF', con=self.engine, schema='HR', if_exists='append',index= False)
            self.logger.debug("Feature Importance and Co-Efficient saved!!!!!")
            self.final_table['APP_USER_NAME'] = self.__config['user']['user_name']
            print(self.final_table.columns)
            #self.final_tabl = self.final_table.head(10)
            self.final_table.to_sql('WF_ANALYTICS_PREDICTED_RESULT', con=self.engine, schema='HR', if_exists='replace',
                                    index=False)
            self.logger.debug("Final Table with Prediction saved!!!!!")
            if self.__exceptionFlag == False:
                self.logger.debug("PREDICTION COMPLETED!!!!!")
            else:
                self.logger.debug("PREDICTION FAILED")
        except Exception as ex:
            self.__exceptionFlag = True
            self.logger.error("PREDICTION FAILED ERRMSG:{}".format(ex))

    def main(self):
        try:
            # Need User input to select the features
            self.user_input = self.__config['features']['default_features'].split(',')
            self.logger.debug("\n WORKFORCE ANALYTICS STARTED")
            workforce.data_loading()
            workforce.datapreprocessing()
            workforce.model_building()
            workforce.prediction()
            if self.__exceptionFlag == False:
                self.logger.debug("WORKFORCE ANALYTICS COMPLETED SUCCESSFULLY.")
            else:
                self.logger.debug("WORKFORCE ANALYTICS FAILED.")

        except Exception as ex:
            self.__exceptionFlag = True
            self.logger.error("MAIN PROCESS FAILED ERRMSG:{}".format(ex))

        finally:
            self.__cnxn.close()
            if self.__exceptionFlag == True:
                self.logger.debug("WORKFORCE ANALYTICS FAILED.")


if __name__ == '__main__':
    workforce = workForceModel('config.txt')
    workforce.main()

