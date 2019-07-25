import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from counterfactual import linear_explanation
import pickle
import timeit

class chris_diverse:

    def __init__(self):
        self.frame = pd.read_csv('adult_frame.csv')

        self.train_ind, self.test_ind = next(ShuffleSplit(test_size=0.20, random_state=17).split(self.frame))

        self.target=np.asarray(self.frame['income']=='>=50k')

        self.inputs=self.frame[self.frame.columns[0:8]]

        self.exp=linear_explanation()

        self.exp.encode_pandas(self.inputs)

        self.exp.train_logistic(self.target[self.train_ind],self.train_ind)

        self.exp.special_val={'workclass': {0: 'Government', -3: 'Other/Unknown', -2: 'Private', -1: 'Self-Employed'}, 'education': {0: 'Assoc', -7: 'Bachelors', -6: 'Doctorate', -5: 'HS-grad', -4: 'Masters', -3: 'Prof-school', -2: 'School', -1: 'Some-college'}, 'marital_status': {0: 'Divorced', -4: 'Married', -3: 'Separated', -2: 'Single', -1: 'Widowed'}, 'occupation': {0: 'Blue-Collar', -5: 'Other/Unknown', -4: 'Professional', -3: 'Sales', -2: 'Service', -1: 'White-Collar'}, 'race': {0: 'Other', -1: 'White'}, 'gender': {0: 'Female', -1: 'Male'}}

        scores = self.exp.evaluate_subset(self.test_ind)
        preds = []
        for s in scores:
            if s >0:
                preds.append(1)
            else:
                preds.append(0)
        preds = np.array(preds)
        acc = accuracy_score(np.around(preds), self.target[self.test_ind])
        print("Accuracy: " + str(acc))


    def explain(self, ix, total_cfs, labels=("'pos'","'neg'")):

        self.inputs_unique = self.inputs.iloc[self.test_ind].drop_duplicates(subset=list(self.inputs.columns)).reset_index(drop=True)
        self.dev_data_sampled = self.inputs_unique.sample(n=500, random_state =17)

        x = self.dev_data_sampled.values[ix]
        print('inputs: ',x)
        print('encoded: ',self.exp.mixed_encode(x))

        pred = self.exp.evaluate(self.exp.mixed_encode(x))
        print("True label:{}, model_pred:{} ({})\n"\
              .format(self.target[self.test_ind[ix]], np.around(pred,2)[0], pred[0]>0))

        text = self.exp.explain_entry(x, total_cfs, labels=("'pos'","'neg'"))
        #The explanation is an ordered list of text strings.
        for t in text:
            print(t)
            print()
        return text

    def get_summary(self, total_cfs, labels=("'pos'","'neg'"), save_filename="chris_summary"):
        self.inputs_unique = self.inputs.iloc[self.test_ind].drop_duplicates(subset=list(self.inputs.columns)).reset_index(drop=True)
        self.dev_data_sampled = self.inputs_unique.sample(n=500, random_state =17)
        print(self.dev_data_sampled.head())

        chris_summary = []
        for x in self.dev_data_sampled.values:
            start_time = timeit.default_timer()
            text = self.exp.explain_entry(x, total_cfs, labels=labels)
            elapsed = timeit.default_timer() - start_time

            summary = [t for t in text]
            summary.extend([round(elapsed,4), len(text)])
            chris_summary.append(summary)

        with open(save_filename+'.data', 'wb') as filehandle:
            pickle.dump(chris_summary, filehandle)
