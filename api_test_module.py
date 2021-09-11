"""
# FILE: api_test_module.py
# AUTHOR: OA
# DESCR: api to return predictions for given model and input
"""
# IMPORT STATEMENTS
import os

import joblib
import pandas as pd


# LOAD MODELS INTO MEMORY
MODEL_PATH = "/home/wrsadmin/tutorials/escalation-predictive-models/model/model.joblib"
model = joblib.load(MODEL_PATH)
print("*Model Loaded**")



# Add in endpoints
def healthcheck():
    """
    DESCR: Return information about to ensure it is in usable state
    PAYLOAD: NONE
    OUTPUT: str - check if api is ok
    """
    return "OK"


def test_model_serve_function(payload):
    """
    DESCR: Add your description
    PAYLOAD: dict - input1:
                    input2:
                    .
                    .
                    .
                    .
    OUTPUT: dict - contains prediction and payload fields
    """
    print(payload)
    input_sample = pd.DataFrame([payload])

    #Make any data processing such as datetime asjustment

    #print(f"Using: {model_day}")
    pred = model.predict(input_sample)
    # probs = model.predict_proba(input_sample)[0]

    # Configure your output format
    # output_payload = {'predcited_class': str(probs[0][0]),
    #                   'predcited_class_prob': float(probs[0][1]),
    #                   'payload': payload
    #                  }

    output_payload = {'predcited_quality': str(pred[0]),
                      'payload': payload
                     }

    return output_payload
