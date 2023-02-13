import os

filename =   "randomForest_model.joblib"


if not os.path.exists("data-models/"+filename):
    print('Not existing')
else:
    print('Yes')