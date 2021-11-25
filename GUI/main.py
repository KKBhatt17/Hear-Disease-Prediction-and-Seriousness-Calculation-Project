from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, SubmitField, DecimalField
from wtforms.validators import DataRequired, NumberRange
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)
app.secret_key = "super secret key"


class UserHeartDataFrom(FlaskForm):
    """
    This class is to create a phone number form and perform validations on it, It inherits the FLASKFORM class from
    flask_wtf forms.
    """
    age = IntegerField(label='Age', validators=[DataRequired(), NumberRange(min=1, max=150)])
    sex = SelectField(label='Sex', choices=['M', 'F'])
    chestPainType = SelectField(label='Chest Pain Type', choices=['ATA', 'TA', 'NAP', 'ASY'])
    restingBP = IntegerField(label='Resting Blood pressure', validators=[DataRequired(), NumberRange(min=1)])
    cholesterol = IntegerField(label='Serum Cholesterol', validators=[DataRequired(), NumberRange(min=1)])
    fastingBS = SelectField(label='Fasting Blood Sugar', choices=['1', '0'])
    restingECG = SelectField(label='Resting ECG', choices=['NORMAL', 'ST'])
    maxHR = IntegerField(label='Maximum Heart Rate', validators=[DataRequired(), NumberRange(min=60, max=202)])
    exerciseAngina = SelectField(label='Exercise Angina', choices=['Y', 'N'])
    old_peak = DecimalField(label="Oldpeak")
    st_slope = SelectField(label='ST Slope', choices=['Up', 'Flat', 'Down'])
    submit = SubmitField(label='Predict Results')
    pass


def predict_heart_disease(reports):
    """
    This function is to predict the chances of heart disease and its severity given the input features.

    Inputs:
        :reports:The test results entered by the user are passed in via this parameter
        :type reports: List

    Returns:
        :: A String specifying the occurrence/non-occurrence of heart disease, along with the probability of the disease
        if it occurs
    """
    # initialising variables and assigning appropriate input features
    Age = reports[0][0]
    RestingBP = reports[0][3]
    Cholesterol = reports[0][4]
    FastingBS = reports[0][5]
    MaxHR = reports[0][7]
    Oldpeak = reports[0][9]
    Sex_F = 0
    Sex_M = 0
    ChestPainType_ASY = 0
    ChestPainType_ATA = 0
    ChestPainType_NAP = 0
    ChestPainType_TA = 0
    RestingECG_LVH = 0
    RestingECG_Normal = 0
    RestingECG_ST = 0
    ExerciseAngina_N = 0
    ExerciseAngina_Y = 0
    ST_Slope_Down = 0
    ST_Slope_Flat = 0
    ST_Slope_Up = 0

    # mapping categorical inputs to one-hot vector notation
    if reports[0][1] == 'M':
        Sex_M = 1
    else:
        Sex_F = 1

    if reports[0][2] == 'ASY':
        ChestPainType_ASY = 1
    elif reports[0][2] == 'ATA':
        ChestPainType_ATA = 1
    elif reports[0][2] == 'NAP':
        ChestPainType_NAP = 1
    else:
        ChestPainType_TA = 1

    if reports[0][6] == 'LVH':
        RestingECG_LVH = 1
    elif reports[0][6] == 'Normal':
        RestingECG_Normal = 1
    else:
        RestingECG_ST = 1

    if reports[0][8] == 'Y':
        ExerciseAngina_Y = 1
    else:
        ExerciseAngina_N = 1

    if reports[0][10] == 'Up':
        ST_Slope_Up = 1
    elif reports[0][10] == 'Down':
        ST_Slope_Down = 1
    else:
        ST_Slope_Flat = 1

    # preparing a list of final inputs which are to be provided to ML model
    new_reports = [
        [Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Sex_F, Sex_M, ChestPainType_ASY, ChestPainType_ATA,
         ChestPainType_NAP, ChestPainType_TA, RestingECG_LVH, RestingECG_Normal, RestingECG_ST, ExerciseAngina_N,
         ExerciseAngina_Y, ST_Slope_Down, ST_Slope_Flat, ST_Slope_Up]]

    df_reports = pd.DataFrame(new_reports,
                              columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                       'Q', 'R', 'S', 'T'])

    demo_feature_array = np.array(df_reports)

    # loading the trained model
    rf_clf_loaded = joblib.load('saved_model_rf_clf.pkl')

    # making predictions
    demo_prediction = rf_clf_loaded.predict(demo_feature_array)

    # computing severity/seriousness/probability of heart disease
    chances_0 = round(rf_clf_loaded.predict_proba(demo_feature_array)[0][0], 2) * 100
    chances_1 = round(rf_clf_loaded.predict_proba(demo_feature_array)[0][1], 2) * 100

    # returning results
    if demo_prediction[0] == 0:
        prediction = "The patient does not have a heart disease"
    elif demo_prediction[0] == 1:
        prediction = "The patient have a heart disease, and the chances are " + str(chances_1) + " %"
    return prediction


@app.route('/', methods=['POST', 'GET'])
def home():
    """
    This route is the home page for the app, It consists of a form which accepts inputs from user, validates the input.
    In case of validation errors it flashes appropriate error messages. If inputs are in correct format, then predict
    the results.
    :return: Shows the results in MODAL FOOTER.
    """
    form = UserHeartDataFrom()
    if form.validate_on_submit():
        # fetching features from the form after necessary validation
        age = form.age.data
        sex = form.sex.data
        chest_pain_type = form.chestPainType.data
        resting_bp = form.restingBP.data
        cholesterol = form.cholesterol.data
        fasting_bs = int(form.fastingBS.data)
        resting_ecg = form.restingECG.data
        max_hr = form.maxHR.data
        exercise_angina = form.exerciseAngina.data
        old_peak = float(form.old_peak.data)
        st_slope = form.st_slope.data

        # making a list of features
        input = []
        input.append(age)
        input.append(sex)
        input.append(chest_pain_type)
        input.append(resting_bp)
        input.append(cholesterol)
        input.append(fasting_bs)
        input.append(resting_ecg)
        input.append(max_hr)
        input.append(exercise_angina)
        input.append(old_peak)
        input.append(st_slope)

        # Predicting result fot given input
        reports = [input]
        prediction = predict_heart_disease(reports)
        result = prediction

        return render_template('home.html', form=form, result=result)
    return render_template('home.html', form=form)


if __name__ == '__main__':
    app.debug = True
    app.run()