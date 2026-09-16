def predict_insurance_cost(model, age, sex, bmi, children, smoker, region):
    """
    Predict insurance cost for a single person.
    
    Parameters:
        model: Trained Linear Regression model
        age (int): Age of the person
        sex (str): 'male' or 'female'
        bmi (float): Body Mass Index
        children (int): Number of children
        smoker (str): 'yes' or 'no'
        region (str): 'southwest', 'southeast', 'northwest', 'northeast'
    
    Returns:
        float: Predicted insurance cost
    """
    sex_encoded = 1 if sex.lower() == 'female' else 0
    smoker_encoded = 1 if smoker.lower() == 'yes' else 0
    
    region_map = {
        'southwest': 0,
        'southeast': 1,
        'northwest': 2,
        'northeast': 3
    }
    region_encoded = region_map.get(region.lower(), 0)
    
    input_data = [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]]
    prediction = model.predict(input_data)[0]
    
    return prediction
