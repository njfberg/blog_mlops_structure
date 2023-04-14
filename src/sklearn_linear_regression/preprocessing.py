def preprocess(input_data):
    """Preprocess the input data."""
    return input_data.drop(["id"], axis=1)
