from google.cloud import aiplatform_v1


def sample_raw_predict():
    # Create a client
    client = aiplatform_v1.PredictionServiceClient()

    # Initialize request argument(s)
    request = aiplatform_v1.RawPredictRequest(
        endpoint=6745569807103426560,
    )

    # Make the request
    response = client.raw_predict(request='{sequences: Snorlax loves my Tesla!}')

    # Handle the response
    print(response)
