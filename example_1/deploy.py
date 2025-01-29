"""
Sample code to deploy to Vertex. Install vertexai SDK first:
`pip install google-cloud-aiplatform`
"""
import random

from google.cloud import aiplatform
from google.cloud.aiplatform import explain

from app.entities import feat_ls


# initialize
PROJECT_ID = "<project_id>"
REGION = "<region>"
aiplatform.init(project=PROJECT_ID, location=REGION)

# deploy model
MODEL_ID = "<model_id>"
VERSION_ID = "default"
IMAGE_URI = "<custom-image-uri>"
MODEL_LOCATION = "<model-gcs-location>"
MODEL_NAME = "<model-display-name>"
explanation_params = explain.ExplanationParameters(
    sampled_shapley_attribution=explain.SampledShapleyAttribution(
        path_count=10,
    ),
)
explanation_metadata = explain.ExplanationMetadata(
    inputs={
        feat: {} for feat in feat_ls
    },
    outputs={
        "output": {}
    }
)
aiplatform.Model.upload(
    serving_container_image_uri=IMAGE_URI,
    artifact_uri=MODEL_LOCATION,
    parent_model=MODEL_ID, # given you are creating a version of an existing model; otherwise, use `model_id` instead
    serving_container_ports=[8080],
    explanation_metadata=explanation_metadata,
    explanation_parameters=explanation_params,
    display_name=MODEL_NAME,
    sync=True,
)

# online deployment
ENDPOINT_ID = "<endpoint_id>"
DEPLOYMENT_NAME = "<your-deploy-name>"
SERVICE_ACCOUNT = "<service-account>"
model = aiplatform.Model(f"projects/{PROJECT_ID}/locations/{REGION}/models/{MODEL_ID}@{VERSION_ID}")
endpoint = aiplatform.Endpoint(f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}")
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name=DEPLOYMENT_NAME,
    traffic_split={"0": 100},
    machine_type="n1-standard-4",
    sync=True,
    service_account=SERVICE_ACCOUNT,
    accelerator_count=0,
)

# Test the endpoint
instances = [
    {feat_col: random.random() for feat_col in feat_ls}
]
pred = endpoint.predict(instances=instances)
print(f"predictions: {pred.predictions}\n")
explained_pred = endpoint.explain(instances=instances)
print(f"Explainations: {explained_pred.explanations}\n")
