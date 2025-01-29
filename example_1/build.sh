#!/bin/bash
IMAGE_URI="<region>-docker.pkg.dev/<project_id>/<artifact-repo>/<image>"
docker build -t $IMAGE_URI --build-arg MODEL_PATH="<gcs-model-path>" .
docker push $IMAGE_URI
