Env variables

```
REGION=us-central1
PROJECT=pokemon-rl-jbro914
REPO=rl-images
IMAGE=showdown-rl
```

Build (from `cloud_deployment` folder)

```
docker buildx build \
  --platform linux/amd64 \
  -t $REGION-docker.pkg.dev/$PROJECT/$REPO/$IMAGE:latest \
  --push .
```

Run job

H100

```
REGION=us-central1
PROJECT=pokemon-rl-jbro914
REPO=rl-images
IMAGE=showdown-rl

gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=showdown-rl-1gpu \
  --worker-pool-spec=machine-type=a3-highgpu-1g,\
accelerator-type=NVIDIA_H100_80GB,\
accelerator-count=1,\
replica-count=1,\
container-image-uri=$REGION-docker.pkg.dev/$PROJECT/$REPO/$IMAGE:latest
```

CPU Only

```

gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=showdown-rl-cpu \
  --worker-pool-spec=machine-type=n2-standard-16,\
replica-count=1,\
container-image-uri=$REGION-docker.pkg.dev/$PROJECT/$REPO/$IMAGE:latest
```
