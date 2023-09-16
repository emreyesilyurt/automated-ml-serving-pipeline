# Automated Pipelines for Training, Validation, and Serving ML Models

This documentation provides an overview of the automated pipeline for training, validation, and serving ML models. In this project, the weatherHistory dataset was used, and you can find the dataset through this link. -> [Kaggle](https://www.kaggle.com/datasets/muthuj7/weather-dataset)

The pipeline includes the following phases:

## Dataset Selection and ML Model Development
- Implement a pipeline for training and validation of an ML model using frameworks like RandomForest or TensorFlow.
- Select and store the best model based on a chosen metric. (Accuracy)

## REST API Development and Containerization
- Implement a REST API using FastAPI to serve the trained ML model.
- Containerize the application using a Dockerfile.

## Automated Build and Push of Docker Image
- Create a GitHub workflow action that builds the Docker image at each pull request.
- Push the Docker image to a container registry to Docker Hub.

## Manifest File for Kubernetes Deployment
- Create a manifest file (`deployment.yaml`, `service.yaml`) to deploy the application on a Kubernetes cluster.
- Include specifications for the deployment and service of the application.
- Apply the manifest file using `kubectl` to deploy the application.


### -> Kubernetes Deployment and Usage

This section provides instructions on deploying and utilizing the application on a Kubernetes cluster. Follow these steps:

1. Ensure you have a Kubernetes cluster available. Docker Desktop has been used as a local cluster in this project also its usable cloud-based solution like AWS EKS, Google Kubernetes Engine, or Azure Kubernetes Service.

2. Navigate to the project directory containing the Kubernetes manifest files (`deployment.yaml`, `service.yaml`, `pod.yaml`).

3. Deploy the application using the following command:

   ```shell
   kubectl apply -f deployment.yaml
   ```
   This command creates the deployment and replicaset for the application in the Kubernetes cluster.

4. Build and create POD from manifest
    ```
    kubectl create -f pod.yaml
    ```

5. Verify that the deployment and pods are running by executing the following command:
    
    ```
    kubectl get deployments
    kubectl get pods
    ```
    This command displays the status of the deployed application and the running pods.


Note: The service.yaml file can be used to expose the application externally if needed, allowing access from outside the cluster. If you decide to use the service, deploy it using the command:
    ```shell
    kubectl apply -f service.yaml
    ```

## Usage and Testing
- Interact with the deployed application through the exposed API endpoints.
- Test the functionality of the API using tools like cURL. 
- Access the application through the assigned IP address and port.
- Scale the deployment using `kubectl scale` to adjust the number of replicas.
- Monitor the application using Kubernetes monitoring and logging tools.

## Repository Structure
The repository for this project should follow the following structure:

```
project-root/
│   README.md                                         # Project README file
│   requirements.txt                                  # Project requirements 
│   data/   
│   │   weatherHistory.csv                            # Project dataset
│   │
│   └───saved_models/
│   │   │   random_forest_model.pkl                   # Saved model
│   │   │
│   └───src/
│   │   │   api.py                                    # FastAPI implementation for serving the ML model
│   │   │   main.py                                   # Code for training and validation of the ML model
│   │
│   └───Dockerfile                                    # Dockerfile for containerizing the application
│   │
│   └───project_venv/                                 # Virtualenv for project
│   │
│   │
│   └───mlflow/                                       # Experiment tracking tool directory
│   │
│   └───manifests/
│   │   │   deployment.yaml                           # Deployment file for Kubernetes deployment
│   │   │   service.yaml                              # Service file for Kubernetes deployment
│   │   │   pod.yaml                                  # Pod file for Kubernetes deployment
│   │
│   └───.github/
│       └───workflows/
│           │   build-push.yaml                       # GitHub workflow for building and pushing Docker image

```


## Getting Started
To get started with the project, follow these steps:

1. Clone the repository: `git clone <repository_url>`.
2. Set up the development environment by installing the necessary dependencies mentioned in `requirements.txt`.
3. Prepare the dataset and place it in the `data/` directory.
4. Train and validate the ML model using the `main.py` script.
5. Select the best model based on the chosen metric and save it in the `saved_models/` directory.
6. Implement the REST API using FastAPI in the `api.py` file.
7. Build the Docker image using the provided Dockerfile: `docker buildx build -t <image_name> ..`
8. Push the Docker image to the container registry: `docker push <image_name>`.

## GitHub Workflow
The GitHub workflow `build-push.yaml` automates the process of building and pushing the Docker image to the container registry. It triggers on each pull request and performs the following steps:

1. Checks out the repository.
2. Sets up Docker Buildx for building multi-architecture images.
3. Logs in to the container registry using the provided credentials.
4. Builds the Docker image based on the Dockerfile and tags it with the desired repository and tag name.
5. Pushes the Docker image to the container registry.

## Kubernetes Deployment
The `deployment.yaml` file contains the manifest for deploying the application on a Kubernetes cluster. It includes specifications for the deployment and service.

## Get Predictions
 The predicted data has not been directed anywhere because it is unknown where the predictions will be used. If you want to view the predictions, you can use the curl command to get the terminal output. Like this: 

    ```
    curl -X POST http://localhost:8080/predict
    ```
 # automated-ml-serving-pipeline
