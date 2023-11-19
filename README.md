# Machine Learning API Service 

This repo is 1 of 2 repos made to streamline the development of production ML containerized services. The helm files live here ([2nd repo](https://github.com/degrasse-python/ml-service)). Machine Learning Prediction Algorithm made with Tensor (or SciKL) and Python's FastAPI. Image has been built with Docker. Location coming soon.
This repo contains the source code you can containerize using modern cloud with k8s or any container orcastrator. 
Focus on your ml work and use this cookie cutter to create an API image for deployment to have a prediction modeling service.
There is an example of a train-test-validation using tensorflow and scikit-learn to build serialized model. 


## Steps to Productionize FastAPI code

The steps involved deploying your FastAPI application in a production environment, ensuring scalability, reliability, and security. Here are the key steps to productionize your FastAPI application:

- [] Web Server Deployment:
    FastAPI includes a development server (uvicorn) that is suitable for testing and development. However, for production, it's recommended to use a production-ready ASGI server like Gunicorn. You can install Gunicorn using:

    bash

`pip install gunicorn`

Then, start your FastAPI application using Gunicorn:

bash

gunicorn -w 4 -k uvicorn.workers.UvicornWorker myapp:app

This command starts Gunicorn with 4 worker processes. Adjust the number of workers based on your server's capacity.

- [] Reverse Proxy:
Use a reverse proxy (e.g., Nginx or Apache) to forward requests to your FastAPI application. This provides additional features like load balancing, SSL termination, and improved security. Configure the reverse proxy to communicate with Gunicorn.

- [] Environment Configuration:
Use environment variables or a configuration system to manage settings like database URLs, secret keys, and other configuration options. FastAPI supports environment variables natively.

- [] Security:
Ensure that your FastAPI application follows security best practices. This includes validating input data, using secure connections (HTTPS), and implementing proper authentication and authorization mechanisms.

- [] Logging:
Implement logging to capture relevant information about your application's behavior. FastAPI integrates well with Python's logging module. Consider using log aggregators for centralized logging.

- [] Monitoring and Metrics:
Implement monitoring and metrics to gain insights into your application's performance. Tools like Prometheus and Grafana can be used to collect and visualize metrics.

- [] Containerization:
Consider containerizing your FastAPI application using Docker. This simplifies deployment and ensures consistent environments across different stages of development and production.

- [] Orchestration:
If your application consists of multiple services, consider using container orchestration tools like Kubernetes to manage deployment, scaling, and orchestration.

- [] Database Management:
Ensure that your database connection is managed efficiently. Use connection pooling for database connections. Implement proper indexing and caching strategies for optimal database performance.

- [] Error Handling:
Implement robust error handling to gracefully handle errors and provide meaningful responses to users. Log errors for debugging and monitoring purposes.

- [] Testing:
Write comprehensive tests to cover different aspects of your application, including unit tests, integration tests, and end-to-end tests. Automated testing helps ensure the reliability of your application.

- [] Documentation:
Keep your API documentation up-to-date using tools like Swagger UI, ReDoc, or FastAPI's built-in documentation system. Clear and accurate documentation is crucial for developers using your API.

## Notes
Use kaniko. Ditch docker.