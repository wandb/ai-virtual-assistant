# Running the API Gateway microservice with all required microservices
```
docker compose -f ./deploy/compose/docker-compose.yaml up -d --build
```

The API Gateway server can be accessed from the swagger url `http://<ip>:9000/docs/`
A static openapi schema can be found [here.](../../docs/api_references/api_gateway_server.json)