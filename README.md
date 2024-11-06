# NIM Agent Blueprint: AI Virtual Assistant for Customer Service

## Overview
With the rise of generative AI, companies are eager to enhance their customer service operations by integrating knowledge bases that are close to sensitive customer data. Traditional solutions often fall short in delivering context-aware, secure, and real-time responses to complex customer queries. This leads to longer resolution times, limited customer satisfaction, and potential data exposure risks. A centralized knowledge base that integrates seamlessly with internal applications and call center tools is vital to improving customer experience while ensuring data governance.
The AI virtual assistant for customer service NIM Agent Blueprint, powered by NVIDIA NeMo Retriever™ and NVIDIA NIM™ microservices, along with retrieval-augmented generation (RAG), offers a streamlined solution for enhancing customer support. It enables context-aware, multi-turn conversations, providing general and personalized Q&A responses based on structured and unstructured data, such as order history and product details.

## Architecture

![Key Generated window.](./docs/imgs/IVA-blueprint-diagram-r5.png)

## Software Components
The RAG-based AI virtual assistant provides a reference to build an enterprise-ready generative AI solution with minimal effort. It contains the following software components:

* NVIDIA NIM microservices
   * Response Generation (Inference)
      * LLM NIM - llama-3.1-70B-instruct
      * NeMo Retriever embedding NIM - NV-Embed-QA-v5
      * NeMo Retriever reranking NIM - Rerank-Mistral-4b-v3
   * [Synthetic Data Generation](./notebooks/synthetic_data_generation.ipynb) for customization
      * Nemotron4-340B
* Orchestrator Agent - Langgraph based
* Text Retrievers - LangChain
* Structured Data (CSV) Ingestion - Postgres Database
* Unstructured Data (PDF) Ingestion - Milvus Database (Vector GPU-optimized)


Docker Compose scripts are provided which spin up the microservices on a single node. When ready for a larger-scale deployment, you can use the included Helm charts to spin up the necessary microservices. You will use sample Jupyter notebooks with the JupyterLab service to interact with the code directly.

The Blueprint contains sample use-case data pertaining to retail product catalog and customer data with purchase history but Developers can build upon this blueprint, by customizing the RAG application to their specific use case.  A sample customer service agent user interface and API-based analytic server for conversation summary and sentiment are also included.

## Key Functionalities
* Personalized Responses: Handles structured and unstructured customer queries (e.g., order details, spending history).
* Multi-Turn Dialogue: Offers context-aware, seamless interactions across multiple questions.
* Custom Conversation Style: Adapts text responses to reflect corporate branding and tone.
* Sentiment Analysis: Analyzes real-time customer interactions to gauge sentiment and adjust responses.
* Multi-Session Support: Allows for multiple user sessions with conversation history and summaries.
* Data Privacy: Integrates with on-premises or cloud-hosted knowledge bases to protect sensitive data.

By integrating NVIDIA NIM and RAG, the system empowers developers to build customer support solutions that can provide faster and more accurate support while maintaining data privacy.

## Target Audience
* Developers
* Data scientists
* Customer support teams

## Get Started

To get started with deployment follow

* [Prerequisites](#basic-prerequisites)
* Deployment
  * [Docker compose](./deploy/compose/README.md)


## Basic Prerequisites

This section lists down the bare mininum requirements to deploy this blueprint. Follow the required [deployment guide](./deploy/) based on your requirement to understand deployment method specific prerequisites.

#### Hardware requirements

##### Option 1: Deploy with NVIDIA hosted endpoints
By default, the blueprint uses the NVIDIA API Catalog hosted endpoints for LLM, embedding and reranking models.  Therefore all that is required is an instance with at least 8 cores and 64GB memory.


##### Option-2: Deploy with NIMs hosted locally
Once you familiarize yourself with the blueprint, you may want to further customize based upon your own use case which requires you to host your own LLM, embedding and reranking models.  In this case you will need access to a GPU accelerated A instance with 8 cores, 64GB memory and 8XA100 or 8XH100.

#### System requirements
Ubuntu 20.04 or 22.04 based machine, with sudo privileges

#### Software requirements
* **NVIDIA AI Enterprise or Developer License**: NVIDIA NIM for LLMs are available for self-hosting under the NVIDIA AI Enterprise (NVAIE) License. [Sign up](https://build.nvidia.com/meta/llama-3-8b-instruct?snippet_tab=Docker&signin=true&integrate_nim=true&self_hosted_api=true) for NVAIE license.
* An **NGC API key** is required to access NGC resources.  To obtain a key, navigate to Blueprint experience on NVIDIA API Catalog. Login / Sign up if needed and "Generate your API Key".

## Sample Data
The blueprint comes with [synthetic sample data](./data/) representing a typical customer service function, including customer profiles, order histories (structured data), and technical product manuals (unstructured data). A notebook is provided to guide users on how to ingest both structured and unstructured data efficiently.
Structured Data: Includes customer profiles and order history
Unstructured Data: Ingests product manuals, product catalogs, and FAQ

## AI Agent
This reference solution implements [different sub-agents using the open-source LangGraph framework and a supervisor agent to orchestrate the entire flow.](./src/agent/) These sub-agents address common customer service tasks for the included sample dataset. They rely on the Llama 3.1 models and NVIDIA NIM microservices for generating responses, converting natural language into SQL queries, and assessing the sentiment of the conversation.

## Key Components
* [**Structured Data Retriever**](./src/retrievers/structured_data/): Works in tandem with a Postgres database and PandasAI to fetch relevant data based on user queries.
* [**Unstructured Data Retriever**](./src/retrievers/unstructured_data/): Processes unstructured data (e.g., PDFs, FAQs) by chunking it, creating embeddings using the NeMo Retriever embedding NIM, and storing it in Milvus for fast retrieval.
* [**Analytics and Admin Operations**](./src/analytics/): To support operational requirements, the blueprint includes reference code and APIs for managing key administrative tasks
   * Storing conversation histories
   * Generating conversation summaries
   * Conducting sentiment analysis on customer interactions
These features ensure that customer service teams can efficiently monitor and evaluate interactions for quality and performance.

## Data Flywheel
The blueprint comes with [pre-built APIs](./docs/api_references/analytics_server.json) that support continuous model improvement. The feedback loop, or “data flywheel,” allows LLM models to be fine-tuned over time to enhance both accuracy and cost-effectiveness. Feedback is collected at multiple points in the process to refine the models’ performance further.

## Known issues
- The Blueprint responses can have significant latency when using [NVIDIA API Catalog cloud hosted models.](#option-1-deploy-with-nvidia-hosted-endpoints)

## Inviting the community to contribute
We're posting these examples on GitHub to support the NVIDIA LLM community and facilitate feedback. We invite contributions! Open a GitHub issue or pull request! See contributing [guidelines here.](./CONTRIBUTING.md)

## License
This NVIDIA NIM-AGENT BLUEPRINT is licensed under the [Apache License, Version 2.0.](./LICENSE.md)
Use of the sample data provided as part of this blueprint is governed by [the NVIDIA asset license.](./data/LICENSE)
