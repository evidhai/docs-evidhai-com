- Day 1
  collapsed:: true
	- Collected Notes and agreed on materials
- Day 2
  collapsed:: true
	- # AWS Certified AI Practitioner (AIF-C01) Plan
	  
	  [https://d1.awsstatic.com/training-and-certification/docs-ai-practitioner/AWS-Certified-AI-Practitioner_Exam-Guide.pdf?p=cert&c=ai&z=3](https://d1.awsstatic.com/training-and-certification/docs-ai-practitioner/AWS-Certified-AI-Practitioner_Exam-Guide.pdf?p=cert&c=ai&z=3)
	  
	  ![image.png](../assets/aif-domains.png)
	  
	  Timelines: 
	  
	  Complete course on 21st April
	  
	  12th May - Exam
	  
	  ---
	- ## Domain 1:  Fundamentals of AI and ML
	- Domain 1: Fundamentals of AI and ML 20%
	- ## Task Statement 1.1: Explain basic AI concepts and terminologies.
	    
	    Objectives:
	    • Define basic AI terms (for example, AI, ML, deep learning, neural networks,
	    computer vision, natural language processing [NLP], model, algorithm,
	    training and inferencing, bias, fairness, fit, large language model [LLM]).
	    • Describe the similarities and differences between AI, ML, and deep learning.
	    • Describe various types of inferencing (for example, batch, real-time).
	    • Describe the different types of data in AI models (for example, labeled and
	    unlabeled, tabular, time-series, image, text, structured and unstructured).
	    • Describe supervised learning, unsupervised learning, and reinforcement
	    learning.
	- ## Task Statement 1.2: Identify practical use cases for AI.
	    
	    Objectives:
	    • Recognize applications where AI/ML can provide value (for example, assist
	    human decision making, solution scalability, automation).
	    • Determine when AI/ML solutions are not appropriate (for example, cost-
	    benefit analyses, situations when a specific outcome is needed instead of a
	    prediction).
	    • Select the appropriate ML techniques for specific use cases (for example,
	    regression, classification, clustering).
	    • Identify examples of real-world AI applications (for example, computer
	    vision, NLP, speech recognition, recommendation systems, fraud detection,
	    forecasting).
	    • Explain the capabilities of AWS managed AI/ML services (for example,
	    SageMaker, Amazon Transcribe, Amazon Translate, Amazon Comprehend,
	    Amazon Lex, Amazon Polly).
	- ## Task Statement 1.3: Describe the ML development lifecycle.
	    Objectives:
	    • Describe components of an ML pipeline (for example, data collection,
	    exploratory data analysis [EDA], data pre-processing, feature engineering,
	    model training, hyperparameter tuning, evaluation, deployment,
	    monitoring).
	    • Understand sources of ML models (for example, open source pre-trained
	    models, training custom models).
	    • Describe methods to use a model in production (for example, managed API
	    service, self-hosted API).
	    • Identify relevant AWS services and features for each stage of an ML pipeline
	    (for example, SageMaker, Amazon SageMaker Data Wrangler, Amazon
	    SageMaker Feature Store, Amazon SageMaker Model Monitor).
	    • Understand fundamental concepts of ML operations (MLOps) (for example,
	    experimentation, repeatable processes, scalable systems, managing
	    technical debt, achieving production readiness, model monitoring, model
	    re-training).
	    • Understand model performance metrics (for example, accuracy, Area Under
	    the ROC Curve [AUC], F1 score) and business metrics (for example, cost per
	    user, development costs, customer feedback, return on investment [ROI]) to
	    evaluate ML models.
	    
	  
	  | Task | Mapped AWS Services | Status |
	  | --- | --- | --- |
	  | Explain basic AI concepts | General – no specific services |  |
	  | Identify practical AI use cases | Amazon SageMaker, Amazon Comprehend, Amazon Transcribe, Amazon Translate, Amazon Lex, Amazon Polly |  |
	  | Describe the ML development lifecycle | Amazon SageMaker, SageMaker Data Wrangler, SageMaker Feature Store, SageMaker Model Monitor, Amazon Augmented AI (A2I) |  |
	  | Understand MLOps concepts | Amazon SageMaker, SageMaker Model Monitor, Amazon CloudWatch |  |
	  | Evaluate model performance | Amazon SageMaker, CloudWatch, Amazon QuickSight (for custom metrics) |  |
	  |  |  |  |
	  |  |  |  |
	  
	  ---
	- ## Domain 2: Fundamentals of Generative AI
	- Domain 2 Fundamentals of Generative AI 24%
	- ## Task Statement 2.1: Explain the basic concepts of generative AI.
	    Objectives:
	    
	    • Understand foundational generative AI concepts (for example, tokens,
	    chunking, embeddings, vectors, prompt engineering, transformer-based
	    LLMs, foundation models, multi-modal models, diffusion models).
	    • Identify potential use cases for generative AI models (for example, image,
	    video, and audio generation; summarization; chatbots; translation; code
	    generation; customer service agents; search; recommendation engines).
	    • Describe the foundation model lifecycle (for example, data selection, model
	    selection, pre-training, fine-tuning, evaluation, deployment, feedback).
	- ## Task Statement 2.2: Understand the capabilities and limitations of generative AI for
	    solving business problems.
	    
	    Objectives:
	    • Describe the advantages of generative AI (for example, adaptability,
	    responsiveness, simplicity).
	    • Identify disadvantages of generative AI solutions (for example,
	    hallucinations, interpretability, inaccuracy, nondeterminism).
	    • Understand various factors to select appropriate generative AI models (for
	    example, model types, performance requirements, capabilities, constraints,
	    compliance).
	    • Determine business value and metrics for generative AI applications (for
	    example, cross-domain performance, efficiency, conversion rate, average
	    revenue per user, accuracy, customer lifetime value).
	- ## Task Statement 2.3: Describe AWS infrastructure and technologies for building
	    generative AI applications.
	    
	    Objectives:
	    • Identify AWS services and features to develop generative AI applications
	    (for example, Amazon SageMaker JumpStart; Amazon Bedrock; PartyRock,
	    an Amazon Bedrock Playground; Amazon Q).
	    • Describe the advantages of using AWS generative AI services to build
	    applications (for example, accessibility, lower barrier to entry, efficiency,
	    cost-effectiveness, speed to market, ability to meet business objectives).
	    • Understand the benefits of AWS infrastructure for generative AI
	    applications (for example, security, compliance, responsibility, safety).
	    • Understand cost tradeoffs of AWS generative AI services (for example,
	    responsiveness, availability, redundancy, performance, regional coverage,
	    token-based pricing, provision throughput, custom models)
	    
	  
	  | Task | Mapped AWS Services | Status |
	  | --- | --- | --- |
	  | Explain generative AI concepts | Amazon Bedrock, PartyRock, Amazon Q, SageMaker JumpStart |  |
	  | Capabilities and limitations of GenAI | Amazon Bedrock, Amazon SageMaker |  |
	  | AWS tech for GenAI apps | Amazon Bedrock, PartyRock, Amazon Q, SageMaker JumpStart, Amazon EC2, Amazon S3, IAM, CloudWatch, Amazon VPC |  |
	  
	  ---
	- ## Domain 3:  Applications of Foundation Models
	- Domain 3: Applications of Foundation Models 28%
	- ## Task Statement 3.1: Describe design considerations for applications that use
	    foundation models.
	    
	    Objectives:
	    • Identify selection criteria to choose pre-trained models (for example, cost,
	    modality, latency, multi-lingual, model size, model complexity,
	    customization, input/output length).
	    • Understand the effect of inference parameters on model responses (for
	    example, temperature, input/output length).
	    • Define Retrieval Augmented Generation (RAG) and describe its business
	    applications (for example, Amazon Bedrock, knowledge base).
	    • Identify AWS services that help store embeddings within vector databases
	    (for example, Amazon OpenSearch Service, Amazon Aurora, Amazon
	    Neptune, Amazon DocumentDB [with MongoDB compatibility], Amazon
	    RDS for PostgreSQL).
	    • Explain the cost tradeoffs of various approaches to foundation model
	    customization (for example, pre-training, fine-tuning, in-context learning,
	    RAG).
	    • Understand the role of agents in multi-step tasks (for example, Agents for
	    Amazon Bedrock).
	- ## Task Statement 3.2: Choose effective prompt engineering techniques.
	    Objectives:
	    
	    • Describe the concepts and constructs of prompt engineering (for example,
	    context, instruction, negative prompts, model latent space).
	    • Understand techniques for prompt engineering (for example, chain-of-
	    thought, zero-shot, single-shot, few-shot, prompt templates).
	    • Understand the benefits and best practices for prompt engineering (for
	    example, response quality improvement, experimentation, guardrails,
	    discovery, specificity and concision, using multiple comments).
	    • Define potential risks and limitations of prompt engineering (for example,
	    exposure, poisoning, hijacking, jailbreaking).
	    Task Statement 3.3: Describe the training and fine-tuning process for foundation
	    models.
	    Objectives:
	    • Describe the key elements of training a foundation model (for example,
	    pre-training, fine-tuning, continuous pre-training).
	    • Define methods for fine-tuning a foundation model (for example,
	    instruction tuning, adapting models for specific domains, transfer learning,
	    continuous pre-training).
	    • Describe how to prepare data to fine-tune a foundation model (for
	    example, data curation, governance, size, labeling, representativeness,
	    reinforcement learning from human feedback [RLHF]).
	- ## Task Statement 3.4: Describe methods to evaluate foundation model performance.
	    Objectives:
	    • Understand approaches to evaluate foundation model performance (for
	    example, human evaluation, benchmark datasets).
	    • Identify relevant metrics to assess foundation model performance (for
	    example, Recall-Oriented Understudy for Gisting Evaluation [ROUGE],
	    Bilingual Evaluation Understudy [BLEU], BERTScore).
	    • Determine whether a foundation model effectively meets business
	    objectives (for example, productivity, user engagement, task engineering).
	    
	  
	  | Task | Mapped AWS Services | Status |
	  | --- | --- | --- |
	  | Design considerations | Amazon Bedrock, Amazon OpenSearch Service, Amazon Neptune, Amazon Aurora, RDS for PostgreSQL, Amazon DocumentDB |  |
	  | Prompt engineering | Amazon Bedrock, PartyRock, Amazon Q |  |
	  | Training/fine-tuning foundation models | Amazon SageMaker, Amazon Bedrock |  |
	  | Evaluate foundation model performance | Amazon SageMaker, SageMaker Model Monitor, Amazon Augmented AI (A2I) |  |
	  
	  ---
	- ## Domain 4: Guidelines for Responsible AI
	- Domain 4: Guidelines for Responsible AI 14%
	- ## Task Statement 4.1: Explain the development of AI systems that are responsible.
	    Objectives:
	    
	    • Identify features of responsible AI (for example, bias, fairness, inclusivity,
	    robustness, safety, veracity).
	    • Understand how to use tools to identify features of responsible AI (for
	    example, Guardrails for Amazon Bedrock).
	    • Understand responsible practices to select a model (for example,
	    environmental considerations, sustainability).
	    • Identify legal risks of working with generative AI (for example, intellectual
	    property infringement claims, biased model outputs, loss of customer trust,
	    end user risk, hallucinations).
	    • Identify characteristics of datasets (for example, inclusivity, diversity,
	    curated data sources, balanced datasets).
	    • Understand effects of bias and variance (for example, effects on
	    demographic groups, inaccuracy, overfitting, underfitting).
	    • Describe tools to detect and monitor bias, trustworthiness, and truthfulness
	    (for example, analyzing label quality, human audits, subgroup analysis,
	    Amazon SageMaker Clarify, SageMaker Model Monitor, Amazon Augmented
	    AI [Amazon A2I]).
	- ## Task Statement 4.2: Recognize the importance of transparent and explainable
	    models.
	    Objectives:
	    
	    • Understand the differences between models that are transparent and
	    explainable and models that are not transparent and explainable.
	    • Understand the tools to identify transparent and explainable models (for
	    example, Amazon SageMaker Model Cards, open source models, data,
	    licensing).
	    • Identify tradeoffs between model safety and transparency (for example,
	    measure interpretability and performance).
	    • Understand principles of human-centered design for explainable AI
	    
	  
	  | Task | Mapped AWS Services | Status |
	  | --- | --- | --- |
	  | Develop responsible AI systems | SageMaker Clarify, SageMaker Model Monitor, Amazon Augmented AI (A2I), Guardrails for Amazon Bedrock |  |
	  | Transparent and explainable models | SageMaker Model Cards, SageMaker JumpStart (open-source models) |  |
	  
	  ---
	- ## Domain 5: Security, Compliance, and Governance for AI Solutions
	- Domain 5: Security, Compliance, and Governance for AI Solutions 14%
	    
	    Task Statement 5.1: Explain methods to secure AI systems.
	    Objectives:
	    
	    • Identify AWS services and features to secure AI systems (for example, IAM
	    roles, policies, and permissions; encryption; Amazon Macie; AWS
	    PrivateLink; AWS shared responsibility model).
	    • Understand the concept of source citation and documenting data origins
	    (for example, data lineage, data cataloging, SageMaker Model Cards).
	    • Describe best practices for secure data engineering (for example, assessing
	    data quality, implementing privacy-enhancing technologies, data access
	    control, data integrity).
	    • Understand security and privacy considerations for AI systems (for example,
	    application security, threat detection, vulnerability management,
	    infrastructure protection, prompt injection, encryption at rest and in
	    transit).
	    
	    Task Statement 5.2: Recognize governance and compliance regulations for AI
	    systems.
	    Objectives:
	    
	    • Identify regulatory compliance standards for AI systems (for example,
	    International Organization for Standardization [ISO], System and
	    Organization Controls [SOC], algorithm accountability laws).
	    • Identify AWS services and features to assist with governance and regulation
	    compliance (for example, AWS Config, Amazon Inspector, AWS Audit
	    Manager, AWS Artifact, AWS CloudTrail, AWS Trusted Advisor).
	    • Describe data governance strategies (for example, data lifecycles, logging,
	    residency, monitoring, observation, retention).
	    • Describe processes to follow governance protocols (for example, policies,
	    review cadence, review strategies, governance frameworks such as the
	    Generative AI Security Scoping Matrix, transparency standards, team
	    training requirements).
	    
	  
	  | Task | Mapped AWS Services | Status |
	  | --- | --- | --- |
	  | Secure AI systems | IAM, AWS Key Management Service (KMS), Amazon Macie, AWS PrivateLink, Amazon S3, SageMaker Model Cards |  |
	  | Compliance and governance | AWS Config, Amazon Inspector, AWS Audit Manager, AWS Artifact, AWS CloudTrail, AWS Trusted Advisor |  |
	  | Data governance strategies | AWS CloudTrail, Amazon Macie, AWS Audit Manager, Amazon S3, Amazon RDS, Amazon Redshift, AWS Glue, Lake Formation |  |
- Day 3
  collapsed:: true
	- ### Vocabulary set 1 summary
	  collapsed:: true
		- **Artificial Intelligence (AI):** AI simulates human
		  intelligence through computer systems, capable of performing tasks that
		  typically require human intelligence.
		- **Machine Learning (ML):** A subset of AI, ML enables
		  systems to learn from data and improve over time without human
		  intervention, using mathematical algorithms to identify patterns and
		  make predictions.
		- **Artificial Neural Networks (ANN):** A subset of ML,
		  ANNs mimic the human brain's network to recognize patterns and improve
		  accuracy through interconnected nodes and neurons.
		- **Deep Learning (DL):** Utilizes ANNs to analyze
		  patterns in data, commonly applied to sound, text, and images, using
		  multiple network layers to identify complex patterns.
		- **Generative AI (GAI):** A subset of DL, GAI models
		  create new content (e.g., images, text, audio) using large datasets,
		  producing outputs that resemble human-created content.
		- **Foundation Models (FM):** Trained on extensive
		  datasets, FMs serve as the basis for developing models that interpret
		  language, generate images, and more, with examples like Stable Diffusion for images and GPT-4 for language.
		- **Large Language Models (LLM):** Used in GAI to generate text by predicting and translating content, LLMs are trained on
		  transformer models and focus on language patterns and algorithms.
		- **Natural Language Processing (NLP):** Enables systems
		  to understand and interpret human language in written and verbal forms,
		  involving natural language understanding (NLU) and generation (NLG).
		- **Transformer Model:** A deep learning architecture that processes text and captures relationships between text elements,
		  supporting LLMs and enabling tasks like language translation and data
		  transformation.
		- **Generative Pretrained Transformer (GPT):** Utilizes
		  transformer models for generating human-like content, extensively used
		  in applications like text summarization and chatbots, exemplified by
		  ChatGPT.
		  
		  <!-- notionvc: 413d4277-dd01-4d13-8a5b-e0987a96141d -->
	- ### Vocabulary set 2 summary
	  collapsed:: true
		- **Responsible AI**: Emphasizes the importance of
		  ethical, lawful, and transparent AI practices to ensure trust and
		  confidence in AI systems. It involves setting principles and frameworks
		  to govern AI's impact on humanity.
		- **Labelled Data**: Refers to data tagged with
		  informative labels, aiding machine learning models in understanding and
		  learning from raw data. It requires human intervention for accurate
		  labeling.
		- **Supervised Learning**: A machine learning method using labeled datasets to predict outputs. The algorithm learns the
		  relationship between input and output data, with corrections made for
		  errors during training.
		- **Unsupervised Learning**: Involves learning from
		  unlabeled data, where the model autonomously identifies patterns and
		  relationships within the data without predefined labels.
		- **Semi-supervised Learning**: Combines supervised and
		  unsupervised learning, using a small set of labeled data alongside a
		  larger set of unlabeled data to improve model training efficiency.
		- **Prompt Engineering**: Involves refining input prompts
		  for large language models to optimize their output. It enhances AI
		  performance by adjusting prompts to produce more relevant responses.
		- **Prompt Chaining**: A technique for creating
		  conversational interactions with language models by using a series of
		  prompts, enhancing contextual awareness and user experience, often used
		  in chatbots.
		- **Retrieval Augmented Generation (RAG)**: A framework
		  that supplements AI models with external factual data to generate
		  accurate and up-to-date responses, enhancing the model's reliability.
		- **Parameters**: Variables within a machine learning
		  model that are adjusted during training to optimize performance and
		  improve the model's ability to generalize data patterns.
		- **Fine Tuning**: The process of adjusting a pre-trained
		  model on specific tasks or datasets to enhance its performance, allowing it to adapt to new data and improve accuracy for targeted applications.
		  
		  <!-- notionvc: c9c53eed-11d2-489e-8ed9-881d18d85ec7 -->
	- ### Vocabulary set 3 summary
	  collapsed:: true
		- **Bias**: In machine learning, bias refers to the
		  distortion in data that can lead to unfair and inaccurate model
		  outcomes. High-quality data is crucial to avoid biased results.
		- **Hallucinations**: AI hallucinations occur when AI
		  generates false responses that seem factual. They can result from biased data or misinterpretation during training.
		- **Temperature**: This parameter in AI models controls
		  the randomness of output. A lower temperature results in more focused
		  responses, while a higher temperature leads to more diverse outputs.
		- **Anthropomorphism**: This is the attribution of human
		  traits to non-human entities, including AI. As AI becomes more complex,
		  people may start to anthropomorphize it, affecting interactions.
		- **Completion**: In NLP, completion refers to the output generated by a model in response to a prompt, such as an answer from a chatbot.
		- **Tokens**: Tokens are the basic units of text input for AI models, which can be words or parts of words. They are essential for processing and generating responses.
		- **Emergence in AI**: Emergence occurs when large models
		  exhibit unexpected behaviors not seen in smaller models, potentially
		  leading to unanticipated and harmful outcomes.
		- **Embeddings**: These are numerical representations of
		  data in multi-dimensional space, capturing semantic relationships. They
		  enhance AI's ability to understand language and images efficiently.
		- **Text Classification**: This involves training models
		  to categorize text based on content, using NLP to understand patterns
		  and context for tasks like sentiment analysis and topic categorization.
		- **Context Window**: This refers to the amount of text an AI model can process at once, determined by the number of tokens. It
		  influences how prompts are engineered for effective responses.
		  
		  <!-- notionvc: 12bad55a-4c14-4c38-8f4f-f16cc8f6aee0 -->
- Day 4
  collapsed:: true
	- ### Ethics and Concern
		- **Data Privacy Concerns**:
			- Generative AI tools like ChatGPT can pose risks to company and user data, as seen in a breach in March 2023.
			- Companies should read privacy statements, align their privacy
			  policies with service providers, and implement additional security
			  checks.
			- Anonymizing user data before using third-party services is recommended to mitigate risks.
		- **Generative AI Bias**:
			- Generative AI can inherit biases from its training data, leading to stereotypes, such as gender biases in generated images.
			- To combat bias, expand datasets, test for existing biases, and be transparent about biases when they cannot be removed.
		- **IP and Copyright Issues**
			- Using Generative AI to create content can lead to IP violations if copyrighted material is involved.
			- In the U.S., works created with Generative AI are not eligible for copyright protection.
			- Companies should be cautious about using Generative AI tools to avoid potential legal issues related to IP and copyright.
			  
			  Tools explored:
			   [Huggingface Bias Explorer](https://huggingface.co/spaces/society-ethics/DiffusionBiasExplorer) 
			   DiffusionBee Tool
	-
	- @@html: {{video https://youtu.be/odgLX52Ulyg}}@@
- Day 5
  collapsed:: true
	- # Day 5
	- Model access
		- By default not all model are enabled to account ,we need to get them enabled under model access section in Bedrock
	- Bedrock Playground
		- Amazon Bedrock's Playgrounds allow users to experiment with different foundation models through a graphical interface, enabling them to
		  determine the most suitable model for their needs.
		- These playgrounds support text, chat, and image-based generative AI
		  applications, offering flexibility in design and development.
		- Users can manipulate prompts and inference parameters to influence
		  model responses, aiding in aligning outputs with specific use cases.
		- Parameters
			- Randomness and diversity parameters include Temperature and Top P, which affect the focus and diversity of model outputs.
			- Length parameters, such as Max completion length and Stop sequences,
			  control the length and stopping points of generated responses.
			- Repetition parameters, including Presence penalty, Count penalty,
			  Frequency penalty, and Penalize special tokens, manage the repetition of tokens in outputs.
			- Adjusting these parameters can significantly alter the completion results
		- Model metrics, such as latency and cost, are available in the Chat
		  playground to help users evaluate model performance and suitability for
		  their use cases.
		- Users can define specific metric criteria to assess if a model meets
		  their requirements, with visual indicators highlighting any
		  discrepancies.
	- Selecting right Models
	    
	    Methods
		- Evaluations can be conducted in three modes: Automatic, Human: Bring
		  your own work team, and Human: AWS Managed work team, with human modes
		  incorporating human judgment.
		- Automatic evaluations involve an 8-step process: selecting a
		  foundation model, task type, metrics, dataset, specifying S3 storage
		  location, selecting IAM role, inference and scoring, and viewing
		  results.
		- Task types for automatic evaluations include general text generation, text summarization, question and answer, and text classification, with
		  metrics like toxicity, accuracy, and robustness.
		- Human evaluations allow for up to two models to be reviewed, with an
		  additional 'Custom' task type for tailored evaluations, and involve
		  setting up a work team, defining metrics, and providing instructions.
		- Human: Bring your own work team evaluations involve selecting models, task types, metrics, dataset location, S3 storage, setting permissions, setting up a work team, providing instructions, submitting the job,
		  completing workforce tasks, and viewing results.
		- Human: AWS Managed work team evaluations require naming the
		  evaluation, scheduling a consultation with AWS, liaising to finalize
		  requirements, and creating the job, with AWS managing the workforce and
		  criteria.
	- Setting up AWS Bedrock API
		- 3 ways:
		    
		    SDK
		    
		    CLI
		    
		    Sagemaker Notebook
		- permissions to access API
			- The AWS SDK supports multiple programming languages, including C++, Go,
			  Java, JavaScript, .NET, Python (Boto3), and Ruby. Each language has its
			  own way of interacting with the Bedrock APIs.
			- To use the AWS CLI, users must download, install, and configure it with the necessary permissions to access Amazon Bedrock.
			- When using an Amazon SageMaker notebook, the role associated with
			  the notebook must have specific permissions to access Amazon Bedrock.
			  This includes an inline policy allowing all Bedrock actions and a trust
			  relationship policy allowing Bedrock and SageMaker services to assume
			  the role.
			- The trust relationship policy is resource-based, defining which
			  entities can assume the role, granting full access to Amazon Bedrock as
			  specified in the inline policy.
			- With a SageMaker notebook, users can utilize the Python (Boto3) SDK to perform and invoke API operations.
	- Evaluate Model performance. - Lab
	    
	    [https://aws.amazon.com/blogs/aws/amazon-bedrock-model-evaluation-is-now-generally-available/](https://aws.amazon.com/blogs/aws/amazon-bedrock-model-evaluation-is-now-generally-available/)
	    
	    Doc page: [https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-jobs-management-create.html](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-jobs-management-create.html)
		- Step References:
			- Sample Dataset:
			    
			    ```jsx
			    {"prompt":"The chemical symbol for gold is", "category":"Chemistry", "referenceResponse":"Au"}
			    {"prompt":"The tallest mountain in the world is", "category":"Geography", "referenceResponse":"Mount Everest"}
			    {"prompt":"The author of 'Great Expectations' is", "category":"Literature", "referenceResponse":"Charles Dickens"}
			    ```
			- Bucket CORS setting for dataset source
			    
			  
			  ```[
			    {
			    "AllowedHeaders": [
			    "*"
			    ],
			    "AllowedMethods": [
			    "GET",
			    "PUT",
			    "POST",
			    "DELETE"
			    ],
			    "AllowedOrigins": [
			    "*"
			    ],
			    "ExposeHeaders": [
			    "Access-Control-Allow-Origin"
			    ]
			    }
			    ]
			  ```
			- Evaluation steps
			    
			    In Bedrock
			    
			  ![image.png](../assets/image_1742582968658_0.png)
- Day 6 
  collapsed:: true
	- ### Prompt Engineering
	- Prompt Engineering Intro
		- Prompts are inputs given to generative AI systems to guide their output, with text-based prompts being the most common.
		- Prompt engineering involves crafting prompts that consistently yield the desired output, combining creativity and an iterative refinement
		  process.
		- The importance of prompt engineering lies in its ability to
		  transform interactions with AI, enabling natural language interfaces
		  over traditional user interfaces.
		- Generative AI is integrated into various products, such as Adobe's
		  Photoshop and Illustrator, GitHub's Copilot, and Salesforce's Einstein,
		  enhancing their capabilities through prompts.
		- Effective prompt engineering is crucial for maximizing the value extracted from generative AI systems.
		- The field is emerging, requiring experimentation, creativity, and patience, with significant rewards for those who master it.
	- collapsed:: true
	  
	  Prompt Anatomy
		- Objective:
		  
		  Goal of the prompt. eg: Summarize the document
		- **Context**: Additional information provided to help generate the desired response, though not always required.
		- **Markers**: Used to indicate specific sections of the prompt, aiding the model in understanding its structure.
	- Prompt creation process
	  collapsed:: true
		- Two main factors:
			- Objective. - Clearly define it
			- Verification process - same as verifying the code is that achieves the objective
		- Choose the right model for your use case
		- Prompt experiments
			- Experiment with prompt by varying parameters like temperature to understand the response
		- Prompt Analysis
			- Try to generate same results with different prompts and see how the prompt influences the response
		- Refinement
			- Refine your prompt further by cuttingout unwanted words in the prompt
		- Model Experiments
			- Now experiment the same with different Models and see how consistency you get same results for same prompt
		- Document
			- Document the result and prompt on various models for future reference
	- Standard Prompt Strategies
		- collapsed:: true
		  
		  Instruction Prompts
			- eg: Translate text to language Tamil
			  
			  > Instructions can be limited in response
		- collapsed:: true
		  
		  Question based Prompts
			- You can get more options and conversational
			- Question-based prompts can be categorized into open, closed, and leading questions,
			  each serving different purposes and influencing the scope and direction
			  of the response.
			- Open questions allow for expansive answers, closed questions
			  anticipate specific answers, and leading questions guide the model
			  towards a particular response.
		- Instructional prompts are explicit and useful for tasks requiring specific formatting, while question-based prompts allow for exploration and conversational interaction.
	- Contextual Prompts
		- Contextual prompts are divided into two sub-categories: role-playing and scenario-based prompts.
		- Role-playing prompts engage the model in a specific role, such as a
		  historical figure or professional, to generate content from different
		  perspectives and simulate creative thinking.
		- Scenario-based prompts set a specific situation, time, or place for
		  the model to consider, allowing exploration of specific events or
		  environments.
		- Both types of prompts require a deep understanding of the topic to create believable and effective contexts.
		- The complexity of the role or scenario should be carefully considered to avoid oversimplification or bias.
		- Role-playing and scenario-based prompts can be used together for more effective results.
		- The process of creating these prompts involves defining objectives,
		  including rules for context, and iteratively crafting the prompt to
		  achieve the desired response.
	- Few Shot Prompts
	  
	  Few-shot learning provides context to models, enabling them to understand and perform tasks more effectively by recognizing patterns from given examples.
	  
	  zero-shot, one-shot, and few-shot prompts:
		- Zero-shot prompts do not include examples and rely on the model's existing knowledge.
		- One-shot prompts include a single example to guide the model.
		- Few-shot prompts include multiple examples to provide a clearer understanding of the task.
	- Chain of Thought Prompts
		- Language models have limitations in performing complex reasoning tasks, such as
		  arithmetic or common-sense reasoning, and chain-of-thought prompts aim
		  to mitigate these limitations.
		- An example problem involving counting apples is used to demonstrate
		  the model's reasoning process. Initially, the model makes a mistake by
		  misinterpreting the phrase "shared an apple" as two apples instead of
		  one.
		- In above case break the problem statement and do one action at a time
		- while models are advancing, they can still provide incorrect answers confidently, so users should verify responses and use prompts to achieve desired outcomes.
	- Evaluating Response Accuracy
		- Generative AI models can provide inaccurate information confidently, highlighting the need for verification of responses.
		- A real-world example is provided where a lawyer used ChatGPT to
		  create a legal brief with fabricated cases, emphasizing the importance
		  of verifying AI-generated information.
		- To ensure accuracy, responses should be cross-referenced with
		  multiple reliable sources, and discrepancies should be investigated.
		- Prompts can impact response accuracy; leading, ambiguous, and biased prompts can lead to inaccuracies.
		- Leading prompts guide responses towards specific outcomes, which can result in fabricated responses.
		- Ambiguous prompts can be interpreted in multiple ways, leading to
		  inaccurate responses, while specific prompts help guide models towards
		  accurate answers.
		- Biased prompts and models can produce inaccurate responses by reflecting biases present in training data.
		- A prompt review process involving diverse perspectives is recommended to identify and avoid biases.
	- Response Formatting
	  
	  Types of response
		- Unstructured data is raw model output, ideal for plaintext consumption or intermediate steps like text-to-speech.
		- Loosely structured data, such as Markdown, enhances readability with minimal syntax for formatting text.
		- Highly structured data formats like JSON, YAML, XML, and CSV are
		  used for software consumption, requiring strict adherence to rules.
		  
		  >  Producing highly structured formats can be challenging due to non-deterministic model behavior, complexity, token consumption, and cost.
		- Lower temperature settings are recommended for predictable formatting, while higher temperatures are for creative responses.
		- Custom formats can be beneficial for specific use cases, simplifying the process and reducing effort compared to standard formats.
		- Consider using `retry` mechanism
	- Response Qualities
		- Language models can produce grammatically correct responses due to their
		  training on diverse data, which can be used for proofreading and editing human-written content.
		- Style and tone are context-dependent, with formal styles suited for
		  legal briefs and casual styles for blog posts. Prompts can adjust these
		  qualities by specifying terms like formal, casual, or friendly.
		- Emotion and sentiment in responses are more subjective and complex
		  to control. Prompts should include desired emotions or sentiments,
		  considering the emotional weight of words and punctuation.
		- Emotion and sentiment analysis requires understanding the model's
		  capabilities and may need pre-processing to avoid skewed results,
		  especially in longer content.
		- It's important to verify the desired qualities with human review,
		  especially for significant use cases, to ensure accuracy and
		  appropriateness.
		  
		  <!-- notionvc: 789c9882-2ab6-4994-a5e1-80b9173db6b9 -->
- Day 7
  collapsed:: true
	- Bedrock Playground : Generate Response from Interface: Lab
	  
	  [https://docs.aws.amazon.com/bedrock/latest/userguide/inference.html](https://docs.aws.amazon.com/bedrock/latest/userguide/inference.html)
	  
	  <!-- notionvc: 4519b9ca-d515-4dfa-8a67-6bb7887b8464 -->
- Day 8
  collapsed:: true
	- Custom Models in Bedrock
	  collapsed:: true
		- Provisioned Throughput is required for using custom models
			- Specifies how much throughput your model will require and for how long
			- Provisioned throughput in Amazon Bedrock requires a long-term commitment and is ideal for consistent performance needs.
			- Provisioned throughput can be purchased through the Bedrock console under 'assessment and deployment.'
			- Users must specify the model, commitment duration (no commitment,
			  one month, or six months), and the number of model units (MUs) needed.
			- AWS does not specify the processing power of an MU; users should contact their account manager for details.
			- By default, accounts have two MUs for models with no commitments, but more can be requested for longer commitments.
			- Once purchased, the provisioned throughput settings cannot be changed.
			- The provisioned throughput can be used in the Bedrock playground for testing and deployed via the Bedrock API for application use.
		- Bedrock makes the clone of Foundation model and then customise it on top of it
		- All customised models are placed within your VPC
		- Once model is customised it can be accessed via Bedrock API
		- Finetuning and more
			- Fine-tuning is used to optimize a model for specific tasks using a small, labeled
			  dataset with examples of prompts and completions, formatted in JSONL.
			- Continued pre-training is suitable for enhancing a model's
			  performance in a specific domain using a large, unlabelled dataset, also in JSONL format.
			- To create a customization job, navigate to the Bedrock console,
			  select a base model from providers like Amazon, Cohere, or Meta, and
			  configure the job by naming it and specifying the dataset location.
			- Upload the dataset to an S3 bucket, adjust hyperparameters, and select a service role with S3 access for data permissions.
			- Once the job is created and run, the customized model can be tested
			  in a playground or used for inference via API, provided there is
			  sufficient provision throughput.
			  
			  <!-- notionvc: 00d61353-8d2d-4e74-a6fb-e80e5318ff2f -->
	- Lab [single prompt api via lambda](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-invoke.html)
- Day 9
  collapsed:: true
	- IAM
		- Evaluation order
			- SCP
			- Resource based policies
			- Permission boundaries
			- Identity based policies
			  
			  <!-- notionvc: 97eafc7a-3baf-436f-ab18-cc1ccb8ae91b -->
- Day 10
  collapsed:: true
	- ### Bedrock Knowledgebase
	  collapsed:: true
		- Knowledge base intro
		  collapsed:: true
			- Knowledge bases allow applications to use both foundation models and external data to provide more reliable and current information.
			- two main stages of knowledge base operations: pre-processing and runtime execution.
			- During pre-processing, data is divided into smaller segments,
			  transformed into embeddings, and stored in a vector index, which helps
			  in assessing semantic similarities between queries and data.
			- In runtime execution, user queries are translated into vectors,
			  matched against the vector index to find semantically similar document
			  segments, and used to generate a tailored response with additional
			  context.
		- Building a knowledge base
			- 4 step process
				- **Step 1: Provide Knowledge Base Details**
					- Assign a name and description to differentiate between multiple knowledge bases.
					- Select an IAM role with permissions for accessing necessary services like Amazon Bedrock, S3, OpenSearch, or Aurora.
					- Optionally, use KMS for encryption and add metadata tags for management.
				- **Step 2: Configure Data Source**
					- Name the data source and provide its URI, which must be stored on Amazon S3.
					- Supported formats include .txt, .md, .html, .doc/.docx, .csv, .xls/.xlsx, and .pdf.
					- Configure encryption settings and choose a chunking strategy for data segmentation.
					- Set a data retention policy and add up to five data sources.
				- **Step 3: Select Embeddings Model and Configure Vector Store**
					- Choose an embedding model, noting that costs vary by model.
					- Create a vector store, either by quick creation (for development) or selecting an existing one.
					- Options include Amazon OpenSearch Serverless, Aurora, Pinecone, and Redis Enterprise Cloud.
				- **Step 4: Review and Create**
					- Verify all configurations before creating the Knowledge Base.
					- Perform a sync to ingest and index data sources, with incremental syncs needed for any data source changes.
		- Interacting with knowledge baseChunk
			- RetrieveAndGenerate API, which allows querying the knowledge base and generating responses based on relevant sources.
			- The Retrieve API is highlighted for directly accessing and retrieving information from the knowledge base.
			- Amazon Bedrock Agents are mentioned as tools to help users automate
			  tasks and interact with the knowledge base through API calls.
			- Additional resources and detailed information about Amazon Bedrock API actions are available through provided URLs.
		- Types of Chunking mechanism
			- **Default chunking** will segment the data into a slice of approximately 300 tokens in
			  size. These tokens can be a whole word, just the beginning or the word,
			  the end, spaces, single characters, and anything in between.
			- **Fixed size chunking** will chunk your data into segments of approximately the same size, with the smallest chunk being 20, and the maximum allowed of 8192. However,
			  this will depend on the model being used.
			- **No chunking** will simply use each file defined as a single chunk.
		- Amazon Bedrock Agents
			- Amazon Bedrock Agents function as automated helpers in AI
			  applications, streamlining development and enhancing user interactions.
			- They facilitate integration between AI components, linking
			  foundation models with data sources and managing user engagement and API requests.
			- Agents help reduce development time and costs by allowing teams to focus on business objectives rather than coding tasks.
			- They enhance customer experience through automation, such as
			  executing tasks like booking reservations while interacting with users
			  in natural language.
			- Managed by Amazon Bedrock, Agents eliminate the need for
			  infrastructure management, handling tasks like encryption and
			  permissions.
			- Agents can converse with users, complete tasks, break down tasks,
			  make API calls, integrate with knowledge bases, and manage source
			  attribution.
			- The Build-time Execution phase involves building and configuring the Agent, associating it with a foundation model, and setting instructions for tasks.
			- Optional components like Action Groups and Knowledge Bases can be added for advanced orchestration.
			- Prompt Templates are used to create prompts for automated orchestration, allowing customization of prompts and actions.
			- Run-time Execution involves preprocessing user requests,
			  orchestrating actions, and generating responses through a sequence of
			  phases.
			- Post-processing, if enabled, allows the Agent to generate a final response to the user.
			  
			  [https://docs.aws.amazon.com/bedrock/latest/userguide/agents-create.html](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-create.html)
			  
			  <!-- notionvc: 6b89ff0e-8837-4122-88e9-613ce4b7ae70 -->
- Day 11
  collapsed:: true
	- Aurora RDS as knowledge base to Bedrock
	  collapsed:: true
		- [https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/AuroraPostgreSQL.VectorDB.html](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/AuroraPostgreSQL.VectorDB.html)
		  
		  [https://youtu.be/MpmV-MehtWs?si=FoIgGmS5j9xZEYAJ](https://youtu.be/MpmV-MehtWs?si=FoIgGmS5j9xZEYAJ)
	- collapsed:: true
	  
	  **Enhancing Generative AI Models With Retrieval-Augmented Generation (RAG)**
		- RAG vs Finetuning vs Prompt Engineering
		- ### RAG vs. Fine-tuning vs. Prompt engineering
		  
		  Fine-tuning a model involves training the model on a specific dataset to improve
		  its performance on a particular task. This process involves adjusting
		  LLM parameters using task-specific data.
		  
		  RAG, on the other hand, focuses on enhancing the quality of the model's responses by
		  incorporating external information. These two approaches occur at
		  different stages of the model's lifecycle; fine-tuning happens during
		  training, while RAG happens during generation.
		  
		  Prompt engineering is the process of refining a model's input to improve the quality of its
		  output. This process ends with a well-crafted prompt that can impact
		  simple tasks such as text translation, summarization, and keyword
		  extraction. However, the response is still dependent on the model's
		  training data and knowledge.
		  
		  RAG goes a step beyond prompt engineering as it augments the prompt with relevant retrieved data as additional context.
	- ### RAG Processes
	  collapsed:: true
		- At a high level, Retrieval-Augmented Generation involves the following processes:
		- Retrieving information from an external source and ranking the information based on relevance
		- Augmenting an LLM prompt to include the top-ranked data as additional context for the LLM
		- Generating the response using the augmented prompt
		- ### Sourcing and indexing external data
		  
		  Before the RAG process begins, external data must be sourced and indexed. This
		  involves collecting relevant data from authoritative sources and
		  organizing it in a way that makes it easy to retrieve.
		  
		  The most common approach is to use vector indexing, where data is organized based
		  on numerical vectors that represent keywords or phrases. This allows
		  the retrieval system to capture semantic relationships between words and
		  phrases.
		  
		  Another word for this process is "embedding". In generative AI, embedding is the process of converting words, images, or videos into numerical vectors. The terms "embeddings" and "vectors" are
		  often used interchangeably in the context of RAG and generative AI.
		  
		  After the external data has been embedded, it can be organized and stored in a vector store or database for quick retrieval.
		- ### Retrieval
		  
		  A relevancy search is performed on the embeddings to retrieve the most
		  relevant data based on a user query. The user query is embedded using
		  the same process as the external data. Once the query has been embedded,
		  it is compared to the indexed external data embeddings to find the most
		  relevant information.
		  
		  This process returns the top-ranked data based on the similarity between the query and the external data.
		- ### Augmentation
		  
		  The original user query is combined with the top-ranked data to create an
		  augmented prompt. Prompt engineering techniques can be used to refine
		  the augmented prompt and ensure that it is compatible with the LLM.
		- ### Generation
		  
		  The augmented prompt is fed into the LLM to generate a more accurate and relevant response.
		  
		  Lab : [https://github.com/aws-samples/amazon-bedrock-rag-workshop/tree/main](https://github.com/aws-samples/amazon-bedrock-rag-workshop/tree/main)
		  
		  <!-- notionvc: b65536cb-79ac-4a8c-8a51-9a09c9f2cf58 -->
- Day 12
  collapsed:: true
	- ### Guidelines of responsible AI
	  collapsed:: true
		- First four dimension of Responsible AI
		  collapsed:: true
			- **Fairness**: AI systems should treat all features and groups equitably, avoiding bias that can occur in training data or during deployment.
			- **Explainability**: AI outputs should be
			  understandable, allowing users to rationalize predictions. Tools like
			  SHAP can help identify influential features, and simpler models can
			  enhance interpretability.
			- **Transparency**: AI systems should be well-documented, providing visibility into their design, training, and deployment
			  processes. This ensures ethical disclosure to stakeholders.
			- **Controllability**: Humans should maintain control
			  over AI systems, with mechanisms for feedback and intervention to
			  prevent AI from deviating or misbehaving.
		- Last four dimension of ResponsibleAI
		  collapsed:: true
			- Veracity and robustness focus on ensuring data accuracy and model reliability, even under attacks or disturbances.
			- Governance involves compliance with policies, procedures, and
			  regulations, including internal ethics and external laws like GDPR.
			- Safety aims to prevent harm to users by implementing fail-safes and risk assessments in AI systems.
			- Privacy and security protect AI systems from breaches and ensure data is anonymized and safely collected.
		- Importance of datasets
		  collapsed:: true
			- The data used in AI systems is crucial for creating a safe and responsible AI environment, impacting performance and fairness.
			- Key characteristics of ethical AI datasets include inclusivity, diversity, balance, and curation.
			- Inclusivity ensures datasets represent a variety of populations and
			  perspectives, preventing marginalization and under-representation.
			- Diversity involves having data from various demographics, such as age, ethnicity, gender, and geography, to avoid biased models.
			- A balanced dataset ensures equal representation of all groups, preventing bias towards majority populations.
			- Curated datasets come from reputable sources, are well-documented,
			  and free from duplicates and noise, ensuring data quality and
			  reliability.
			- Overall, datasets should be explainable, curated, balanced, diverse, and inclusive to ensure ethical AI development.
			  
			  <!-- notionvc: dab6be9c-e544-4e3a-9916-6c72e0775e58 -->
	- ### Bedrock Security and Privacy
	  collapsed:: true
		- Intro
		  collapsed:: true
			- Customer data in Bedrock is encrypted both in transit and at rest, using TLS and AWS Key Management Service (KMS) for managing encryption keys.
			- Bedrock ensures that customer data is not used to train models and is not stored in service logs.
			- When fine-tuning models, Bedrock creates a unique copy of the model
			  for private training with customer data, allowing full control over data sharing.
			- AWS PrivateLink can be used to ensure secure traffic between a VPC and Amazon Bedrock, avoiding the public internet.
		- Auditing at Bedrock
			- Auditing is supported through AWS CloudTrail for tracking user activity and
			  Amazon CloudWatch for monitoring metrics and logging model invocations.
			- Data sovereignty is maintained by storing Bedrock data in the AWS region of use.
			- Amazon Bedrock complies with various standards, including ISO, SOC, CSA STAR Level 2, HIPAA, and GDPR.
			- AWS Artifact provides more information on compliance standards.
		- Guardrails at Bedrock
			- Amazon Bedrock includes automated mechanisms to detect and prevent misuse, but Guardrails provide enhanced security.
			- Guardrails offer granular controls for administrators to filter both user inputs and model outputs, ensuring interactions remain
			  appropriate.
			- Four categories of policies can be defined:
				- Denied topics to avoid certain subjects.
				- Content filters to set thresholds for harmful content.
				- PII redaction to protect user privacy by removing personally identifiable information.
				- Word filters to block specific words or phrases.
			- Guardrails provide an additional layer of protection beyond the
			  built-in safeguards of foundation models and can be consistently applied across applications.
			- They can also be deployed with Amazon Bedrock Agents, which assist employees with tasks like querying internal databases.
			  
			  <!-- notionvc: 7596fa90-743a-4f9f-baea-5211a7ec8100 -->
- Day 13
  collapsed:: true
	- Lab: Guardrail in Bedrock
		- https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-create.html
	-
- Day 14
  collapsed:: true
	- Sagemaker and ML workflow
		- Amazon SageMaker is a fully managed service for building, training, and
		  deploying machine learning models, catering to data scientists, machine
		  learning engineers, and developers.
		- It offers various environments like SageMaker Studio, Studio
		  Classic, Studio Lab, and IDE options such as Code Editor and Notebook
		  Instances, which support languages like R and Python and include
		  pre-installed libraries like TensorFlow, PyTorch, and Scikit-learn.
		- The machine learning workflow in SageMaker involves four main steps: data preparation, model building, model training, and model deployment.
		- Data preparation is crucial and often the most time-consuming step, involving data collection, cleaning, and transformation.
		- Model building follows data preparation, where SageMaker provides
		  frameworks and pre-trained models to assist in selecting the right
		  model.
		- Model training involves setting up training runs, evaluation
		  conditions, and hyperparameter tuning to optimize model performance.
		- After training, the model is deployed into a production environment, where continuous monitoring is necessary to manage performance changes
		  and hardware scaling.
	- Preparing your data for sagemaker
		- The quality of data is crucial when building and training machine
		  learning models, emphasizing the importance of clean and well-structured data.
		- Initial data preparation involves cleaning up data by removing
		  duplicates, handling null or missing values, and performing necessary
		  transformations like encoding and normalizing data.
		- Depending on dataset size, sampling or augmenting data might be necessary to balance efficiency, cost, and avoid oversampling.
		- Amazon SageMaker offers tools like Data Wrangler and Ground Truth to simplify data preparation.
		- Data Wrangler provides a visual interface for preparing data with
		  minimal coding, allowing for data transformations and creating
		  repeatable workflows.
		- Ground Truth assists in large-scale data labeling, offering semi-automated solutions involving human and machine collaboration.
		- Additional SageMaker features include Processing for distributed
		  data jobs, Feature Store for managing machine learning features, and
		  Clarify for detecting potential bias in models.
		  
		  >
	- Wrangler for data visualisation
	  
	  > 
	  
	  Ground truth to create labelling jobs
	- Sagemaker Notebooks
		- SageMaker Notebook Instances are managed Jupyter Notebooks that facilitate data
		  exploration, preprocessing, feature engineering, and model development
		  using frameworks like TensorFlow, PyTorch, and Scikit-learn.
		- These instances are pre-configured with necessary libraries and tools, eliminating the need to manage infrastructure.
		- Users can choose different instance types, with larger ones offering faster data processing but at a higher cost.
		- The setup includes specifying an IEM role, optional KMS key for encryption, and associating with a Git repository.
		- Once provisioned, users can access Jupyter or Jupyter Lab, which
		  offers numerous example notebooks for various tasks, including Amazon
		  algorithms and SageMaker processing.
		- Example notebooks come with annotations, sample data, and code, aiding in model building without starting from scratch.
		- Users can perform data labeling directly in SageMaker Notebooks
		  using Python, bypassing tools like Data Wrangler or Ground Truth.
		- SageMaker Autopilot automates the ML process by selecting optimal
		  algorithms, while SageMaker JumpStart provides pre-trained models and
		  templates for customization.
	- Training ML model with sagemaker
		- Begin training and tuning your model after building it. For smaller
		  datasets, training can be done directly in a Notebook instance, but
		  larger datasets or complex processes benefit from using SageMaker's
		  training platform.
		- Choose an algorithm for training. SageMaker offers over 15 built-in
		  algorithms for various machine learning tasks, including supervised,
		  unsupervised learning, and natural language processing. Custom and
		  third-party algorithms are also supported.
		- Create a training job by naming it, setting permissions, and
		  selecting an algorithm. Choose between file mode (for smaller datasets)
		  and pipe mode (for larger datasets) for data input.
		- Configure hardware resources for the training job, including
		  instance type, number of instances, and storage. Consider using managed
		  spot instances to reduce costs, with SageMaker handling interruptions.
		- Set hyperparameters specific to the chosen algorithm to control
		  training. Use hyperparameter tuning jobs to automate the process and
		  optimize metrics like accuracy.
		- Configure input data channels from sources like S3, EFS, or FSx for
		  Lustre, and separate train and test channels for model evaluation.
		- Specify an output path in S3 for storing training results. Monitor the training job's progress in the console.
		- Utilize SageMaker Debugger for debugging and optimizing training
		  jobs, and SageMaker Distributed Training for scaling jobs across
		  multiple instances.
		- Explore hands-on labs to gain experience with building and training models using Amazon SageMaker.
	- Deploy ML models
		- Four inference options are highlighted: real-time inference with provisioned endpoints, serverless inference with serverless endpoints, asynchronous inference for larger payloads, and batch transform jobs for large
		  datasets.
		- SageMaker endpoints require configuration, with two types:
		  provisioned (requiring upfront infrastructure) and serverless (scaling
		  automatically).
		- Serverless endpoints are cost-effective and suitable for workloads
		  without strict latency requirements, while provisioned endpoints are
		  ideal for real-time workloads with low latency.
		- Asynchronous inference allows for queuing requests and receiving
		  notifications upon completion, with options for data capture and
		  analysis.
		- Batch transform jobs handle large payloads and long runtimes, requiring specification of input and output data locations.
		- SageMaker Inference Recommender can help choose the best deployment configuration.
		- Post-deployment, continuous monitoring is necessary to manage model
		  performance and lifecycle, with tools like SageMaker Shadow Tests for
		  assessing changes.
		  
		  > 
		  
		  Sagemaker shadow support -  To test pre prod
	- summary
		- SageMaker offers various environments and tools, including SageMaker Studio,
		  Studio Classic, Studio Lab, Code Editor, and Notebook Instances, which
		  support machine learning tasks with pre-installed libraries like
		  TensorFlow, PyTorch, and Scikit-learn.
		- The four-step machine learning workflow was discussed: data preparation, model building, training and tuning, and deployment.
		- Data preparation is crucial and time-consuming, with tools like Data Wrangler and Ground Truth aiding in the process.
		- SageMaker Notebook Instances facilitate model building by providing managed Jupyter Notebooks with numerous example notebooks.
		- Training involves choosing algorithms, with SageMaker offering over
		  fifteen built-in options and support for third-party and custom
		  algorithms.
		- Hyperparameter tuning is automated through SageMaker, allowing for optimization of model performance.
		- Deployment options include real-time inference, serverless inference, asynchronous inference, and batch transform jobs.
		- Continuous monitoring of deployed models is necessary to manage their lifecycle and adapt to changes in data and paradigms.
		  
		  <!-- notionvc: 80fe6661-2743-4f7f-ab97-e6b48727619c -->
- Day 15
  collapsed:: true
	- ### Texttract
	- Intro for Textract
	  collapsed:: true
		- Amazon Textract is a machine learning service that automates the
		  extraction of text and data from various documents, including typed and
		  handwritten text in financial reports, medical records, and tax forms.
		- It addresses the challenges of manual data extraction, which is
		  often time-consuming and costly, by processing documents automatically
		  and accurately without manual intervention.
		- Textract offers pre-trained and customizable features, enabling
		  businesses to automate tasks like loan processing and information
		  extraction from invoices and receipts, significantly speeding up
		  workflows and reducing costs.
		- The service is easy to integrate into applications using a simple
		  API, allowing for quick analysis and data extraction from millions of
		  documents, with a pay-as-you-go pricing model.
		- Key features include the AnalyzeDocument API for extracting
		  information from structured documents, the AnalyzeExpense API for
		  processing financial documents, and the AnalyzeID API for analyzing ID
		  documents.
		- Textract supports various use cases, such as digitizing legal
		  contracts for easy search and retrieval, enhancing natural language
		  processing in healthcare, and automating data extraction for financial
		  analysis and insurance claims processing.
	- How it works?
	  collapsed:: true
		- Amazon Textract is a service that processes documents in various
		  formats like PDF, JPEG, and PNG, supporting both single and multi-page
		  documents.
		- It uses machine learning to identify text elements and their
		  relationships, performing five key types of document extraction: text,
		  forms, tables, query responses, and signatures.
		- Text extraction retrieves raw text, form extraction identifies
		  key-value pairs, table extraction processes tabular data, query
		  responses extract specific information, and signature detection
		  identifies signatures.
		- Textract can handle both structured and unstructured data, making it versatile for different document processing needs, such as analyzing
		  invoices and receipts without templates.
		- It can identify vendor names even when not explicitly labeled and standardizes terms across different documents.
		- Textract provides detailed output as block objects, which include
		  pages, lines, words, form data, tables, and layout information, with
		  confidence scores indicating accuracy.
		- The service integrates with other AWS services and allows developers to automate text and data extraction, improving efficiency and reducing errors in document processing tasks.
	- Custom queries in Textract
	  collapsed:: true
		- Amazon Textract allows customization of its output using custom queries to better fit specific business needs.
		- Users can create adapters by annotating and labeling sample documents to identify the information they want to extract.
		- Documents can be uploaded from a computer or imported from an Amazon S3 bucket for training and testing datasets.
		- The adapter learns unique patterns and structures from the
		  documents, and auto-labeling can detect similar fields across documents.
		- Each adapter is assigned an ID and version, which are used in
		  requests to the AnalyzeDocument API or StartDocumentAnalysis operation.
		- Multiple adapters and versions can be specified for different pages in multi-page documents.
		- Adapter performance can be improved by adding more documents to the
		  training dataset and retraining, with performance evaluated using
		  metrics like F1 Score, Precision, and Recall.
		- Custom queries enhance data extraction accuracy for specific
		  document formats, benefiting industries like healthcare, banking, and
		  insurance by tailoring the model to extract relevant information
		  accurately.
	- Processing Docs sync vs assync
	  collapsed:: true
		- Synchronous operations provide real-time processing, returning immediate results,
		  ideal for quick document analysis and customer-facing applications.
		- They are suitable for small batches of documents and time-sensitive
		  workflows but have limitations like a 10 MB document size cap and a
		  30-second processing time limit.
		- Asynchronous operations allow for processing without immediate
		  results, beneficial for large documents or batches, without the size and time constraints of synchronous operations.
		- Asynchronous operations are ideal for high-volume scenarios and integrate well with systems that handle background processing.
		- Amazon Textract offers specific APIs for both synchronous and
		  asynchronous operations, each suited for different document processing
		  needs.
	- Best practices for Textract
	  collapsed:: true
		- Ensure documents are in a language supported by Amazon Textract
		  (English, Spanish, German, Italian, French, Portuguese) and have a
		  high-quality image resolution of at least 150 DPI.
		- Use supported file formats (PDF, JPEG, PNG) without converting or downsampling before uploading.
		- For tables, ensure they are visually distinct from other elements and text is upright.
		- Optimize document images for synchronous operations to improve speed and accuracy, handle errors effectively, and consider document size
		  limits.
		- For asynchronous operations, use a robust polling mechanism with
		  exponential backoff and consider using Amazon SNS for job completion
		  notifications.
		- Store results in Amazon S3 for scalable and durable data management.
		- Pay attention to confidence scores (0-100) to determine the
		  reliability of predictions, setting thresholds based on application
		  sensitivity.
		- Incorporate human review for sensitive applications to enhance accuracy.
		- For custom queries, use diverse sample data, logical and specific query structures, and maintain consistent annotation styles.
	- Security in Textract
	  collapsed:: true
		- Amazon Textract operates under the AWS Shared Responsibility Model, where AWS
		  secures the infrastructure, and customers manage content security.
		- Use AWS IAM Identity Center to set up user accounts with appropriate permissions and enable multi-factor authentication for added security.
		- Secure data transmissions with SSL or TLS to protect data in transit.
		- Enable AWS CloudTrail for logging API and user activities to maintain visibility and accountability.
		- Utilize AWS encryption solutions and default security controls to strengthen data protection.
		- For sensitive data, consider using advanced security services like Amazon Macie to discover and protect sensitive information.
		- Exercise caution when handling sensitive data to avoid exposing it in diagnostic logs.
		- Amazon Textract uses robust encryption methods for data at rest and
		  in transit, including server-side encryption with Amazon S3-managed keys or AWS Key Management Service.
		- The service uses transport layer security and VPC endpoints for secure communication during document processing.
	- Logging and Monitoring
	  collapsed:: true
		- Amazon Textract integrates with Amazon CloudWatch for comprehensive
		  monitoring, allowing tracking of individual operations and global
		  metrics.
		- CloudWatch can monitor metrics like server errors and success rates, and set up alarms for specific thresholds to maintain workflow
		  reliability.
		- CloudWatch alarms respond to sustained state changes, triggering actions like notifications if conditions persist.
		- AWS CloudTrail provides logging capabilities, capturing all API calls to Amazon Textract for security and compliance.
		- CloudTrail logs include details like requester's identity, helping track actions and maintain an audit trail.
		- Amazon Textract supports resource tagging for better organization
		  and access management, allowing efficient categorization and permission
		  control.
	- ### Polly
	- Helps to achieve text to speech
	- Getting started with Polly
		- Amazon Polly can be integrated into applications using AWS software
		  development kits (SDKs), the AWS management console, or the command line interface (CLI).
		- The AWS SDK for Python is recommended for creating a Polly client, sending text synthesis requests, and handling audio streams.
		- Users can generate high-quality speech for various applications,
		  such as e-learning platforms and customer service, using the SDK.
		- The Amazon Polly console allows users to select engine types,
		  languages, and voices to synthesize speech, which can be listened to
		  directly.
		- The AWS CLI can also be used to interact with Amazon Polly, though
		  it requires saving the output to a file to listen to the synthesized
		  speech.
		- To use the AWS CLI, users must download and configure it, create a named profile, and verify the setup.
	- Language offerings
		- Amazon Polly offers a variety of lifelike voices and supports multiple languages, allowing for global application development.
		- It provides voices for languages such as Arabic, Chinese, Danish,
		  Dutch, and English, with each voice optimized for accurate pronunciation and natural intonation.
		- Users can test voices using their own text in the AWS Management
		  Console, with most languages offering at least one male and one female
		  voice.
		- Voice speed varies naturally among different voices, and users can
		  adjust speed using Speech Synthesis Markup Language (SSML) tags.
		- SSML allows for preset speed adjustments (extra slow to extra fast) or custom speed settings between 20% and 200%.
		- The Brand Voice feature enables the creation of custom voices that
		  represent a brand's persona, using neural text-to-speech technology and
		  generative AI.
		- Amazon Polly can train neural voices for specific speaking styles,
		  such as a newscaster style, to match different situational speech
		  patterns.
	- Voice Engines in polly
		- Amazon Polly offers a wide range of lifelike voices in multiple languages, suitable for global applications.
		- The standard engine uses concatenative synthesis, piecing together recorded phonemes to create natural-sounding speech.
		- The neural text-to-speech engine uses advanced techniques to produce more natural and high-quality voices by converting phonemes into
		  spectrograms and then into audio signals.
		- The long-form engine is designed for engaging, human-like voices
		  suitable for lengthy content, using deep learning to replicate speech
		  nuances and emotions.
		- The generative engine employs a billion-parameter transformer and
		  convolution-based decoder to create highly adaptive and emotionally
		  engaging voices, leveraging large language models.
		- These technologies enable the creation of realistic and engaging
		  synthetic speech for various applications, such as customer assistants
		  and virtual trainers.
	- SSML with Polly
		- SSML is an XML-based language that allows users to control various aspects
		  of speech, such as pronunciation, volume, pitch, and speed.
		- By incorporating SSML tags, users can instruct Amazon Polly to
		  include pauses, change speech rate or pitch, emphasize words, use
		  phonetic pronunciation, and add effects like breathing or whispering.
		- In the AWS management console, users can enter SSML-enhanced text to customize speech output beyond default settings.
		- The prosody element in SSML can adjust volume, pitch, and speed, and these attributes can be combined for simultaneous adjustments.
		- Breathing sounds and whisper effects can be added using specific SSML tags to make speech sound more natural.
		- The emphasis element highlights specific words or phrases, making them stand out in the speech output.
		- The say-as element provides context for how text should be spoken,
		  ensuring correct pronunciation of numbers, dates, and other text types.
		- Overall, SSML allows for the creation of dynamic and engaging speech tailored to specific needs.
		  
		  <!-- notionvc: d6eb59a6-5b21-410c-b5be-0b0779a1a2aa -->
- Day 16
  collapsed:: true
	- Rekognition
	- Cpmprehend
	- Textract
- Day 17
  collapsed:: true
	- > This Day is marathon means we spent longer dueration than 45min to speedup the process
	-
	- Sagemaker for MLops
	  collapsed:: true
		- SageMaker integrates with AWS DevOps tools like CodeCommit, CodeBuild, 
		  CodeDeploy, and CodePipeline, facilitating continuous delivery and 
		  deployment for machine learning applications.
		- CodePipeline supports rapid feature updates and model improvements, 
		  with visualization capabilities aiding in process monitoring and 
		  tracking.
		- SageMaker Model Monitor automates the tracking of model performance,
		  detecting deviations and drifts, and alerting data scientists for 
		  timely retraining.
		- SageMaker Pipelines automate the entire model building process, from
		  data preparation to model validation, with flexible execution options, 
		  including scheduled runs or event-triggered processes.
	- Sagemaker Studio
	  collapsed:: true
		- Amazon SageMaker Studio is an integrated development environment designed to streamline machine learning workflows.
		- It combines various functionalities, allowing users to write code, 
		  perform visualizations, debug, track, and monitor model performance in a
		  single platform.
		- SageMaker Studio works with SageMaker Pipelines to automate and manage end-to-end machine learning workflows.
		- SageMaker Debugger helps diagnose and resolve issues like 
		  overfitting and vanishing gradients during training, offering insights 
		  and recommendations for improvement.
		- Amazon SageMaker Model Monitor continuously checks models in 
		  production for anomalies, such as data drift, and alerts users to 
		  maintain model accuracy.
		- The platform supports managing numerous training jobs by organizing,
		  tracking, and comparing experiments, enhancing workflow efficiency.
	- Sagemaker pipeline
	  collapsed:: true
		- Amazon SageMaker Model Building Pipelines is a workflow 
		  orchestration service designed to automate the machine learning process,
		  from data pre-processing to model monitoring.
		- It integrates CICD practices into machine learning workflows, 
		  optimizing training and deployment capabilities while reducing the need 
		  for extensive workflow tools.
		- SageMaker Pipelines offers prebuilt templates, making it accessible 
		  even to those with limited CICD knowledge, and allows pipeline creation 
		  using the SageMaker Python SDK or JSON Schema.
		- The service uses a Directed Acyclic Graph (DAG) to manage the 
		  execution order of pipeline steps, ensuring a one-way flow of data 
		  without loops.
		- Pipelines consist of various steps like processing, training, and 
		  evaluation, with dependencies defined by data outputs and custom 
		  attributes.
		- Step failures can occur due to resource constraints or service 
		  errors, but retry policies can enhance pipeline resilience by 
		  automatically retrying failed steps.
		- Amazon EventBridge can trigger pipeline executions based on specific events, automating and improving workflow efficiency.
		- SageMaker Studio allows users to monitor pipeline executions, view 
		  DAGs, and access detailed information about each step, including status 
		  and runtime metrics.
		- The final outputs of a pipeline can include files and models, with 
		  artifacts tracked at each stage to understand how outcomes were 
		  achieved.
	- Sagemaker operators for kubernetes
	  collapsed:: true
		- SageMaker operators for Kubernetes simplify the management of containerized
		  machine learning models by allowing direct management of SageMaker
		  resources within Kubernetes.
		- This integration helps DevOps teams streamline complex workflows and infrastructure management, enhancing control, portability, and
		  performance.
		- Kubeflow Pipelines enable the creation and deployment of scalable
		  machine learning workflows using Docker containers, with components that perform specific tasks in the pipeline.
		- SageMaker components for Kubeflow Pipelines allow users to manage
		  machine learning workflows through a user interface and SDK, shifting
		  compute workloads from Kubernetes clusters to SageMaker.
		- Two versions of SageMaker components are available, with version 2
		  offering enhanced resource management across various applications.
		- Specific components include Ground Truth for labeling jobs, Workteam for private work team jobs, Data Processing for processing jobs, and
		  training and hyperparameter optimization components for managing
		  training jobs.
		  
		  <!-- notionvc: 81a6d871-b634-484f-9e28-c8537f491a0a -->
	- Sagemaker Projects
	  collapsed:: true
		- SageMaker projects facilitate efficient code sharing, consistent code quality,
		  and strict version control in collaborative machine learning projects.
		- Projects are provisioned from the AWS Service Catalog using custom
		  or SageMaker provided templates, which include ready-made templates for
		  quick starts with machine learning workflows and CI/CD.
		- SageMaker templates support AWS tools like CodeBuild, CodeCommit,
		  CodePipeline, and third-party tools like GitHub and Jenkins for CI/CD
		  workflow automation.
		- A typical SageMaker project includes repositories with example code
		  for building and deploying machine learning solutions, allowing
		  organizations to customize resources to their needs.
		- The MLOps template for model building and training manages data
		  processing, feature extraction, model training/testing, and model
		  registry integration.
		- This template includes an AWS CodeCommit repository with sample
		  Python code, an AWS CodePipeline with source and build steps, and an
		  Amazon S3 bucket for storing artifacts.
		- If pre-made templates don't meet specific requirements, organizations can create customized templates.
		- The model deployment template streamlines deploying models from the
		  SageMaker model registry to endpoints for real-time inference,
		  automatically initiating deployment when a new model version is
		  registered.
		- It includes a CodeCommit repository with configuration files, AWS
		  CloudFormation templates, and a CodePipeline for deploying models to
		  staging and production environments.
		- The pipeline uses a CloudWatch event to trigger automatically upon
		  model package version approval or rejection and stores artifacts in an
		  Amazon S3 bucket.
		  
		  <!-- notionvc: 50797e92-79ac-4469-ada0-6bb5fa4adf76 -->
	-
	-
	- ## Amazon Q
	- Intro of Amazon Q
	  collapsed:: true
		- Amazon Q is a generative AI-powered assistant introduced at the 
		  re:Invent 2023 conference, designed to enhance interaction with 
		  organizational data, source code, and AWS services.
		- It can generate source code in various programming languages, 
		  connect to corporate data systems, synthesize large data sets, and 
		  tailor interactions based on user roles.
		- Similar to tools like ChatGPT, Amazon Q enables conversational, 
		  context-based interactions and can be used to build generative AI web 
		  applications.
		- It can integrate with enterprise systems like ServiceNow, 
		  Salesforce, Jira, and Slack to provide fast, accurate answers, create 
		  content, and interact with data.
		- Example uses include assisting graphic designers with branding 
		  guidelines, summarizing meeting notes for business analysts, and 
		  generating social media campaigns for marketing specialists.
		- Amazon Q is not limited to a single interface; it integrates across 
		  various platforms, including the AWS console, development IDEs, and 
		  other AWS services like Amazon QuickSight and Amazon Connect.
	- Amazon Q in Quicksight
	  collapsed:: true
		- Amazon Q in QuickSight enhances business intelligence by using
		  natural language to create and refine dashboards and reports, helping
		  users understand data insights.
		- It allows users to ask questions conversationally to uncover reasons behind data trends, such as sales decreases in specific regions.
		- Amazon Q can generate executive summaries and tailor them for different organizational stakeholders.
		- In customer service, Amazon Q integrates with Amazon Connect to
		  assist call center agents by processing conversations in real-time,
		  offering solutions, and reducing the need for supervisor intervention.
		- For supply chain management, Amazon Q in AWS Supply Chain enables
		  querying of data to assess current status and explore potential
		  scenarios, aiding in risk mitigation and resilience planning.
		  
		  <!-- notionvc: 72fc7bd5-d80e-4cf8-b3f6-b2500ab12f35 -->
	- Amazon Q for Data engineer
	  collapsed:: true
		- Amazon Q in AWS Glue provides a chat interface that helps users with data
		  integration tasks by offering guidance, generating code, and
		  troubleshooting issues.
		- Users can interact with Q in plain English to receive expert advice, sample code, and solutions for AWS Glue-specific problems.
		- Amazon Q also integrates with Amazon Redshift, allowing users to
		  generate SQL queries in the Redshift Query Editor using natural
		  language.
		- This integration aims to enhance productivity by reducing the need
		  for extensive coding expertise and simplifying data querying processes.
		  
		  <!-- notionvc: 28eff32e-0865-43b2-9b89-30e9c34d69ca -->
	- Amazon Q in codecatalyst
	  collapsed:: true
		- Amazon Q enhances productivity in CodeCatalyst through three main features:
			- **Pull Request Summaries**: Q automatically generates descriptions for pull requests by analyzing changes between source and destination code branches.
			- **Comment Summaries**: Q summarizes feedback comments on pull requests, identifying common suggestions, though these summaries are temporary.
			- **Issue Assignment**: Developers can assign issues to
			  Amazon Q, which attempts to create solutions. Users must specify
			  feedback levels, workflow file updates, and the repository for Q to work in.
		- Once an issue is assigned, Q drafts a solution, creates a branch, commits code, and generates a pull request for review.
		- Overall, Amazon Q in CodeCatalyst enhances productivity and workflow efficiency.
		  
		  <!-- notionvc: f32e78ef-2ea9-4f61-a7ab-4745c75eaad5 -->
	- Amazon Q in IDE
	  collapsed:: true
		- The /clear command refreshes the chat window, providing a clean slate for new interactions.
		- The /help command offers information about Amazon Q's capabilities, limitations, and useful commands.
		- Two specific commands for Amazon CodeWhisperer users are highlighted: /transform and /dev.
		- The /transform command assists in upgrading code bases to newer
		  versions, identifying necessary changes, and creating tests to ensure
		  successful upgrades.
		- The /dev command provides code suggestions for developing new
		  features by analyzing project files and creating implementation plans
		  based on user input.
		- These commands enhance developer efficiency by simplifying code upgrades and feature development.
		  
		  <!-- notionvc: afb145ee-0227-49df-977a-6105ceb07fd5 -->
	-
	-
	- ## Comprehend
	- Intro of comprehend
	  collapsed:: true
		- Amazon Comprehend is a machine learning service under AWS that uses
		  natural language processing (NLP) to extract insights from text
		  documents.
		- NLP is a subfield of linguistics, computer science, and AI focused
		  on interactions between computers and human language, enabling computers to understand document content and language nuances.
		- Amazon Comprehend can analyze documents at scale, extracting key phrases, entities, sentiment, and more using various APIs.
		- Key phrases are noun phrases identified with a confidence score, helping applications determine their relevance.
		- Sentiment analysis determines the emotional context of text,
		  categorizing it as positive, negative, neutral, or mixed, with
		  percentage scores for each.
		- Entities are references to people, places, events, dates, and commercial items, each classified with a confidence score.
		- The service can identify and redact personally identifiable information (PII) to manage security risks.
		- It supports multiple languages, determining the dominant language in a text with a confidence rating.
		- Syntax analysis classifies words by their syntactic function, building an understanding of word relationships.
		- Topic modeling identifies common themes in large text corpora, organizing documents into categories.
		- Amazon Comprehend is a fully managed, continuously trained NLP
		  service that analyzes text in various formats to provide meaningful
		  insights.
		  
		  <!-- notionvc: 1e02c31e-88fc-4c15-bc62-db200c60e02d -->
	- Three types of models to process data
	  collapsed:: true
		- Single document processing is asynchronous and suitable for analyzing one
		  document at a time, with operations like DetectDominantLanguage and
		  DetectSentiment.
		- Multi-document synchronous processing allows analysis of up to 25
		  documents simultaneously using Batch operations, providing individual
		  results for each document.
		- Asynchronous batch processing is ideal for large documents or
		  quantities, requiring data in Amazon S3 and UTF-8 format, with
		  additional operations for topic modeling.
		  
		  <!-- notionvc: e8b632a8-9c10-4695-8682-f8c4ee46c81c -->
	- Comprehend features and usecase
	  collapsed:: true
		- Amazon Comprehend offers a feature called Comprehend Custom, allowing users to create machine learning models tailored to their specific
		  organizational needs, enabling the detection of custom classifications
		  and entities in text.
		- The service provides API integration, making it easy to incorporate
		  sophisticated text analysis into existing applications without needing
		  specialized expertise in textual analysis.
		- Amazon Comprehend integrates with various AWS services, such as AWS
		  Lambda, Amazon S3, AWS Key Management Service, and Kinesis Data
		  Firehose, facilitating seamless data processing and analysis.
		- Security is emphasized through integration with IAM for access
		  control and KMS for data encryption, ensuring secure handling of
		  potentially sensitive information.
		- The service is highly scalable, capable of analyzing millions of
		  documents to provide valuable insights, which can be leveraged to
		  improve business outcomes, such as responding quickly to customer
		  sentiment.
		- Amazon Comprehend's deep learning models are continuously trained with global data, enhancing accuracy over time.
		- A related service, Amazon Comprehend Medical, applies NLP to extract
		  and identify medical and healthcare-related attributes from unstructured medical text, aiding in faster diagnosis and treatment decisions.
		  
		  <!-- notionvc: b9d9262e-3ecc-4d50-b4e0-ef18ce37f768 -->
	-
	-
	- ## Security
	- Securing AI systems
	  collapsed:: true
		- Key security practices include using encryption for data in transit and at rest, and managing permissions with least privilege.
		- AWS services like KMS and AWS Certificate Manager are recommended
		  for encryption, while IAM is crucial for managing access to applications and data.
		- The lecture explains identity-based and resource-level policies,
		  noting that not all AWS services support resource-level policies.
		- VPCs, VPC Endpoints, and AWS PrivateLink are discussed as essential networking elements for secure Cloud deployments.
		- Specific security considerations for Amazon's AI services, Bedrock
		  and SageMaker, are covered, including preventing Prompt Injection
		  attacks and using GuardDuty for monitoring vulnerabilities.
		- For SageMaker, data protection involves using CloudTrail, Amazon
		  Macie, and encryption, while data quality is ensured with SageMaker Data Wrangler and AWS Glue Data Quality.
		- Model cards in SageMaker are used for documenting machine-learning models to support governance and audits.
		  
		  <!-- notionvc: 6006fe02-110a-46ec-9403-cbb718bc6858 -->
	- Services to focus for security
	  collapsed:: true
		- **Audit Manager**: Assesses and manages risk in AWS
		  environments to assist with compliance to industry standards like ISO
		  27001, HIPAA, and PCI DSS. It centralizes audit management and
		  collaboration.
		- **CloudTrail**: Tracks all events in the Cloud environment, answering who, what, and when, aiding in governance and transparency.
		- **AWS Artifact**: Centralizes management of compliance documents and supports notifications for document updates.
		- **SageMaker Clarify**: Evaluates AI models for compliance with standards like ISO 42001, detecting bias and risks during data preparation.
		- **Amazon Inspector**: Scans EC2 instances, containers,
		  and Lambda functions for vulnerabilities, helping maintain compliance
		  with standards like PCI DSS and NIST.
		- **Amazon Macie**: Inspects S3 data for potential PII leaks, ensuring AI models maintain neutrality and protect proprietary information.
		- **S3**: Default choice for storing input and output data in AWS, with a focus on storage classes, bucket policies, and encryption.
		- **Trusted Advisor**: Aligns AWS environments with best practices, offering recommendations for security, cost optimization, and performance.
		- **Guardrails for Bedrock**: Provides AI-specific safeguards, blocking harmful content, filtering sensitive data, and ensuring relevant AI responses.
		  
		  <!-- notionvc: 8a35950c-ce44-40a1-bb2b-606345510eab -->
	-
	- ## Audit manager
	- Intro
		- AWS Audit Manager is a service designed to continuously audit AWS
		  usage and environments, helping assess risk and compliance with
		  regulatory and industry standards.
		- It automates evidence collection for auditors and security
		  professionals, facilitating the creation of reports for stakeholders or
		  governmental review.
		- The service maps AWS usage to controls, ensuring alignment with
		  governmental or industry requirements through pre-configured frameworks
		  like GDPR and CIS.
		- AWS Audit Manager supports collaboration among internal audit members, GRC, and IT/SecOps teams.
		- To start using AWS Audit Manager, users must create an assessment,
		  which involves collecting evidence related to a specific framework.
		- AWS provides a library of managed frameworks and allows for the creation of custom frameworks to meet specific needs.
		- The audit process involves roles such as Audit Owner, who manages
		  assessments, and Audit Delegates, who are subject matter experts
		  reviewing specific controls.
		- Audit Manager continuously monitors AWS services specified in the framework, collecting evidence to verify compliance.
		- Users can create assessments through the AWS console or CLI/API,
		  specifying details like assessment name, framework, and AWS accounts
		  involved.
		- The assessment report summarizes collected evidence and includes direct links to evidence files, but does not assess compliance.
		- Reports are structured with sections like a cover page, overview,
		  and evidence summary, and are stored in an S3 bucket for access.
		  
		  <!-- notionvc: 9b7b19be-0252-4f41-bfbe-ab24caf71f15 -->
- Day 18
  collapsed:: true
	- ## KMS
	- Components of KMS
		- AWS KMS Keys are primary keys used for cryptographic operations like
		  encryption and decryption. They can be symmetric (using the same key for encryption and decryption) or asymmetric (using a public and private
		  key pair).
		- Symmetric KMS keys remain within KMS for security, while asymmetric keys allow the public key to be used outside KMS.
		- There are three types of keys based on ownership: Customer managed, AWS managed, and AWS owned keys.
		- Customer managed keys offer more control over permissions and management, including key policies, rotation, and deletion.
		- AWS managed keys are automatically generated and managed by AWS, with no administrative duties for the user.
		- Hash-based Message Authentication Codes (HMAC) are used to verify
		  data integrity and authenticity, with HMAC keys created as symmetric
		  keys within KMS.
		- Data keys, generated by KMS keys, are used outside KMS for
		  encryption, while data key pairs are asymmetric and used for client-side cryptography.
		- Key material is crucial for cryptographic algorithms, with options
		  to source it from AWS KMS, external key managers, or CloudHSM.
		- Key rotation is recommended for security, with automatic rotation
		  available for customer managed keys, while AWS manages rotation for AWS
		  managed and owned keys.
		- Key policies and grants are used to control access to KMS keys, with grants providing temporary access without altering key policies.
		  
		  <!-- notionvc: 88d19f8a-2ff1-4e40-bbd7-337538258a02 -->
	-
	- ## Cloudtrail
	- Cloudtrails Intro
		- AWS CloudTrail is a service that records and tracks events, including API and non-API requests, within an AWS account.
		- It categorizes events into three types: Management Events, Data Events, and CloudTrail Insight Events.
		- Management Events track management operations on AWS resources, 
		  while Data Events focus on resource operations like S3 object-level 
		  activities.
		- Insight Events capture unusual activities, helping identify potential issues.
		- CloudTrail is enabled by default for new AWS accounts, allowing 
		  event viewing through the Event History in the AWS Management Console.
		- Users can create CloudTrail Trails to store, review, and analyze 
		  events beyond the Event History, with data stored in Amazon S3 or sent 
		  to Amazon CloudWatch Logs.
		- There are three types of Trails: All Region Trail, Single Region 
		  Trail, and AWS Organization Trail, each serving different scopes and 
		  purposes.
		- CloudTrail Lake allows storing and querying events for up to 7 years, using SQL queries to extract specific data for analysis.
		- CloudTrail Lakes can also capture log events from AWS Config and 
		  external sources, integrating with partners like CrowdStrike and GitHub 
		  through CloudTrail channels.
	- Benifits of Cloudtrail
		- AWS CloudTrail captures extensive data across regions and organizations, offering significant benefits for businesses.
		- It serves as a security tool by identifying unauthorized events, 
		  allowing security teams to investigate and prevent future occurrences.
		- CloudTrail consolidates activity records from multiple regions into a
		  single S3 bucket, facilitating data analysis and pattern 
		  identification.
		- It enhances visibility into AWS environments, helping detect unusual behavior and providing early warnings of potential attacks.
		- CloudTrail Insights tracks irregular API behavior, capturing additional metadata to understand the cause of anomalies.
		- Insights are stored separately and can be reviewed via the AWS management console.
		- CloudTrail maintains a detailed audit of API calls and configuration changes, aiding in governance and regulatory compliance.
		- Each recorded event includes comprehensive information such as the 
		  principal, account ID, username, event time, source, and more.
- Day 19
  collapsed:: true
	- AWS Sagemaker Jump start
		- Foundation Models are widely used due to their general knowledge base and ability to be customized for specific use cases.
		- AWS offers services like Amazon Bedrock and SageMaker JumpStart to facilitate access to these models.
		- SageMaker JumpStart is integrated into SageMaker Studio, providing a
		  hub of pre-trained Foundation Models for customization and evaluation.
		- It offers both proprietary and publicly available models for various
		  applications, including text and image generation, computer vision, and
		  natural language processing.
		- Users can access JumpStart through SageMaker Studio or the Amazon SageMaker console, where they can select and deploy models.
		- JumpStart provides example scripts and notebooks to help users set up, deploy, and evaluate models.
		- Users can train, deploy, and evaluate models, with options to fine-tune datasets and customize deployment settings.
		- JumpStart simplifies the use of Foundation Models while allowing control over infrastructure and deployment details.
- Day 20
	- General Model Evaluation Techniques
		- ROC and AUC are used to evaluate model performance across different thresholds, especially for imbalanced classes.
		- RMSE is specific to regression models, indicating how close predictions are to actual values.
		- The choice of metrics depends on the model type, algorithm, and 
		  business goals, with tools like SageMaker offering built-in metrics and 
		  visualizations to assist in evaluation.