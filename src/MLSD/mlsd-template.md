# ML System Design Formula Template (non-LLM approach)

Note: 
* It is updated from the original [mlsd-template.md](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-template.md)
* It assumes a non-LLM based approach (for LLM based approach, we may have a difference template. In particular, we may not need features engineering model training, such as )

1. Problem Formulation
    * Clarifying questions
    * Use cases and business goals
    * Requirements
    * Constraints
    * Data: sources and availability
    * Assumptions
    * Can we go with non-ML approach instead
    * If not, what is the ML formulation

2. Metrics  
    * Offline metrics
    * Online metrics

3. Architectural Components  
    * High level architecture 

4. Data Collection and Preparation
    * Data needs
    * Data Sources
    * Data storage
    * ML Data types
    * Labelling

5. Feature Engineering
    * Feature selection 
    * Feature representation 
    * Feature preprocessing 

6. Model Development and Offline Evaluation
    * Model selection 
    * Dataset construction 
    * Training Hardware and Capacity
    * Model Training
    * Model eval and HP tuning 
    * Training and eval performance metric tuning
    * Model Iterations

7. Prediction Service
    * Serving Hardware and Capacity
    * Inference Runtime    
    * Model deployment to serving hardware
    * Model size reduction and prediction service performance metric tuning 
    * Service monitoring, alerting, oncall (i.e. MLOps)

8. Recurrent or Online Training (Optional)

9. Online Testing and Model Release
    * A/B Test and pick candidate model for production
    * Model release 

10. Monitoring and Updates 
    * Monitoring model prediction quality and performance metrics
    * Updates (e.g. add new features, model techniques, grows model size, serving with new inference hardwares, etc.)
