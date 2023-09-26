# ENGR-E-516-ECC-Final-Project

### Large-scale Artificial Neural Network: MapReduce-based for Machine Learning

### Project goals:

- Understanding how the mapreduce framework works.
- Implementing mapreduce framework using hadoop.
- Using the mrjob library in python on a sample mapreduce use-case.
- Creating datasets for each model (in text form for mrjob).
- Applying mapreduce framework on several machine learning models:
  - Linear regression
  - Logistic regression
  - k-Nearest neighbours
  - Naive bayes
  - Decision trees
  - Random forest
  - Digit classifier neural network
- Implement all the above algorithms using the mapreduce framework (creating mappers and reducers)
- Converting the mapreduce mappers and reducers to mrjob.
- Verifying the results using the scikit-learns implementation of these models.
- Implementing neural network based deep learning using mapreduce framework.
- Comparing the results of mapreduce vs training on single instance.
- Running the algorithms on Jetstream instances parallely.

### Related work and gap analysis:

The paper "Large-scale distributed deep networks" describes the architecture and techniques used by Google to train very large deep neural networks on distributed computing systems. The authors present a system called "DistBelief" that is capable of training neural networks with billions of parameters on thousands of CPU and GPU processors. The system uses a data-parallel approach to distribute the computations and the data across the nodes in the network, and it includes several optimizations to improve efficiency and scalability, such as model and data parallelism, asynchronous SGD, and parameter server architectures.
The paper "Large-scale Artificial Neural Network: MapReduce-based Deep Learning" proposes a system for training large-scale artificial neural networks using the MapReduce paradigm. Although the paper offers a comprehensive analysis of the suggested system, there are certain possible research gaps that might be addressed in subsequent work, such as:
- The proposed system is evaluated on a limited set of datasets. It would be useful to evaluate the system on a wider range of datasets to determine its performance characteristics in different scenarios.
- The paper focuses on the training phase of large-scale neural networks, but does not address issues related to deployment and inference. Future work could explore ways to efficiently deploy and infer large-scale neural networks in a distributed setting.

Overall, the paper "Large-scale Artificial Neural Network: MapReduce-based Deep Learning" makes a valuable contribution to the field of large-scale deep learning, but there is room for further research to address some of the potential gaps in the work.

### Proposed tasks

a) **Understanding and Implementing Map-reduce**: Implement the mapreduce framework on a sample use-case to understand how the framework works using the hadoop and mrjob library.

(b) **Generating dataset for machine learning models**: Generate datasets for classification, regression and digits for the classification, regression and the digit classifier model.

(c) **Implement the classification models in mapreduce architecture**: Convert each of the regression models (Linear regressiom, ridge regression, decision trees) to the mapreduce architecture, implement these algorithms from scratch using mrjob library in python.

(d) **Implement the classification models in mapreduce architecture**: Convert each of the regression models (Logistic regression, kNN, naive bayes) to the mapreduce architecture, implement these algorithms from scratch using mrjob library in python.

(e) **Compare the models**: Compare the results from the mapreduce architecture with the results from scikit-learn implementation of the models.

(f) **Train digit classifier using the mapreduce framework**: Implement a simple digit classifier and train using the mapreduce framework. Evaluate the model using the weights obtained from the mapreduce.

(g) **Run the algorithms on jetstream instance**: Push the code on the jetstream instance and run the mrjob to obtain results.

(h) **Compare the performance**: Compare the performance and other metrics (CPU usage, time consumption) of the mapreduce implementation and training the algorithms using scikit-learn on a single instance.



### Preliminary Overview:

We have observed that the map-reduce framework is particularly useful for applying machine learning algorithms to large datasets. This is because map-reduce allows us to distribute the processing of data across multiple nodes in a cluster, enabling us to process large amounts of data in parallel. In addition, the map-reduce framework is fault-tolerant, which means that if any of the nodes in the cluster fail during processing, the system will automatically reassign their tasks to other nodes, ensuring that the processing continues without interruption.

Through our analysis, we have found that using map-reduce to implement machine learning algorithms can significantly reduce the time required for processing large datasets compared to traditional single-machine approaches. This is because map-reduce allows us to parallelize the processing of data, which can result in significant speedups when working with large datasets. In addition, we have found that using map-reduce can be particularly effective for algorithms that involve iterative processing, such as training neural networks, as it allows us to distribute the work of each iteration across multiple nodes in the cluster, reducing the overall time required for training.

Overall, our preliminary analysis suggests that using the map-reduce framework for machine learning can offer significant advantages over traditional single-machine approaches, particularly when working with large datasets.


### References:

[1] Sejnowski TJ. Bell AJ. An information-maximization approach to blind separation and blind deconvolution. In Neural Computation, 1995.

[2] O. Chapelle. Training a support vector machine in the primal. Journal of Machine Learning Research (submitted), 2006.

[3] W. S. Cleveland and S. J. Devlin. Locally weighted regression: An approach to regression analysis by local fitting. In J. Amer. Statist. Assoc. 83, pages 596–610, 1988.

[4] L. Csanky. Fast parallel matrix inversion algorithms. SIAM J. Comput., 5(4):618–623, 1976.

[5] A. Silvescu D. Caragea and V. Honavar. A framework for learning from distributed data using sufficient statistics and its application to learning decision trees.
International Journal of Hybrid Intelligent Systems, 2003.

[6] J.Ferreira,M.B.Vellasco,M.A.C.Pacheco,R.Carlos,andH.Barbosa, “Data mining techniques on the evaluation of wireless churn.” in ESANN, 2004, pp. 483–488.

[7] H. Hwang, T. Jung, and E. Suh, “An ltv model and customer seg- mentation based on customer value: a case study on the wireless telecommunication industry,” Expert systems with applications, vol. 26, no. 2, pp. 181–188, 2004.

[8] L. G. Valiant, “A theory of the learnable,” Communications of the ACM, vol. 27, no. 11, pp. 1134–1142, 1984






