# Privacy-preserving-SVM_SPU
This project is constructed with SPU from https://github.com/secretflow, and it implements a Support Vector Machine (SVM) classifier and integrates it with Secure Multi-Party Computation (SMC) to protect privacy. Specifically, the code uses JAX and SecretFlow libraries for SMC computations. In a loop, it iterates over different test set sizes and records the time taken for each run of the code, and finally uses the matplotlib library to plot the test set size against running time.
The MulticlassSVM here is achieved by applying 'ovr',i.e. 'one VS rest' technique.
