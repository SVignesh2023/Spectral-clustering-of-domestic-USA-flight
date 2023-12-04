# USA-flight-spectral-clustering
My MSc dissertation. Perform spectral clustering algorithms on domestic USA flight dataset.

<b>Abstract</b>
<br>
In recent years, spectral clustering has garnered significant attention from academia due to its strong theoretical foundations in algebraic graph theory and excellent clustering performance. Spectral clustering treats the clustering problem as a graph partitioning problem and has a key advantage of being able to cluster data without making any assumptions about the shapes of the clusters. This project focuses on applying spectral clustering algorithms to a USA domestic flight dataset spanning the years from 1990 to 2009. The spectral clustering algorithms used here are Shi and Malik (2000) and Ng, Jordan, and Weiss (2002). Additionally, five different clustering algorithms, namely K-Means++, Agglomerative Hierarchical Clustering, Fuzzy C-Means++, DBSCAN and Mean Shift are implemented to apply on the eigenvector matrix produced by spectral clustering. Finally, a web-based application is developed, enabling users to customize inputs to these algorithms and visualize the resulting clusters. This is a full stack project. The backend framework was built using the Python programming language. NetworkX library is used to build the graph representation of the dataset, and to compute its associated matrices. The python script is converted into a web application using Flask library. The frontend framework is built using HTML, CSS, and JavaScript languages. This project produced promising results by applying spectral clustering to a flight dataset, suggesting its potential for broader application in realworld datasets. It is also observed that a tradeoff between quality of clusters and distribution of quantity of objects across clusters is expected when selecting one type of spectral clustering algorithm over the other. 

<b>Keywords:</b> Clustering, Spectral clustering, NetworkX, Flask, Full stack

<b>User guide for project setup</b>

1. Download the dataset from: https://www.kaggle.com/datasets/flashgordon/usa-airport-dataset. (Since the dataset is too large, it cannot be attached here, hence it needs to be downloaded separately) Keep it in the root folder.

2. Python version 3.10 was used. This tutorial was followed to setup a virtual environment, to install flask library and to lay down a boilerplate code for Flask in Visual Studio Code: https://code.visualstudio.com/docs/python/tutorial-flask. An advantage of implementing a virtual environment is that it avoids installing Flask into a global Python environment and gives the owner exact control over the libraries used in an application.

3. pip install all the python libraries in the ‘requirements.txt’ file, present in the project source code. These libraries must be installed when the virtual environment is active.

4. After writing down the code for the python file, HTML file, JavaScript file and the CSS file, the python file must be run using the following command in the command line terminal: python -m flask run.

5. The webpage would be available on localhost to view. The address to the webpage can be found in the message displayed in the command line terminal.

Thank You!
