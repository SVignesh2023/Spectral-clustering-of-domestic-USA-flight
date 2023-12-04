from flask import Flask,render_template,request,jsonify

import pandas as pd
import numpy as np
import random
import math

import networkx as nx

import skfuzzy as fuzz

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift

import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)
storedData={}
createDict=0

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get_time", methods=['POST'])
def get_time():
    """
    Function which provides the user with the estimated time taken for the clustering operation to produce the output based on the form values they have selected.

    Parameters:
            none
    
    Returns:
            JSON object
    """
    global storedData
    global createDict
    dfLookup=pd.read_csv('Results.csv')
    starting_date=str(request.form['start_date'])
    ending_date=str(request.form['end_date'])
    top_percent=int(request.form['top'])
    spec_choice=str(request.form['spectral_choice'])
    alg_method=str(request.form['algorithm_choice'])

    countExist=0
    keyList=['Starting date','Ending date','Top percent']
    for x in keyList:
        if x in storedData.keys():
            countExist+=1

    if countExist==3:
        if (storedData['Starting date']==starting_date) and (storedData['Ending date']==ending_date) and (storedData['Top percent']==top_percent):
            createDict=1
        else:
            createDict=0
    else:
        createDict=0
    
    if createDict==1:
        timeTaken=7
    else:    
        timeTaken=math.ceil(dfLookup.loc[(dfLookup['Top']==top_percent)&(dfLookup['Spectral']==spec_choice)&(dfLookup['Clustering']==alg_method)]['Time'].values[0])
    
    return jsonify({"time_json": timeTaken})

@app.route('/call_python_script', methods=['POST'])
def call_python_script():
    """
    Function which calls the main python script. Also serves as the 'link' connecting the webpage to the python script. It passes on the webpage form inputs to the python script inputs.
    Output of python script. Two outputs, the USA map containing the clusters and the cluster evaluation results, both in JSON format.

    Parameters:
            none

    Returns:
            JSON object
    """
    global storedData
    global createDict
    starting_date=str(request.form['start_date'])
    ending_date=str(request.form['end_date'])
    top_percent=int(request.form['top'])
    spec_choice=str(request.form['spectral_choice'])
    alg_method=str(request.form['algorithm_choice'])
    k_val=str(request.form['num_clusters'])
    view_mode=str(request.form['viewing'])

    #Get the number of clusters
    k_value=int(k_val[:-1])    

    if createDict==1:
        image_results,eval_results=performClustering(spec_choice,alg_method,k_value,view_mode,storedData['Dataframe'],storedData['Graph'])
    else:
        image_results,eval_results,finished_graph,uniqueDf=main(starting_date,ending_date,top_percent,spec_choice,alg_method,k_value,view_mode)
        storedData={'Starting date':starting_date,'Ending date':ending_date,'Top percent':top_percent,'Graph':finished_graph,'Dataframe':uniqueDf}    

    return jsonify(image_results=pio.to_json(image_results),evaluation_output=eval_results)

if __name__=="__main__":
    app.run(debug=True)

def evaluationMatrix(data_matrix,labelList):
    """
    Produces the cluster evaluation scores. Returns a dictionary containing the evaluation method as the kye and its score as the value.

    Parameters:
            data_matrix: numpy array - A matrix containing all the objects.
            labelList: list - A list containing which cluster number each object is assigned to.    

    Returns:
            dictionary
    """
    silScore=round(sklearn.metrics.silhouette_score(X=data_matrix,labels=labelList),2)
    chScore=round(sklearn.metrics.calinski_harabasz_score(X=data_matrix,labels=labelList),2)
    dbScore=round(sklearn.metrics.davies_bouldin_score(X=data_matrix,labels=labelList),2)
    evalDict={'Silhouette_Score':silScore,'Calinski_Harabasz_Score':chScore,'Davies_Bouldin_Score':dbScore,'Number_of_clusters':int(max(labelList)+1)}
    return evalDict  

def k_meansPlusPlus(numClusters,data_matrix):
    """
    Performs k-means++ clustering algorithm on the dataset. Returns a list containing which cluster number each object is assigned to.    

    Parameters:
            numClusters: integer - The number of clusters a clustering algorithm should produce.
            data_matrix: numpy array - A matrix containing all the objects.
            
    Returns:
            list
    """
    kmeans=KMeans(n_clusters=numClusters,random_state=0,init='k-means++',n_init='auto').fit(data_matrix)
    return kmeans.labels_

def AHC(numClusters,data_matrix):
    """
    Performs Agglomerative Hierarchical Clustering(AHC) algorithm on the dataset. Returns a list containing which cluster number each object is assigned to. 

    Parameters:
            numClusters: integer - The number of clusters a clustering algorithm should produce.
            data_matrix: numpy array - A matrix containing all the objects.

    Returns:
            list
    """
    ahc=AgglomerativeClustering(n_clusters=numClusters,linkage='average').fit(data_matrix)
    return ahc.labels_

def fuzzyAlg(numClusters,data_matrix):
    """
    Performs Fuzzy C-means++ algorithm on the dataset. Initially performs the '++' initialization to select better cluster centers. Then it continues on with standard Fuzzy C-means algorithm. 
    Returns a list containing which cluster number each object is assigned to. 

    Parameters:
            numClusters: integer - The number of clusters a clustering algorithm should produce.
            data_matrix: numpy array - A matrix containing all the objects.

    Returns:
            list
    """
    # '++' initialization to select the initial number of centers
    random.seed(0)
    initialCentersIndex=[]
    allIndex=[i for i in range(0,len(data_matrix))]
    firstIndex=random.randint(0,(len(data_matrix)-1))
    initialCentersIndex.append(firstIndex)
    distanceDict={}
    
    for num_outer in range(1,numClusters):
        probList=[]
        totalDist=0

        for i in range(len(data_matrix)):
            tempDistList=[]
            for num_inner in range(0,len(initialCentersIndex)):
                tempDistList.append(np.linalg.norm(data_matrix[initialCentersIndex[num_inner]]-data_matrix[i])**2)
            minDist=min(tempDistList)
            totalDist+=minDist
            distanceDict.update({i:minDist})
        
        probList=[(x/totalDist) for x in distanceDict.values()]
        newCenter=random.choices(allIndex,weights=probList,k=1)
        initialCentersIndex.append(newCenter[0])
    
    initialCentersObjects=np.take(a=data_matrix,indices=initialCentersIndex,axis=0)
    
    #Calculate the initial guess matrix based on the best centers
    membershipVal=2
    raw_membershipMatrix=[]
    distanceMatrix=[]
    for i in range(len(data_matrix)):
        rowDistance=[]
        for j in range(len(initialCentersObjects)):
            rowDistance.append(np.linalg.norm(initialCentersObjects[j]-data_matrix[i])**2)
        distanceMatrix.append(rowDistance)
    
    for i in range(len(distanceMatrix)):
        rowMembership=[]
        if i in initialCentersIndex:
            for z in range(len(initialCentersObjects)):
                if (initialCentersObjects[z]==data_matrix[i]).any():
                    rowMembership.append(1)
                else:
                    rowMembership.append(0)
            raw_membershipMatrix.append(rowMembership)
            continue
        for j in range(len(distanceMatrix[i])):
            eleSum=0
            for k in range(len(distanceMatrix[i])):
                eleSum+=(distanceMatrix[i][j]/distanceMatrix[i][k])
            rowMembership.append((eleSum**(-1/(membershipVal-1))))
        raw_membershipMatrix.append(rowMembership)

    membershipMatrix=[[raw_membershipMatrix[j][i] for j in range(len(raw_membershipMatrix))] for i in range(len(raw_membershipMatrix[0]))]

    #Implementation of fuzzy C-Means++ clustering algorithm. Seed value fixed to 0 for initial fuzzy c-partitioned matrix
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data=data_matrix.T,c=numClusters,m=membershipVal,error=0.05,maxiter=1000,init=membershipMatrix)
    cluster_membership=np.argmax(u,axis=0)
    return cluster_membership

def dbscanAlg(data_matrix):
    """
    Performs Density-based spatial clustering of applications with noise (DBSCAN) clustering algorithm on the dataset. Returns a list containing which cluster number each object is assigned to.

    Parameters:
            data_matrix: numpy array - A matrix containing all the objects.

    Returns:
            list
    """
    clustering=DBSCAN(eps=0.4, min_samples=10).fit(data_matrix)
    
    labelList=clustering.labels_
    uniqueList=set(labelList)
    noiseLabel=max(uniqueList)+1
    
    #If number of clusters is already greater than 10, all the cluster objects in (10+i)th cluster and noise labels are dumped into the 10th cluster
    if 9 in uniqueList:
        for i in range(len(labelList)):
            if labelList[i]==-1 or labelList[i]>9:
                labelList[i]=9
    
    #Noisy samples are given the label -1. Hence changing the cluster label from -1 to (max(labels)+1) as '-1' key doesn't exist
    else:
        for i in range(len(labelList)):
            if labelList[i]==-1:
                labelList[i]=noiseLabel

    return labelList

def meanShift(data_matrix):
    """
    Performs Mean Shift clustering algorithm on the dataset. Returns a list containing which cluster number each object is assigned to.

    Parameters:
            data_matrix: numpy array - A matrix containing all the objects.

    Returns:
            list
    """
    bndWidth=sklearn.cluster.estimate_bandwidth(X=data_matrix,random_state=0)
    mean_shift=MeanShift(bandwidth=bndWidth).fit(data_matrix)
    
    labelList=mean_shift.labels_
    uniqueList=set(labelList)
    
    #Mean shift algorithm may tend to produce more than 10 clusters.
    #If number of clusters is already greater than 10, all the cluster objects in (10+i)th cluster are dumped into the 10th cluster
    if 9 in uniqueList:
        for i in range(len(labelList)):
            if labelList[i]>9:
                labelList[i]=9
    
    return labelList

def userMenu(numClusters,data_matrix,algChoice):
    """
    A menu for selecting which clustering algorithm to apply based on the user's choice. Returns a list (passed on previously by the respective clustering algorithm selected) containing which cluster number each object is assigned to.

    Parameters:
            numClusters: integer - The number of clusters a clustering algorithm should produce.
            data_matrix: numpy array - A matrix containing all the objects.
            algChoice: string - A choice made by the user about which clustering algorithm to apply.

    Returns:
            list
    """
    if algChoice=="1b":
        return k_meansPlusPlus(numClusters,data_matrix)
    elif algChoice=="2b":
        return AHC(numClusters,data_matrix)
    elif algChoice=="3b":
        return fuzzyAlg(numClusters,data_matrix)
    elif algChoice=="4b":
        return dbscanAlg(data_matrix)
    elif algChoice=="5b":
        return meanShift(data_matrix)
    else:
        pass

def vanilla_spec(A,D,k,algChoice,lookAhead):
    """
    Performs the Vanilla (Shi and Malik (2000)) spectral clustering algorithm on the dataset. 
    Spectral clustering is always applied on the dataset initially to store the object indices in a list which may lead to infinite/NaN/complex number values during spectral clustering.
    These objects(nodes) are then removed from the dataset. Spectral clustering is applied again to the dataset, this time leading to no problems.

    Parameters:
            A: numpy array - Adjacency matrix representation of the dataset.
            D: numpy array - Degree matrix representation of the dataset.
            k: integer - The number of clusters a clustering algorithm should produce.
            algChoice: string - A choice made by the user about which clustering algorithm to apply.
            lookAhead: boolean - Whether to apply the spectral clustering algorithm 'as it is' (for lookAhead=False) or to store the object indices in a list which may lead to infinite/NaN/complex number values during spectral clustering (for lookAhead=True).

    Returns:
            list, numpy array (for both boolean values of lookAhead)
    """
    I=np.identity(len(A))
    L_rw=I-np.matmul(np.linalg.inv(D),A)
    eigenvalues,eigenvectors=np.linalg.eig(L_rw)
    eigenvalues=eigenvalues[np.argsort(eigenvalues)]
    eigenvectors=eigenvectors[:,np.argsort(eigenvalues)]
    eigenvalues=eigenvalues[1:k]
    eigenvectors=eigenvectors[:,1:k]

    is_complex=eigenvectors.dtype==np.complex128
    
    #Create a list of indices for which rows contain NaN or non-zero complex data types
    if lookAhead:
        indices=np.argwhere(np.isnan(eigenvectors))[:,0]
        indices=list(set(indices))
        if is_complex:
            imaginary_parts=np.imag(eigenvectors)
            nonzeroRows=np.any(imaginary_parts!=0,axis=1)
            nonzeroIndices=list(np.where(nonzeroRows)[0])
            indices.extend(nonzeroIndices)
        return indices,eigenvectors
    else:
        if is_complex:
            eigenvectors=np.real(eigenvectors)
        return userMenu(numClusters=k,data_matrix=eigenvectors,algChoice=algChoice),eigenvectors

def variant_spec(A,D,k,algChoice,lookAhead):
    """
    Performs the Variant (Ng, Jordan, and Weiss (2002)) spectral clustering algorithm on the dataset. 
    Spectral clustering is always applied on the dataset initially to store the object indices in a list which may lead to infinite/NaN/complex number values during spectral clustering.
    These objects(nodes) are then removed from the dataset. Spectral clustering is applied again to the dataset, this time leading to no problems.

    Parameters:
            A: numpy array - Adjacency matrix representation of the dataset.
            D: numpy array - Degree matrix representation of the dataset.
            k: integer - The number of clusters a clustering algorithm should produce.
            algChoice: string - A choice made by the user about which clustering algorithm to apply.
            lookAhead: boolean - Whether to apply the spectral clustering algorithm 'as it is' (for lookAhead=False) or to store the object indices in a list which may lead to infinite/NaN/complex number values during spectral clustering (for lookAhead=True).

    Returns:
            list, numpy array (for both boolean values of lookAhead)
    
    """
    L=D-A
    D_ele=np.linalg.inv(np.sqrt(D))
    L_sym=np.matmul(np.matmul(D_ele,L),D_ele)
    eigenvalues,eigenvectors=np.linalg.eig(L_sym)
    eigenvalues=eigenvalues[np.argsort(eigenvalues)]
    eigenvectors=eigenvectors[:,np.argsort(eigenvalues)]
    eigenvalues=eigenvalues[1:k]
    eigenvectors=eigenvectors[:,1:k]
    
    #implement normalization
    for i in range(len(eigenvectors)):
        sum=0
        for j in range(len(eigenvectors[i])):
            sum+=eigenvectors[i][j]**2
        eigenvectors[i]=np.divide(eigenvectors[i],np.sqrt(sum))

    is_complex=eigenvectors.dtype==np.complex128

    #Create a list of indices for which rows contain NaN or non-zero complex data types 
    if lookAhead:
        indices=np.argwhere(np.isnan(eigenvectors))[:,0]
        indices=list(set(indices))
        if is_complex:
            imaginary_parts=np.imag(eigenvectors)
            nonzeroRows=np.any(imaginary_parts!=0,axis=1)
            nonzeroIndices=list(np.where(nonzeroRows)[0])
            indices.extend(nonzeroIndices)
        return indices,eigenvectors
    else:
        if is_complex:
            eigenvectors=np.real(eigenvectors)
        return userMenu(numClusters=k,data_matrix=eigenvectors,algChoice=algChoice),eigenvectors
   
def main(user_startDate,user_endDate,top_percent,specClusterVar,clusterAlgChoice,kValue,viewOption):
    """
    The main() part of the python script. It returns the cluster map visualization, evaluation results, graph representation of the dataset and finally, a dataframe consisting of three columns(airport name, latitude coordinate, longitude coordinate) for all airports.

    Parameters:
            user_startDate: string - The starting date of the dataset.
            user_endDate: string - The ending date of the dataset.
            top_percent: integer - Top percentage of the airports(nodes) in the within the starting and ending date of the dataset which receive the most flight traffic.
            specClusterVar: string - User choice of which spectral clustering algorithm to apply.
            clusterAlgChoice: string - A choice made by the user about which clustering algorithm to apply.
            kValue: integer - The number of clusters a clustering algorithm should produce.
            viewOption: string - Whether to produce a clustering output visualization which is suitable for colorblindness or not.

    Returns:
            plotly figure, dictionary, networkx graph and pandas dataframe

    """
    #Load in the dataset
    df_raw=pd.read_csv('Airports2.csv')

    #Clean the dataset
    #NaN values removed
    df=df_raw.dropna()
    df.reset_index(inplace=True,drop=True)

    #Flights value=0 removed
    df=df[df['Flights']!=0]
    df.reset_index(inplace=True,drop=True)

    #Flights for which origin and destination airports are the same is removed
    df=df[df['Origin_airport'] != df['Destination_airport']]
    df.reset_index(inplace=True,drop=True)

    #Flights which have 0 passengers and 0 seats are removed
    df = df[(df['Passengers'] != 0) | (df['Seats'] != 0)]
    df.reset_index(inplace=True,drop=True)

    #Flights which have 0 distance removed
    df=df[df['Distance']!=0]
    df.reset_index(inplace=True,drop=True)

    # Converting data into graph representation
    # User Input (Dataset)
    #Give date ranges in 'MM-YYYY' format
    sd=user_startDate.split('-')[1]+'-'+user_startDate.split('-')[0]+'-'+'01'
    ed=user_endDate.split('-')[1]+'-'+user_endDate.split('-')[0]+'-'+'01'
    startDate=pd.to_datetime(sd)
    endDate=pd.to_datetime(ed)

    #Filters out the rows which lie between the given timeframe
    df_preGraph=df.loc[:,['Origin_airport','Destination_airport','Passengers','Seats','Flights','Fly_date','Origin_population','Destination_population']]
    df_preGraph.reset_index(inplace=True,drop=True)
    df_preGraph['Fly_date']=pd.to_datetime(df_preGraph['Fly_date'])
    df_preGraph=df_preGraph[(df_preGraph['Fly_date']>=startDate) & (df_preGraph['Fly_date']<=endDate)]
    df_preGraph.reset_index(inplace=True,drop=True)

    #remove all rows where the number of seats are 0
    df_preGraph=df_preGraph[df_preGraph['Seats']!=0]
    df_preGraph.reset_index(inplace=True,drop=True)

    #Create the 'edge' column for the future graph
    df_preGraph['Edges'] = df_preGraph['Passengers'] / np.sqrt(df_preGraph['Origin_population'] * df_preGraph['Destination_population'])
    boolMask=df_preGraph['Passengers']==0
    df_preGraph.loc[boolMask,'Edges']=(df_preGraph.loc[boolMask,'Flights']/np.sqrt(df_preGraph.loc[boolMask,'Origin_population']*df_preGraph.loc[boolMask,'Destination_population']))
    df_preGraph=df_preGraph.drop(['Passengers','Seats','Fly_date','Origin_population','Destination_population'],axis=1)

    #A flight from airport A->B takes place every year. So the edge value has to be aggregated. Flights value is aggregated too for selection of top% of airports
    df_graph=df_preGraph.groupby(['Origin_airport','Destination_airport',],as_index=False).agg({'Edges':'sum','Flights':'sum'})
    df_graph.reset_index(inplace=True,drop=True)

    #Selecting top% of busiest airports
    busy_rows=int(len(df_graph)*(top_percent/100))
    df_graph=df_graph.nlargest(busy_rows,'Flights')
    df_graph=df_graph.drop(['Flights'],axis=1)
    df_graph.reset_index(inplace=True,drop=True)

    #Sort origin airports in alphabetical order
    df_graph = df_graph.sort_values('Origin_airport')
    df_graph.reset_index(inplace=True,drop=True)

    # Create and populate the graph
    G=nx.Graph()

    raw_edges=[tuple(row) for row in df_graph.values]
    edges=[]
    rejectPairs=[]

    for element in raw_edges:
        chk_wt=df_graph.loc[(df_graph['Origin_airport']==element[1])&(df_graph['Destination_airport']==element[0]),'Edges'].values
        if len(chk_wt)==0:
            G.add_edge(element[0],element[1],weight=element[2])
            edges.append(element)
        elif set([element[0],element[1]]) in rejectPairs:
            pass
        else:
            netwt=element[2]+chk_wt[0]
            G.add_edge(element[0],element[1],weight=netwt)
            edges.append(tuple([element[0],element[1],netwt]))
            rejectPairs.append(set([element[0],element[1]]))

    G.add_weighted_edges_from(edges)
    
    #Create a dataframe which contains columns: 'Airport','Latitude' and 'Longitude'.
    df_visual=df_raw.dropna()
    df_visual.reset_index(inplace=True,drop=True)
    df_visual=df_raw.loc[:,['Destination_airport','Dest_airport_lat','Dest_airport_long']]
    df_visual=df_visual.drop_duplicates(subset='Destination_airport')
    df_visual.columns=['Airport','Lat','Long']
    df_visual.reset_index(inplace=True,drop=True)

    """
    Both the variants of spectral clustering are applied to see which objects(nodes) produces infinite/NaN/complex number values. These are actually subgraphs, and cause problems during eigen vector matrix calculation.
    The problematic objects are stored in a list. These nodes are then removed from the graph.
    This process is repeated until no problematic objects remain, and a pure graph is formed.
    """
    while(True):
        badObjects=[]
        nodes_list=list(G.nodes)
        adj_temp_matrix=nx.to_numpy_array(G=G,nodelist=nodes_list)
        degree_temp_matrix=np.diag(adj_temp_matrix.sum(axis=1))
        badNodesIndex2,tempEigen2=variant_spec(adj_temp_matrix,degree_temp_matrix,kValue,clusterAlgChoice,True)
        badNodesIndex1,tempEigen1=vanilla_spec(adj_temp_matrix,degree_temp_matrix,kValue,clusterAlgChoice,True)
        mergedList=list(set(badNodesIndex1+badNodesIndex2))
        if len(mergedList)==0:
            break
        else:
            for i in range(len(mergedList)):
                badObjects.append(nodes_list[mergedList[i]])
            G.remove_nodes_from(badObjects)

    producedFigure,evalResults = performClustering(specClusterVar,clusterAlgChoice,kValue,viewOption,df_visual,G)
    return producedFigure,evalResults,G,df_visual

def performClustering(specClusterVar,clusterAlgChoice,kValue,viewOption,df_visual,G):
    """
    A function which calls the respective functions to perform clustering operation on the dataset. After clustering is achieved, it calls another function to visualize the clusters formed.
    It returns the cluster map visualization and the evaluation results.

    Parameters:
            specClusterVar: string - User choice of which spectral clustering algorithm to apply.
            clusterAlgChoice: string - A choice made by the user about which clustering algorithm to apply.
            kValue: integer - The number of clusters a clustering algorithm should produce.
            viewOption: string - Whether to produce a clustering output visualization which is suitable for colorblindness or not.
            df_visual: pandas object - A pandas dataframe consisting of three columns(airport name, latitude coordinate, longitude coordinate) for all airports.
            G: networkx graph object - A graph representation of the dataset.

    Returns:
            plotly figure, dictionary
    """
    if clusterAlgChoice=="4b" or clusterAlgChoice=="5b":
        kValue=5
    
    nodes_list=list(G.nodes)
    adj_matrix=nx.to_numpy_array(G=G,nodelist=nodes_list)
    degree_matrix=np.diag(adj_matrix.sum(axis=1))

    #Select spectral clustering algorithm variant
    if specClusterVar=="1a":
        labelList,eigenMatrix=vanilla_spec(adj_matrix,degree_matrix,kValue,clusterAlgChoice,False)
    else:
        labelList,eigenMatrix=variant_spec(adj_matrix,degree_matrix,kValue,clusterAlgChoice,False)

    labelsDict={nodes_list:labels for nodes_list,labels in zip(nodes_list,labelList)}

    # Visualize the results on USA map

    #Create a dictionary for cluster number as keys and color as values
    colorList=[]
    colorDict={0:'blue',1:'cyan',2:'red',3:'purple',4:'orange',5:'green',6:'black',7:'magenta',8:'yellow',9:'brown'}
    for x in labelsDict.values():
        colorList.append(colorDict[x])

    #Colour-blind option, creates a dictionary of shapes
    shapeList=[]
    shapeDict={0:'circle',1:'star',2:'square',3:'cross',4:'hourglass',5:'triangle-up',6:'pentagon',7:'x',8:'diamond',9:'hexagon'}
    for x in labelsDict.values():
        shapeList.append(shapeDict[x])

    long_list=[]
    lat_list=[]
    for i in range(len(nodes_list)):
        long_list.append(df_visual.loc[df_visual['Airport']==nodes_list[i],'Long'].values[0])
        lat_list.append(df_visual.loc[df_visual['Airport']==nodes_list[i],'Lat'].values[0])

    # Evaluation metrics
    if len(set(labelList))==1:
        evalMetrics={"No score":"Evaluation scores cannot be produced only one cluster is formed!"}
    else:
        evalMetrics=evaluationMatrix(eigenMatrix,labelList)

    return visual_customize(lat_list,long_list,nodes_list,colorList,shapeList,viewOption), evalMetrics

def visual_customize(lat_list,long_list,nodes_list,colorList,shapeList,viewOption):
    """
    Function to project the clusters on to the map of USA. This is achieved via plotly. It returns the clsuter map visualization.

    Parameters:
            lat_list: list - A list containing the latitude coordinates of all airports in the dataset.
            long_list: list - A list containing the longitude coordinates of all airports in the dataset.
            nodes_list: list - A list containing the name of all the airports in the dataset.
            colorList: list - A list containing the color type(for graph visualization) for each airport present in the dataset.
            shapeList: list - A list containing the marker type(for graph visualization) for each airport present in the dataset.
            viewOption: string - Whether to produce a clustering output visualization which is suitable for colorblindness or not.

    Returns:
            plotly figure
    """
    #User selects the cluster viewing (color, shape , or both)
    if viewOption=="No":
        colorUser=colorList
    else:
        colorUser='blue'

    #Create a plotly figure for visualization
    trace_usa=go.Scattergeo(
        lat=lat_list,
        lon=long_list,
        text=nodes_list,
        mode='markers',
        marker=dict(
            size=8,
            color=colorUser,
            opacity=0.7,
            symbol=shapeList
        ),
        hovertemplate="<b>%{text}</b><br><extra></extra>" #Default representation was the airport coordinates. It was changed to only retain the airport name when the user hovers the cursor over the airport on the plotly image. 
    )

    fig=go.Figure(data=trace_usa)
    fig.update_geos(scope='usa', showland=True)

    fig.update_layout(
        showlegend=False,
        height=700,  
        margin=dict(l=0, r=0, t=0, b=0),  #Adjust the top, bottom, left and right margins
    )

    return fig
