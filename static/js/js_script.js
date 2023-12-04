function formValidation(){
    /*
    This function performs form validation upon form submission. In case any errors are found in any of the input fields, the appropriate error message is displayed next to it.
    It returns a boolean value. 'true' if no errors are found, and 'false' if one or more errors are found.

    Parameters:
            none

    Returns:
            boolean
    */
    resetErrors()
    var sError=""
    var eError=""
    var s1=false
    var e1=false
    var startVal=document.getElementById("start_date").value
    var endVal=document.getElementById("end_date").value

    //check for error in starting date input field
    if(startVal==""){
        //check if input field is empty
        sError="Empty value not allowed!"
    } else{
        //regex expression to see whether correct date format is entered
        if(!startVal.match(/^(0[1-9]|1[0-2])-[0-9]{4}$/)){
           sError="Incorrect date format!" 
        } else{
            var sM=parseInt(startVal.split('-')[0],10)
            var sY=parseInt(startVal.split('-')[1])
            //check whether date lies between the date range of the dataset
            if((sY<1990)||(sY>2009)){
                sError="Date value is outside the specified range!"
            } else{
                s1=true
            }
        }
    }

    //check for error in ending date input field
    if(endVal==""){
        //check if input field is empty
        eError="Empty value not allowed!"
    } else{
        //regex expression to see whether correct date format is entered
        if(!endVal.match(/^(0[1-9]|1[0-2])-[0-9]{4}$/)){
           eError="Incorrect date format!" 
        } else{
            var eM=parseInt(endVal.split('-')[0],10)
            var eY=parseInt(endVal.split('-')[1])
            //check whether date lies between the date range of the dataset
            if((eY<1990) || (eY>2009)){
                eError="Date value is outside the specified range!"
            } else{
                e1=true
            }
        }
    }

    //check if the starting date is greater than the ending date
    if(e1&&s1){    
        if((sY>eY)||((sM>eM)&&(sY==eY))){
            sError="Start date cannot be greater than end date!"
        }
    }

    var error_image1=document.getElementById("ex_image1")
    var error_image2=document.getElementById("ex_image2")
    var startProp=document.getElementById("start_error")
    var endProp=document.getElementById("end_error")

    //display error messages(if any)
    if((sError=="")&&(eError=="")){
        startProp.style.display="none"
        endProp.style.display="none"
        error_image1.style.visibility="hidden"
        error_image2.style.visibility="hidden"
        return true
    } else{
        if(sError!=""){
            error_image1.style.visibility="visible"
            startProp.style.display="block"
            startProp.innerText=sError
            startProp.style.top = (error_image1.offsetTop - startProp.clientHeight) + 'px';
            startProp.style.right = error_image1.offsetRight + 'px';
        }

        if(eError!=""){
            error_image2.style.visibility="visible"
            endProp.style.display="block"
            endProp.innerText=eError
            endProp.style.top = (error_image2.offsetTop - endProp.clientHeight) + 'px';
            endProp.style.right = error_image2.offsetRight + 'px';
        }

        return false
    }
}

function resetErrors(){
    /*
    This function resets the error messages next to the input fields regardless of whether they were present or not.

    Parameters:
            none

    Returns:
            void
    */
    var error_image1=document.getElementById("ex_image1")
    var error_image2=document.getElementById("ex_image2")
    var startProp=document.getElementById("start_error")
    var endProp=document.getElementById("end_error")
    startProp.style.display="none"
    endProp.style.display="none"
    error_image1.style.visibility="hidden"
    error_image2.style.visibility="hidden"
}

function emToPx(emValue) {
    /*
    This function converts 'em' measurement to 'pixel' measurement. Used only for generating the appropriate size of the plotly figure.
    Returns the 'pixel' measurement.

    Parameters:
            emValue: integer - 'em' measurement of the element.

    Returns:
            integer
    */
    //Get the font size of the root element in pixels
    const fontSize=parseFloat(getComputedStyle(document.documentElement).fontSize);
    return emValue*fontSize;
}

function popup(element, displayStatus) {
    /*
    This function displays the popup message above a particular element if the user hovers over it.
    Depending whether the display status is 'show' or 'hide' the popup message is appropriately shown or hidden.

    Parameters:
            element: object - The element over which the popup is to be displayed.
            displayStatus: string - Can take two values: 'show' or 'hide'.

    Returns:
            void
    */
    var hoverText = document.getElementById("popuptext");
    hoverText.innerText = "A valid date is between 01-1990 to 12-2009";
    if(displayStatus=="show"){
        hoverText.style.display = 'block';
        hoverText.style.top = (element.offsetTop - hoverText.clientHeight) + 'px';
        hoverText.style.right = element.offsetRight + 'px';
    }else{
        hoverText.style.display = 'none';
    }
}

function evalInfo(element,displayStatus){
    /*
    This function displays the popup message above a particular element if the user hovers over it.
    Depending whether the display status is 'show' or 'hide' the popup message is appropriately shown or hidden.

    Parameters:
            element: object - The element over which the popup is to be displayed.
            displayStatus: string - Can take two values: 'show' or 'hide'.

    Returns:
            void
    */
    var scoreText=document.getElementById("eval_info")
    if(displayStatus=="show"){
        scoreText.style.display="block"
        scoreText.style.bottom=(element.clientHeight/2) + 'px'
        scoreText.style.right=(element.clientWidth/2) + 'px'

    } else{
        scoreText.style.display="none"
    }
}

//main() body
document.addEventListener("DOMContentLoaded", function() {
    var alg=document.getElementById("algorithm_choice");
    var numCls=document.getElementById("num_clusters");
  
    //Add event listener to the dropdown1 to watch for changes
    alg.addEventListener("change", function() {
      //Check dropdown1 for a particular value so that dropdown2 can be disabled
      if ((alg.value==="4b")||(alg.value=="5b")){
        numCls.disabled=true;
      } else{
        numCls.disabled=false;
      }
    });

    document.getElementById("formData").addEventListener("submit", function(event) {
        event.preventDefault(); //Prevent the default form submission behavior

        if(formValidation()){
            //Disable the submit and reset button temporarily when the python script is running
            document.getElementById("submit_button").disabled=true
            document.getElementById("reset_button").disabled=true

            //Display 'generating...' text for the evaluation metrics container
            document.getElementById("eval_text").innerText="Generating the evaluation metrics."
            document.getElementById("sil_score").innerText="Silhoutte Score: Generating..."
            document.getElementById("ch_score").innerText="Calinski-Harabasz Score: Generating..."
            document.getElementById("db_score").innerText="Davies Bouldin Score: Generating..."
            
            //Display the loading animation when the python script is running. Also hide the plotly figure(if generated)
            document.getElementById("image_container").style.display="none"
            document.getElementById("animation_container").style.display="flex";

            const formData = new FormData(this); //Collect the form data

            const viewingCheckbox = document.getElementById("viewing");
            if(document.getElementById("num_clusters").disabled){
                formData.append('num_clusters','5c')
            }
            if (!viewingCheckbox.checked) {
                formData.append('viewing', 'No')
            }

            //Display the estimated time taken for running the python script
            fetch('/get_time',{
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const timeVar=JSON.parse(data.time_json)
                document.getElementById('ani_text2').innerText = "(It would take at most "+timeVar.toString()+" seconds to display the output)"

                //Call the python script which runs the clustering algorithm
                fetch('/call_python_script', {
                method: "POST",
                body: formData
                })
                
                .then(response => response.json())
                .then(data => {
                try {
                    //Parse the JSON data received from the server into a JavaScript object
                    const figureData = JSON.parse(data.image_results)
    
                    //Hide the loading animation when the python script has finished running running. Also display the image container for plotly image to be visible
                    document.getElementById("animation_container").style.display="none"
                    document.getElementById("image_container").style.display="flex"
    
                    //Use Plotly.react() to create or update the plotly figure on the webpage
                    figureData.layout.width=emToPx(66)
                    figureData.layout.height=emToPx(33)
                    Plotly.react('plotlyGraph', figureData)
    
                    //Display various evaluation scores
                    const evalDict=data.evaluation_output
                    if(Object.keys(evalDict).length==1){
                        document.getElementById("eval_text").innerText="Evaluation metrics cannot be produced with only one cluster!"
                        document.getElementById("sil_score").innerText="N/A"
                        document.getElementById("ch_score").innerText="N/A"
                        document.getElementById("db_score").innerText="N/A"
                    }else{
                        const dropdown1 = document.getElementById("spectral_choice")
                        const dropdown2 = document.getElementById("algorithm_choice")
                        const selectedIndex1 = dropdown1.selectedIndex
                        const selectedIndex2 = dropdown2.selectedIndex
                        document.getElementById("eval_text").innerText=dropdown2.options[selectedIndex2].text+" with initial "+dropdown1.options[selectedIndex1].text+" spectral clustering produced "+evalDict.Number_of_clusters+" clusters. The evaluation metrics are:"
                        document.getElementById("sil_score").innerText="Silhoutte Score: "+evalDict.Silhouette_Score
                        document.getElementById("ch_score").innerText="Calinski-Harabasz Score: "+evalDict.Calinski_Harabasz_Score
                        document.getElementById("db_score").innerText="Davies Bouldin Score: "+evalDict.Davies_Bouldin_Score
                    }
    
                } catch(error){
                    console.error("Error parsing JSON or missing 'image_results':", error);    
                }
                //Display the re-enable the functionality of the submit and reset button once the python script produces the output on the webpage.
                document.getElementById("submit_button").disabled=false
                document.getElementById("reset_button").disabled=false
    
                })
                .catch(error=>console.error("Error:",error));
            })
        }
    });
});
