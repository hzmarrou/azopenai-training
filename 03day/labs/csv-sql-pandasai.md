
### Talking to SQL 
 
In this lab, we're diving into the world of LLM driven data analysis and visualization using two powerful tools, Langchain and its `csv` and `sql` agents and Panda's AI. In Langchain, we are going to use SQL Database Agent and Panda's Dataframe Agent.

* Set up your environement by importing the necessary python packages, see the explain `.ipynb`  notebooks     
* Inititate you Large Language Model (LLM) Model    
* create sql db and toolkit
* Create a SQL Agent
* Ask the agent to describe the `playlisttrack` table 
* Ask the agent to  list the total sales per country. Which country's customers spent the most?
* Ask the agent to show the total number of tracks in each playlist. The Playlist name should be included in the result.


### Talking to CSV
Open the `titanic.csv` text reader (Notepad, Notepad ++, ..) and investigate visually the data     
* Please read the `csv` file `titanic.csv`   
* set up your azure LLM and your csv agent using langchain 
* Start investigating your csv file using the csv agent from Langchain. 


### Talking to `TSV` using PandasAI
* Open the `nyc-parking-violations.tsv` text reader (Notepad, Notepad ++, ..) and investigate visually the data.          
* Please read the `tsv` file `nyc-parking-violations.tsv`   
* set up your azure LLM and PandasAI
* Start investigating your TSV file using the chat function form PandasAI. Ask questions like  
        * Compute basic statistics such as the number of distinct values, empty values, and data types of the columns.
        * What is the minimum and maximum value for column 'Issue Date'
        * What is the schema for this data table?

#### Cleaning your data with PandasAI

* Open the `misspellings.csv` text reader (Notepad, Notepad ++, ..) and investigate visually the data.     
* Please read the `misspellings.csv` file.
* Count the values in of the attribute `Borough` the dataframe,  count the empty values as well, ans replace them with `Unknown`
* Find the misspeld states in the column `Borough`
* Replace them with the right spelling for the `state`






