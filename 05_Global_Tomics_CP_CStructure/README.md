## Analyzing Performance of Single Models and Combo Models
Analyze_Model_Performance.ipynb: Visualization of the single and joint models.
Single_Models_plotly.py: Script producing a plotly plot for visualization of single model performance
Joint_Models_plotly.py: Script producing a plotly plot for visualization of joint model performance

## All Helper Models and Functions
Helper_Models.py: Models used by one or several scripts in one location
Helper_Functions.py: Functions used in one or several scripts in one location

## Subdirectories
UPPMAX_Combo_Scripts:
All scripts used to get the final combination models results. These scripts do not work with the paths found in the pharmbio pod, and the models are built differently since the all of the relevant data had to be moved to the high performance cluster.

Functional_But_Unused_Scripts: This a directory filled with scripts used to combine models. However, since the GPUs were needed during the last part of the thesis project, I switched to using UPPMAX. Therefore, the scripts are not complete and were not used for the final results of the project, although if somebody within pharmbio were to work on this project, these are probably the scripts they would need to use.

saved_images: Location of saved images used by scripts in Functional_But_Unused_Scripts
