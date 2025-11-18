# Basic Loadflow-ai
## Installation
Please Use Python 3.9!

Clone of the Git Repo
<pre><code>
git clone https://github.com/timonOconrad/loadflow-ai.git
</code></pre>

Install all requirements with pip
<pre><code>
pip install -r requirements.txt
</code></pre>

Download the train data (https://www.dropbox.com/scl/fi/omuifzdu60k9sl3vl9lr3/updated_parquet_file.parquet?rlkey=yws01l5duxdkwjckreay0oqqb&dl=0) and store it in the data folder.
## Grid:
An IEEE 5 test grid was used as the grid, in which the values at the buses were varied. More information on this can be found in the master's thesis (https://github.com/timonOconrad/static-voltage-stability-AI/blob/main/masterarbeit_final.pdf) in the chapter on data generation.

## Note:
We use the preparatory work from the following repo (https://github.com/timonOconrad/static-voltage-stability-AI).
