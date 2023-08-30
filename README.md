# Diomedes
Diomedes is an LSTM and FeedForward Neural Network with heuristic rules for investment advisory.

# Summary
Diomedes serves as an automated investment advisory. It guides users on how to allocate their investments across various assets based on their financial profile, market conditions, and some predictive analytics.

# Functionality:

**1. Data Retrieval and Preprocessing:**

Using the yfinance library, Diomedes fetches stock market data, specifically for the S&P 500 index (^GSPC). The data is then adjusted based on certain market conditions and some features (like return rate) are calculated.

**2. Predictive Modeling:**

The data is processed to create lagged features for the model. Two neural network models are trained on this data: 

***a. A traditional feed-forward deep neural network (DNN), and***
 
***b. A more advanced Long Short-Term Memory (LSTM) neural network, suitable for time series data.*** 

Both models predict the potential future return of the market, with the average of the two predictions being used for final decision-making.

**3. Interactive Investment Advisory:**

The user interacts with Diomedes to provide personal financial details and preferences, such as age, income level, risk tolerance, savings, and investment goals. Based on the user's inputs and the market's predicted returns, Diomedes provides personalized investment allocation suggestions across various assets like stocks, bonds, real estate, and cash. In addition, Diomedes also provides reasons for each allocation decision, aiding user understanding.

**4. Output:**

The user is presented with a recommended dollar amount for investment in each asset type, along with reasons for these recommendations. A disclaimer reminds users to conduct their own research before making any investment decisions.

# Benefits for the End User:

**1. Personalized Recommendations:** 
The tool offers tailored investment advice based on individual financial circumstances and preferences.

**2. Data-driven Insights:** 
By leveraging historical market data and advanced machine learning models, the tool attempts to provide more informed investment strategies.

**3. Transparency:** 
The reasons given for each allocation decision help users understand the rationale behind the recommendations.

**4. Ease of Use:** 
The interactive nature of Diomedes makes it user-friendly, and it can serve as a starting point for those uncertain about how to allocate their investments.

**Note:** 
While the code offers data-driven insights, it's crucial for users to understand that investments are subject to risks, and it's always best to seek advice from a certified financial planner or conduct personal research before making investment decisions.

# Requirements

a. A Terminal Window

b. Python. Most modern computers have Python installed by default, but you can check to see by running this code:

```
python3 --version
```

If a version number prints out, you are good to go.

# Clone This Repository

Open your terminal window and run the command below:

```
git clone https://github.com/pavondunbar/Diomedes && cd Diomedes
```

# Create The Python Virtual Environment

A Python virtual environment for Diomedes is ideal because it isolates the AI model from other Python applications running on your system, thereby preventing library conflicts.

To create the virtual environment, run either

```
virtualenv FinAdvisor
```

OR

```
python3 -m venv FinAdvisor
```

This will create a Python virtual environment called "FinAdvisor".

# Activate The Virtual Environment

Run the below command to activate the FinAdvisor virtual environment

```
source FinAdvisor/bin/activate
```

The output will prepend (FinAdvisor) to our command line prompt, indicating you are now in the FinAdvisor virtual environment.

**Note:**
To deactivate the FinAdvisor virtual environment, simply run this command in your terminal:

```
deactivate
```

This command will end and detach the FinAdvisor virtual environment session.

# Install Python Libraries

Diomedes requires certain Python libraries for it to function properly. You can install these libraries by running the below command using 'pip':

```
pip install yfinance tensorflow numpy ta scikit-learn
```

# Activate The Diomedes AI Model

Now the fun part begins!  Simply type this command into your terminal window to initialize the Diomedes AI model:

```
python3 Diomedes.py
```

Diomedes will print a welcome message on your screen and begin to ask you a few questions to gauge your risk tolerance, market outlook, and investment goals.

Once you are done answering the questions, Diomedes will start iterating over the S&P 500 dataset and the yfinance dataset to detect patterns and anomalies as it relates **to your unique situation.**

After Diomedes is done training itself and analyzing the data, Diomedes will output an asset allocation portfolio ideal to your situation along with reasons as to why it determined the asset allocation specific to you.

# Conclusion

Thank you for using Diomedes, your personal AI Financial Planner. Let us know if you have any questions or suggestions to make it better!

If you have an issue, please open an Issue ticket above.  
