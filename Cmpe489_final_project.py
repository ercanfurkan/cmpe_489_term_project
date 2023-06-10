#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary libraries
import pymc as pm
import pandas as pd
import numpy as np


# In[2]:


#The data is coming from pymc library
test_scores = pd.read_csv(pm.get_data("test_scores.csv"), index_col=0)
test_scores.head()


# In[3]:


#The distribution of the test scores
test_scores["score"].hist();


# In[4]:


test_scores.info()


# In[5]:


# Dropping missing values 
X = test_scores.dropna().astype(float)
y = X.pop("score")

# Standardizing the features
X -= X.mean()
X /= X.std()

N, D = X.shape


# In[6]:


#Filling missing values
Filled_data=test_scores.copy()
Filled_data["family_inv"]=Filled_data["family_inv"].fillna(Filled_data["family_inv"].mean())
Filled_data["prev_disab"]=Filled_data["prev_disab"].fillna(Filled_data["prev_disab"].mode()[0])
Filled_data["mother_hs"]=Filled_data["mother_hs"].fillna(Filled_data["mother_hs"].mode()[0])

Filled_data.info()


# In[7]:


X_f = Filled_data.astype(float)
y_f = X_f.pop("score")

# Standardizing the features
X_f -= X_f.mean()
X_f /= X_f.std()

N_f, D_f = X_f.shape


# In[8]:


#Data with missing values dropped
X.info()


# In[9]:


#The parameters that will be hold for dropped data
D0 = int(D / 2)
D0_f = int(D_f / 2)
D1=3


# In[10]:


from sklearn.model_selection import train_test_split
#Train test sets of dropped data
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y, test_size=0.20, random_state=42)


# In[11]:


from sklearn.model_selection import train_test_split
#Train test sets of filled data
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.20, random_state=42)


# In[44]:


import pytensor.tensor as at

with pm.Model(coords={"predictors": X_train_d.columns.values}) as test_score_model:
    # Prior on error SD
    sigma = pm.HalfNormal("sigma", 25)

    # Global shrinkage prior
    tau = pm.HalfStudentT("tau", 2, D0 / (D - D0) * sigma / np.sqrt(N))
    # Local shrinkage prior
    lam = pm.HalfStudentT("lam", 2, dims="predictors")
    c2 = pm.InverseGamma("c2", 1, 0.1)
    z = pm.Normal("z", 0.0, 1.0, dims="predictors")
    # Shrunken coefficients
    beta = pm.Deterministic(
        "beta", z * tau * lam * at.sqrt(c2 / (c2 + tau**2 * lam**2)), dims="predictors"
    )
    
    Xt_d = pm.MutableData("Xt_d", X_train_d)
    yt_d = pm.MutableData("yt_d", y_train_d)
    # No shrinkage on intercept
    beta0 = pm.Normal("beta0", 100, 25.0)

    scores = pm.Normal("scores", beta0 + at.dot(Xt_d, beta), sigma, observed=yt_d)


# In[99]:


#Graph of the dropped model
pm.model_to_graphviz(test_score_model)


# In[46]:


#Prior sampling to see whether our assumptions on priors make sense
with test_score_model:
    prior_samples = pm.sample_prior_predictive(200)
    
import arviz as az
import matplotlib.pyplot as plt
az.plot_dist(
    test_scores["score"].values,
    kind="hist",
    color="C1",
    hist_kwargs=dict(alpha=0.6),
    label="observed",
)
az.plot_dist(
    prior_samples.prior_predictive["scores"],
    kind="hist",
    hist_kwargs=dict(alpha=0.6),
    label="simulated",
)
plt.xticks(rotation=45);


# In[47]:


#Creating the model with the sampling
with test_score_model:
    idata = pm.sample(3000, tune=2000, random_seed=42,target_accept=0.99)


# In[48]:


#Plots of our model
az.plot_trace(idata, var_names=["tau", "sigma", "c2"]);


# In[49]:


az.plot_energy(idata);


# In[50]:


az.plot_forest(idata, var_names=["beta"], combined=True, hdi_prob=0.95, r_hat=True);


# In[51]:


az.summary(idata, round_to=2)


# In[53]:


with test_score_model:
    pm.set_data({"Xt_d":X_test_d
                 , "yt_d": y_test_d})
    idata.extend(pm.sample_posterior_predictive(idata))


# In[54]:


test_pred_d = idata.posterior_predictive["scores"].mean(dim=["chain", "draw"])


# In[83]:


from sklearn.metrics import mean_squared_error
import numpy as np
#Mean Squared Error
print(np.sqrt(mean_squared_error(y_test_d, test_pred_d)))
#Our Mean squared error is 22,7 for dropped model


# In[84]:


#The model with filled data
import pytensor.tensor as at

with pm.Model(coords={"predictors": X_train_f.columns.values}) as test_score_model_f:
    # Prior on error SD
    sigma = pm.HalfNormal("sigma", 25)

    # Global shrinkage prior
    tau = pm.HalfStudentT("tau", 2, D0_f / (D_f - D0_f) * sigma / np.sqrt(N_f))
    # Local shrinkage prior
    lam = pm.HalfStudentT("lam", 2, dims="predictors")
    c2 = pm.InverseGamma("c2", 1, 0.1)
    z = pm.Normal("z", 0.0, 1.0, dims="predictors")
    # Shrunken coefficients
    beta = pm.Deterministic(
        "beta", z * tau * lam * at.sqrt(c2 / (c2 + tau**2 * lam**2)), dims="predictors"
    )
    
    Xt_f = pm.MutableData("Xt_f", X_train_f)
    yt_f = pm.MutableData("yt_f", y_train_f)
    # No shrinkage on intercept
    beta0 = pm.Normal("beta0", 100, 25.0)

    scores = pm.Normal("scores", beta0 + at.dot(Xt_f, beta), sigma, observed=yt_f)


# In[85]:


#Graph of the dropped model
pm.model_to_graphviz(test_score_model_f)


# In[87]:


#Prior sampling to see whether our assumptions on priors make sense
with test_score_model_f:
    prior_samples = pm.sample_prior_predictive(200)
    
import arviz as az
import matplotlib.pyplot as plt
az.plot_dist(
    test_scores["score"].values,
    kind="hist",
    color="C1",
    hist_kwargs=dict(alpha=0.6),
    label="observed",
)
az.plot_dist(
    prior_samples.prior_predictive["scores"],
    kind="hist",
    hist_kwargs=dict(alpha=0.6),
    label="simulated",
)
plt.xticks(rotation=45);


# In[88]:


#Creating the model with the sampling (filled model)
with test_score_model_f:
    idata = pm.sample(3000, tune=2000, random_seed=42,target_accept=0.99)


# In[89]:


#Plots of our model (filled)
az.plot_trace(idata, var_names=["tau", "sigma", "c2"]);


# In[93]:


az.plot_energy(idata);


# In[94]:


az.plot_forest(idata, var_names=["beta"], combined=True, hdi_prob=0.95, r_hat=True);


# In[95]:


az.summary(idata, round_to=2)


# In[96]:


with test_score_model_f:
    pm.set_data({"Xt_f":X_test_f
                 , "yt_f": y_test_f})
    idata.extend(pm.sample_posterior_predictive(idata))


# In[97]:


test_pred_f = idata.posterior_predictive["scores"].mean(dim=["chain", "draw"])


# In[98]:


from sklearn.metrics import mean_squared_error
import numpy as np
#Mean Squared Error
print(np.sqrt(mean_squared_error(y_test_f, test_pred_f)))
#Our Mean squared error is 22,62 for dropped model

