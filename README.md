# cmpe_489_term_project
The Git Page to keep the codes of the Cmpe489 project. Spring 2023
This is the readme file for the term project Cmpe489.

The code includes a Hierarchical Bayesian Model that is applied to a common dataset taken from
the Listening and Spoken Language Data Repository (LSL-DR) which is an international data repository that which stores the demographics and longitudinal outcomes.
The children who have hearing loss and are enrolled in programs focused on supporting listening and spoken language development. 
Researchers are interested in discovering factors related to improvements in educational outcomes within these programs. The data set have the features as explained below:

Male: The feature that shows the gender as 1-0
Siblings: The number of siblings in the household 
Family_inv: Index of family involvement which is an index to measure involvement of the family
In the program
Non_english: Whether the primary household language is not English 
Prev_disab: Presence of a previous disability
Non_white : Whether the race is white – non-white (1-0)
Age_test : Age at the time of testing (in months)
Non_severe_hl: Whether hearing loss is not severe (1-0)
Mother_hs : Whether the subject’s mother obtained a high school diploma or better ()
Early_ident: Whether the hearing impairment was identified by 3 months of age ().
Score: The outcome variable(target) is a standardized test score in one of several learning domains.

To predict the score variable there are various features in the dataset. First what we should do is the feauture selection. 
To do this a general method is to use regularization models such as lasso or ridge regression. 
In those models unnecessary variables are penalized and their coefficients shrunk towards 0 by the model. I
n a Bayesian context, we apply an appropriate priorvdistribution to the regression coefficients to make this happen. 
One such prior is the hierarchical regularized horseshoe, which uses two regularization strategies, one global and a set of local parameters, one for each coefficient. 
 
By using the above feature selection method as embedded in the top hieararchy of the model the code creates a Hierarchical Bayesian Regression model to
predict the scores of the children.
