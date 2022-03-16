# Security_Healthcare_MiniProject
Healthcare Model of Heart Disease and inclusion of Security in the Domain


“SECURITY IN HEALTHCARE SYSTEMS”

Problem statement

To design and implement a robust machine learning model to enhance the security of healthcare systems. The model aims to detect the inaccuracy in the data as a result of cyber-attack which can further lead to malfunctioning of automated machines used for diagnosis and medication.

Abstract

The healthcare industry has historically been an early adoption of technology developments, benefiting tremendously from them. Machine learning (a subset of artificial intelligence) is being used in a variety of health-related fields, including the invention of new medical treatments, the handling of patient data and records, and the treatment of chronic diseases.

But unfortunately machine learning models are known to be vulnerable to adversary-created inputs. Such adversarial examples can be derived from regular inputs, for example, by introducing minor—yet carefully chosen—perturbations that can be life-threatening as well.

Introduction
Healthcare cybersecurity has emerged as one of the most serious risks to the healthcare business. Because of the specifics outlined in the Health Insurance Portability and Accountability Act (HIPAA) laws, as well as the ethical commitment to help patients and the damage that healthcare security breaches can have on their lives, IT professionals must constantly address healthcare data security issues.

Electronic health records, or EHRs, include a wealth of sensitive information about patients' medical history, making hospital network security a top IT priority. EHRs allow physicians and other healthcare professionals, as well as insurance companies, to exchange critical information.

The networked aspect of contemporary healthcare, on the other hand, poses IT security problems, since storing so much vital data in an area that practically everyone uses makes it a visible target for hackers and thieves. In reality, the significance of data security in the healthcare industry has never been greater. Medical organizations must be diligent in developing precautions against World Wide Web dangers now more than ever, which is why a thorough awareness of the risks and measures available is critical.

While this may appear to be a simple task, healthcare data security poses a number of obstacles that are both typical in the IT profession and particular to hospital cybersecurity.




Why are healthcare information systems a target for cybercriminals?

The paradox of shared healthcare information is that it both makes patients safer and puts them at danger. The greater the network grows, the more valuable it becomes in providing high-quality medical treatment, but its data becomes more appealing to thieves.

In many cybercrimes, the attacker's purpose is to obtain information - either to sell or to utilise for personal gain. With the information available in electronic health records, a stranger may set up appointments, conduct costly medical procedures, or get prescription drugs in the patient's name. In such circumstances, the patient or the healthcare organisation may be held liable for the expenses or drugs.

Healthcare organisations have been subjected to direct attacks as well. Once a hacker gains access to a network, they can install ransomware to encrypt files or disable vital services until a certain ransom is paid. Because healthcare is such a time-sensitive industry, businesses are frequently forced to pay the ransom and hope that the money is eventually retrieved.

Although less prevalent, network-connected devices can potentially be hijacked to deliver wrong medicines or modify the operation of a machine. These innovations endanger patients' lives since a hacker may use this access to commit terrorism or take a health practitioner hostage. Healthcare providers cannot afford these possible hazards in medical scenarios when a decimal point or a tiny adjustment in dose might mean the difference between life and death.

Regardless of the hacker's motives, it's clear why network security is so critical.

What Should You Do If Your Healthcare Organization Experiences A Security Breach?

Maintaining a secure network may appear to be a lot of effort, but managing reports following a cyber-breach will be at least as much work — and that work is on top of your responsibilities to rectify the area that caused the violation in the first place.

If you suspect that your patient information has been compromised, take the following steps:

Report the Breach: If you discover unprotected or compromised network activity, you must notify the US Department of Health and Human Services; however, reporting timelines will vary based on the number of persons affected by the breach.
Distribute Information: Help your patients spot indications of fraud, such as unacknowledged medical bills and unfounded claims from insurance providers, even before you become a victim.
Re-examine Your Network: If an attacker obtained access to your organization's network, you must analyse the occurrence and protect any vulnerabilities that allowed threats to enter. This is an excellent moment to hire network specialists who can identify the existing gap as well as analyse for future issues and implement defences against future assaults.




Motivation

Securing a network can seem like an overwhelming — perhaps even impossible — task. Not only must all possibilities be considered when developing a strategy, but you must also find a means to offer significant maintenance to protect systems from being obsolete by the newest hacking tactics and to remain in accordance with newly revised legislation.
Despite these challenges, linked networks of patient information will continue to expand and include more of the medical industry. The consequences of information theft are too severe to risk, thus network security should be a top priority for every healthcare firm.



This project aims to implement a robust machine learning model that can efficiently predict the disease of a human, based on the symptoms that he/she possesses. Let us look into how we can approach this machine learning problem.

Approach used:



Gathering the Data: Data preparation is the primary step for any machine learning problem. We will be using a dataset from Kaggle for this problem. This dataset consists of two CSV files, one for training and one for testing. 

Cleaning the Data: Cleaning is the most important step in a machine learning project. The quality of our data determines the quality of our machine learning model. So it is always necessary to clean the data before feeding it to the model for training.

Model Building: After gathering and cleaning the data, the data is ready and can be used to train a machine learning model. We will be using this cleaned data to train the Support Vector Classifier, Naive Bayes Classifier, and Random Forest Classifier. We will be using a confusion matrix to determine the quality of the models.

Inference: We are comparing the accuracy determined by the clean data as well as with the adversarial example and detecting the outliers accordingly.

Models Used 

Logistic Regression-Logistic Regression is a method for estimating discrete values (typically binary values like 0/1) from a set of independent variables. It predicts the likelihood of an occurrence by fitting data to a logit function. It is also known as logit regression.

KNN Algorithm (K-Nearest Neighbours) - This approach is useful for both classification and regression problems. It appears to be more extensively utilised in the Data Science business to tackle categorization challenges. It's a straightforward algorithm that maintains all existing examples and classifies any new cases based on a majority vote of its neighbors. The case is then allocated to the class that has the most in common with it. This measurement is carried out via a distance function.

Random Forest Algorithm - A Random Forest is a collection of decision trees. Each tree is classed, and the tree "votes" for that class, in order to classify a new item based on its attributes. The classification with the highest votes is chosen by the forest (over all the trees in the forest).

SVM (Support Vector Machine) Algorithm - The SVM algorithm is a classification algorithm in which raw data is represented as points in an n-dimensional space (where n is the number of features you have). The value of each feature is then assigned to a specific point, making it simple to categorise the data. Classifier lines can be used to separate data and plot it on a graph.

Adversarial Examples

Specifically, adversarial examples are inputs to machine learning models designed by an attacker to cause the model to make a mistake. A machine learning adversarial example is a sample of input data or some new data that has been modified very slightly in a way that will cause a classification error. 
 
It is done by using a poisoning attack. A poisoning attack occurs when an adversary is able to inject bad data into your model's training pool, thus making it learn something it shouldn't be learning.

As a comparison we deployed the same model on two datasets - one clean and the other poisoned contain just 5% new poised data having adversarial examples.



Conclusion

Results showcase a 5% decrease in accuracy even after the hyper tuning of the model

One possible way to determine the degree of poisoned samples is to find the outliers and use it to detect any adversarial examples

Outlier detection can be used as a binary-classification issue to remove potentially deadly adversarial examples from a uniform reference distribution dataset by substituting them as outliers and then removing them.

Limitation
Limitations of the outlier model is that in most cases we as a user will hardly face any difficulties or suspect any adversarial data in an model, but we may face problem during the inference time of the model and in case we use any new algorithm to solve the error, it would be very time-consuming.
Future Scope
Scope of this project could be to dive deep into existing algorithms and propose some new defences against adversarial attacks like purifying the model from the alleged attack by the adversary and find some new methods to increase accuracy of the data before sending it for training and testing.



 References

Adversarial Attacks Against Medical Deep Learning Systems Samuel G. Finlayson
Adversarial-examples-in-deep-learning
Study on prediction of health care data using machine learning International Journal of Scientific & Technology Research
 A Machine Learning Approach for Heart Attack Prediction Suraj Kumar Gupta
