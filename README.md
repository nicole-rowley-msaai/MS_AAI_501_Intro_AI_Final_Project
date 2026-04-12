# MS_AAI_501_Intro_AI_Final_Project

**Introduction**

Financial fraud remains a major challenge for banks and e-commerce platforms as digital
transactions continue to grow. Fraud detection systems must identify rare fraudulent transactions
among millions of legitimate purchases while minimizing disruptions for normal customers. A
major difficulty is extreme class imbalance, where fraudulent transactions represent only a small
fraction of the data. Because rule-based systems struggle to adapt to evolving fraud strategies,
machine learning approaches are increasingly used to identify patterns in large transaction
datasets.

**Objectives**

This project proposes a hybrid machine learning pipeline using a dataset of
568,630 transaction records. The system will combine supervised classification
models and unsupervised anomaly detection to detect both known and emerging fraud patterns.
The supervised component will compare Random Forest and Gradient Boosting classifiers to
predict the probability that a transaction is fraudulent. These ensemble models are well suited to
structured financial data because they capture nonlinear feature relationships and perform well
with high-dimensional datasets. The system will also apply Isolation Forest or k-means
clustering to identify anomalous transactions that may represent new fraud patterns that are not
present in the labeled training data.

**Experiment Design**

Define a problem using a dataset (publicly available or self-provided) and describe it in terms of its real-world organizational or business application. The complexity level of the problem should be at least comparable to one of your assignments. The problem should use at least two different types of AI and machine learning algorithms that we have studied in this course, such as Classification, Clustering, and Regression, in an investigation of the analytics solution to the problem. This investigation must include some aspects of experimental comparison. Depending on the problem, you may choose to experiment with different types of algorithms, e.g., different types of classifiers, and some experiments with tuning parameters of the algorithms. Alternatively, if your problem is suitable, you may use multiple algorithms (Clustering + Classification, etc.). Note that if there are a larger number of attributes in your selected dataset, you can try some type of feature selection to reduce the number of attributes. You may use summary statistics and visualization techniques to help you explain your findings in this project.

**Requirements**

Some rules/tips about choosing AI challenges and data sets for your final projects:

Do not choose the problems that we have already analyzed in the course.
The dataset should not be small or made up. For this course, "small" is defined as fewer than 1000 examples in the dataset.
Choose a data set that does not require excessive data preprocessing. That said, some data preprocessing is expected.

The final teamwork project is an essential component of our courses in the AAI program. This project is representative of the kind of collaborative project you may work on during your career and in real-world projects. A significant portion of your final grade is drawn from your participation in this final group project, so you are strongly encouraged to work within your team and ensure that each team member contributes equally to the final project deliverables. Team members should plan to have clear and ongoing communication with other members and engage with the project and its deliverables each week. Lack of participation and engagement with both your team and your final project can result in a failing grade for the course. While these are the expectations for the project work, if you do experience difficulty with project advancement or challenges with team dynamics, contact your instructor for assistance promptly. If you are unable to perform the project as a team, contact your instructor to explore pursuing the entire final project independently.

**Final Project Presentation**

Prepare and record a final team project presentation by the end of Week 7. You may use any recording software you wish. Ensure that the sound quality of your video is good and that each member presents an equal portion of the presentation. The final project presentation is a chance to explain your problem and approach, provide your analysis results and interpretation, showcase what you have accomplished, and discuss your next step plan.

Upload your final team project presentation slides and group presentation, which should be between 20 and 30 minutes. Note that EVERY group member must participate in the final team project presentation. You might need to upload the recording to a video-sharing website, such as YouTube.com or Vimeo.com, and share the link to the recording on the title page of your presentation slides.

**Final Project Paper and Code**

You must submit a well-written report on your final project and the complete, well-documented, and clean source code by the end of Week 7. Your report (without Appendices), including text and selected tables/graphs, should be approximately 10 pages in length and describe the AI algorithms you implemented or deployed together with the data on which they were tested. Furthermore, you should include a detailed analysis of results or system performance (depending on the project's scope). Write and submit your final project report in APA 7 style, similar to (sample professional paper). 

The report, in Word or PDF, should contain the following contents: 

A description of the purpose, goals, and scope of your project. You should include references to papers you read on which your project and any algorithms you used are based. Include a discussion of whether you adapted a published algorithm or devised a new one, the range of problems and issues you addressed, and the relation of these problems and issues to the techniques and ideas covered in this course.
A clear specification of the AI algorithms you used, with analysis, evaluation, and critique of the algorithm and your implementation. For algorithm comparison, it is preferred that empirical comparison results are presented graphically.
An appendix that provides a list of each project participant and their detailed contributions to the project. 
Your code should be clearly documented. Submit your code with the project report to Canvas (or GitHub link directly to a public repository in your report and Canvas). Remember that your project report serves as the tour guide for your readers to be able to repeat your project process and discover the same patterns as you did. Only one member of your team will need to submit the deliverables during this final project.

