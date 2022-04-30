import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az


class GenerateDataAndFitModel:
    def __init__(self):
        np.random.seed(42)
        self.true_m = 0.5
        self.true_b = -1.3
        self.true_logs = np.log(0.3)


    def generateData(self):
        x = np.sort(np.random.uniform(0, 5, 50))
        y = self.true_b + self.true_m * x + np.exp(self.true_logs) * np.random.randn(len(x))
        return [x,y]

    def plotData(self, x,y):
        plt.plot(x,y, ".k")
        plt.ylim(-2, 2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def runModel(self):
        x,y=self.generateData()
        self.plotData(x,y)
        self.fitModel(x,y)


    def fitModel(self, x, y):
        """
        Fit a linear regression model y = mx + b using pymc3

        priors:
            - for m, b: their mean are assumed to be uniform distributions in [-5,5]
            - and their sd is assumed to be half normally distributed in [0,20]

        :param x:
        :param y:
        :return:
        """
        with pm.Model() as model:
            # Define the priors on each parameter:
            m = pm.Uniform("m", lower=-5, upper=5)
            b = pm.Uniform("b", lower=-5, upper=5)
            sd = pm.HalfNormal("sd", lower=0, upper=20)

            # Define the likelihood. A few comments:
            #  1. For mathematical operations like "exp", you can't use
            #     numpy. Instead, use the mathematical operations defined
            #     in "pm.math".
            #  2. To condition on data, you use the "observed" keyword
            #     argument to any distribution. In this case, we want to
            #     use the "Normal" distribution (look up the docs for
            #     this).
            pm.Normal("obs", mu=m * x + b, sd=sd, observed=y)

            # This is how you will sample the model. Take a look at the
            # docs to see that other parameters that are available.
            trace = pm.sample(
                draws=1000, tune=1000, chains=2, cores=2, return_inferencedata=True
            )
            az.plot_trace(trace,var_names=['m','b','sd'])


m = GenerateDataAndFitModel()
m.runModel()