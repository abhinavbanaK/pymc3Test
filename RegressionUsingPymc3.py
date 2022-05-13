import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import arviz as az
import bambi as bmb
from pymc3 import HalfCauchy, Model, Normal, glm, plot_posterior_predictive_glm, sample


class GenerateDataAndFitModel:
    def __init__(self):
        np.random.seed(42)
        self.true_m = 0.5
        self.true_b = -1.3
        self.true_logs = np.log(0.3)
        self.data = None


    def generateData(self):
        x = np.sort(np.random.uniform(0, 5, 50))
        self.regression_line = self.true_b + self.true_m * x
        y = self.regression_line + np.exp(self.true_logs) * np.random.randn(len(x))
        self.data = pd.DataFrame(dict(x=x, y=y))

        return [x,y]

    def plotData(self, x,y,trace):
        az.plot_trace(trace,figsize=(10,7))
        plt.show()
        plt.figure(figsize=(7, 7))
        plt.plot(x, y, "x", label="data")
        plot_posterior_predictive_glm(trace,samples=100,label="posterior predictive regression lines")
        plt.plot(x,self.regression_line,label="True Regression Line", lw=3.0, c="y")
        plt.title("Posterior predictive regression lines")
        plt.legend(loc=0)
        plt.xlabel("x")
        plt.ylabel("y");

        pm.plots.energyplot(trace)

        pm.forestplot(trace)
        plt.show()

    def runModel(self):
        x,y=self.generateData()

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
            sd = pm.HalfNormal("sd", sd=20)

            # Define the likelihood. A few comments:
            #  1. For mathematical operations like "exp", you can't use
            #     numpy. Instead, use the mathematical operations defined
            #     in "pm.math".
            #  2. To condition on data, you use the "observed" keyword
            #     argument to any distribution. In this case, we want to
            #     use the "Normal" distribution (look up the docs for
            #     this).
            fittedCurve=pm.Normal("obs", mu=m * x + b, sd=sd, observed=y)

            # This is how you will sample the model. Take a look at the
            # docs to see that other parameters that are available.
            trace = pm.sample(
                draws=1000, tune=1000, chains=2, cores=2, return_inferencedata=True
            )
        model = bmb.Model("y ~ x", self.data)
        trace = model.fit(draws=3000)
        self.plotData(x,y,trace)



m = GenerateDataAndFitModel()
m.runModel()