import numpy as np
import math
import scipy as sc
import scipy.stats as sct

# Somewhat based on https://github.com/numenta/NAB/blob/master/nab/detectors/gaussian/windowedGaussian_detector.py
# BUT, we use established libraries like scipy to determine the Q-function which is just the survival function.

def get_anomaly_scores(values, window_size, step_size):
    anomaly_scores = []
    window_data = []
    step_buffer = []
    mean = 0
    std = 1
    for i in range(len(values)):
        anomaly_score = 0.0
        
        # Calculate the anomaly score using the Q-function.
        # The Q-function = tail distribution function of the normal distribution.
        # More specifically, Q(x) = P(X > x) 
        # according to https://en.wikipedia.org/wiki/Q-function
        # The survival function does the same thing:
        # sf(x, loc, scale) returns P(X > x) for a normal distribution with mean loc and stdev scale
        # API: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        # examples: https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p
        if len(window_data) > 0:
            # If the probability of something being greater than x is low,
            # that means that x is pretty big compared to the rest of our data.
            # So we want our anomaly score to reflect that and be large.
            # Thus, the anomaly score is 1 - Q-function.
            anomaly_score = 1 - sct.norm.sf(x=values[i],loc=mean,scale=std)

        anomaly_scores.append(anomaly_score)

        # You determine the mean and std from window_data.

        # So if window_data has not reached the size we want for our windows,
        # we append the current value to window_data and update our mean and std.
        if len(window_data) < window_size:
            window_data.append(values[i])
            mean = np.mean(window_data)
            std = np.std(window_data)
            # We do this to prevent division by 0 error.
            if std == 0.0:
                std = .000001

        # In this situation, window_data has reached the size we want for our windows.
        # We have a SLIDING window, but we do not slide every time step!
        # Instead, we slide every step_size many time steps.
        # To make sure we only slide every step_size many time steps,
        # we have a step_buffer that stores time steps until we make our next slide.
        else:
            step_buffer.append(values[i])

            # It is time to slide our window bc step_size many time steps have passed!
            if len(step_buffer) == step_size:
                # Slide window_data forward by step_size.
                # This means we are losing the first step_size many
                # time steps in window_data.
                window_data = window_data[step_size:]
                # Now we extend window_data by our step_buffer!
                window_data.extend(step_buffer)
                # Now that we have stored the step_buffer
                # into window_data, we can reset our step_buffer.
                step_buffer = []
                # Since we just slid our window, let's update the mean and std
                mean = np.mean(window_data)
                std = np.std(window_data)
                if std == 0.0:
                    std = .000001

    return anomaly_scores