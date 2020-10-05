FROM ubuntu:18.04

# If we don't do this here we'll get interactive prompts to setup tzdata from apt
RUN ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV R_LIBS_SITE /usr/local/lib/R/site-library

# Ubuntu packages
RUN apt-get -y update

RUN apt-get -y install --no-install-recommends gnupg software-properties-common
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
RUN apt-get -y update

RUN apt-get -y install --no-install-recommends r-base r-base-dev r-base-core r-recommended python3-pip curl python3-dev git
RUN apt-get -y install --no-install-recommends libcurl4-gnutls-dev libudunits2-dev libssl-dev libxml2 libxml2-dev libfontconfig1-dev libcairo2-dev

# R packages
WORKDIR /app
COPY webr_0.1.0.tar.gz .

RUN Rscript -e "options(install.packages.check.source = 'no')"
RUN Rscript -e "install.packages('devtools')"
RUN Rscript -e "devtools::install_github('twitter/AnomalyDetection')"
RUN Rscript -e "install.packages('moonBook')"
RUN Rscript -e "install.packages('/app/webr_0.1.0.tar.gz', repos = NULL, type = 'source')"
RUN Rscript -e "install.packages('forecast')"
RUN Rscript -e "install.packages('stlplus')"
RUN Rscript -e "devtools::install_github('robjhyndman/anomalous')"

# Python packages

RUN pip3 install --upgrade pip setuptools wheel

# NOTE: We do this because installing FB Prophet in a requirements file actually does it twice because there are
# some hard dependencies it requires and it'll fail the first time.

RUN pip3 install pandas holidays pystan convertdate lunardate LunarCalendar tqdm
RUN pip3 install fbprophet

COPY ./exercise_1 /app/tutorial/exercise_1
COPY ./exercise_2 /app/tutorial/exercise_2
COPY ./exercise_3 /app/tutorial/exercise_3
COPY ./exercise_4 /app/tutorial/exercise_4
COPY ./exercise_5 /app/tutorial/exercise_5
COPY ./exercise_6 /app/tutorial/exercise_6
COPY ./exercise_7 /app/tutorial/exercise_7
COPY ./exercise_8 /app/tutorial/exercise_8
COPY ./exercise_9 /app/tutorial/exercise_9

COPY ./docker-start.sh /app
COPY ./requirements.txt /app

# Python
RUN pip3 install -r requirements.txt

# BCD
RUN git clone https://github.com/hildensia/bayesian_changepoint_detection.git /app/bayesian_changepoint_detection
WORKDIR /app/bayesian_changepoint_detection
RUN python3 setup.py install

# Clean up notebook environment
WORKDIR /app
RUN rm -rf ./tutorial/exercise_3/bayesian_changepoint_detection ./bayesian_changepoint_detection ./webr_0.1.0.tar.gz ./tutorial/exercise_*/requirements.txt

EXPOSE 8080

CMD /app/docker-start.sh
