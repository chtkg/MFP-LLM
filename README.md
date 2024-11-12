# MFP-LLM
Overview

MFP-LLM aims to utilize large language models (LLMs) for metro passenger flow prediction. By training on large-scale datasets, this project uses advanced natural language processing techniques to predict passenger flow for different time periods and metro lines. This method provides precise forecasts, helping optimize operations and improve service efficiency for metro systems.

Datasets

The model is evaluated using metro passenger flow datasets from two Chinese cities:

SHMetro: Data from the Shanghai metro system covering the period from July 1, 2016, to September 30, 2016. This dataset includes records from 288 metro stations with 958 physical edges, handling an average of 8.82 million passenger transactions daily. A total of 811.8 million transaction records were collected, each containing passenger ID, entry and exit stations, and corresponding timestamps. Passenger inflow and outflow for each station are recorded every 15 minutes. The dataset is divided into training (the first two months and the last three weeks), validation (remaining dates), and testing sets.

HZMetro: Data from the Hangzhou metro system covering January 2019. This dataset includes 80 stations with 248 physical edges, handling an average daily passenger flow of 2.35 million. Passenger flow is recorded every 15 minutes, capturing inflow and outflow data at each station. The dataset is split into training (January 1-18), validation (January 19-20), and testing (January 21-25) sets.

Model Architecture

