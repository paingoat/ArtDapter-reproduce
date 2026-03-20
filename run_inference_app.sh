#!/usr/bin/env bash

PYTHONPATH=. streamlit run inference/app.py --server.port 8502 --server.address 0.0.0.0
