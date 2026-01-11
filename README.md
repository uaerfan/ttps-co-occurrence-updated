Option 1: Using Docker (Recommended)
1. Build the image: docker build -t ttp-analysis .
2. Run the analysis: docker run ttp-analysis

Option 2:
Python 3.12.12 version in Google Colab.(Recommended)

Steps to reproduce the results:
Step 01: Download the Git repository by running: git clone https://github.com/uaerfan/ttps-co-occurrence-updated.git
Step 02: Go to the repository with cd
Step 03: Install dependencies by running: !pip install -r requirements.txt
Step 04: Run main.py to reproduce the results by running: !python3 main.py
