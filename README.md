# Maya Dataset Creation
The Repository contains the code for dataset creation for the Training the Maya: Multilingual Aya Model

## Prerequisites
- If you have GPU in your machine, Make sure ensure all the [requirements](https://docs.cupy.dev/en/latest/install.html#requirements) are satisfied.

- If you don't have a GPU, you can contribute by writing the CPU based code and raise the PR.🤗

## Setup
**Note:** All the steps mentioned below are for Windows Machine. <br>

If you have other machines and successfully able to setup and run the translation script. Please contribute to README.md to add the steps.🤗

1. Create and activate virtual environment using the below commands.
```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

2. To install all the required packages run the below command:<br>
**Note:** Based on your cuda version, you may need to change the requirements.txt file.
```bash
pip install -r requirements.txt
```

3. Create `.env` file and place it in your root folder. <br>
**Note:** To get the Cohere API Key, You can DM `Surya Guthikonda` or `Karthik` in Discord.
```env
COHERE_API_KEY=<API_KEY>
```

## Execution
Note: Currently, sample dataset of 100 rows are provided in `translation/data/blip_laion_cc_sbu_100.json`. You can use original version
Move to the translation folder from root folder and run the script as follows:

```bash
cd translation\
python run.py
```

## Contribution Guidelines
Follow the discussion in `#maya-data-team` discord channel to understand the pending tasks.
1. View the Issues Tab for the tasks information.
2. Clone the repository and create your seperate branch to work on the Issue.
3. Once you have finished the task, you can raise PR from your branch to `main` by tagging the Issue.
4. Either `Surya Guthikonda` or `Karthik` will review the PR and merge into the `main`

**Note:** If you have any doubts, please feel free to ping in the discord.