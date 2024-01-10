import argparse
import json
import openai
import re
import time
# Create the parser
parser = argparse.ArgumentParser(description='Process OpenAI API key and JSONL file path.')

# Add arguments
parser.add_argument('--openai_api_key', default = "", help='Your OpenAI API key')
parser.add_argument('--answer_file', default = "answer.jsonl",help='Path to the JSONL file')

# Parse arguments
args = parser.parse_args()

openai.api_key = args.openai_api_key
NUM_SECONDS_TO_SLEEP = 10
# Define a function to query the OpenAI API and evaluate the answer
def get_yes_no_answer(question):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4-0314',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no.'
                }, {
                    'role': 'user',
                    'content': question,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    answer = response['choices'][0]['message']['content']
    yes_no_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)

    if yes_no_regex.match(answer):
        return answer.lower()
    else:
        return "Could not determine yes or no."


num_correct, num_total = 0, 0
# Continue with the processing of the JSONL file
with open(args.answer_file, 'r') as file:
    index, round_correct = 0, 0
    for line in file:
        data = json.loads(line)
        question, correct_answer, model_response = data["prompt"], data["answer"], data["response"]
        question4gpt = f"Given the following question {question}, the correct answer is {correct_answer}. Does the following answer correctly answers the question, answer:{model_response}?"
        gpt_grade = get_yes_no_answer(question4gpt)

        index += 1

        if gpt_grade=="yes":
            round_correct += 1
        if index == 2:
            index = 0
            if round_correct == 2:
                num_correct += 1
            round_correct = 0

        num_total += 1
print(f"The accuracy is {num_correct/num_total}")