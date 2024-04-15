import openai

class askGPT():
    def __init__(self):
        # fill in the api key here
        openai.api_key = xxx
        
    def ask_gpt(self, question):
        with open('description/system_message.txt', 'r') as f:
            system_message = f.read()
        with open('description/ask_example.txt', 'r') as f:
            example_ask = f.read()
        with open('description/res_example.txt', 'r') as f:
            example_res = f.read()
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": example_ask},
                {"role": "assistant", "content": example_res},
                {"role": "user", "content": question}
            ]
        )
        return completion.choices[0].message['content']
    
    def ask_gpt_short_conversation(self, question):
        with open('concise_qa/system_message.txt', 'r') as f:
            system_message = f.read()
        with open('concise_qa/ask_example.txt', 'r') as f:
            example_ask = f.read()
        with open('concise_qa/res_example.txt', 'r') as f:
            example_res = f.read()
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": example_ask},
                {"role": "assistant", "content": example_res},
                {"role": "user", "content": question}
            ]
        )
        return completion.choices[0].message['content']

    def ask_gpt_conversation(self, question):
        with open('conversation/system_message.txt', 'r') as f:
            system_message = f.read()
        with open('conversation/ask_example.txt', 'r') as f:
            example_ask = f.read()
        with open('conversation/res_example.txt', 'r') as f:
            example_res = f.read()
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": example_ask},
                {"role": "assistant", "content": example_res},
                {"role": "user", "content": question}
            ]
        )
        return completion.choices[0].message['content']

    