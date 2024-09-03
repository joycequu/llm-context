text_types = ["news", "values"]
for context_length in range(0, 4000, 1000):
    for text_type in text_types:
        mutate_info =  "generalDAN_" + text_type + "_" + str(context_length)
        completion_filepath = 'completions/completion_mistral_' + mutate_info + '_harmful.json'
        with open(completion_filepath, 'w') as file:
            file.write("")