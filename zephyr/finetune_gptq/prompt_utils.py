def to_chat_text(example, instruction_field:str, target_field:str):

    '''
    Helper function to process the dataset sample by adding prompt and clean if necessary.

    Args:
    example: Data sample

    Returns:
    processed_example: Data sample post processing
    '''

    processed_example = "<|system|>\n You are a support chatbot who helps with user queries chatbot who always responds in the style of a professional.</s>\n<|user|>\n" + example[instruction] + "</s>\n<|assistant|>\n" + example[target]
    return processed_example

 