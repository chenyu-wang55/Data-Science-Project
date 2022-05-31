from utilities import utils


def count_total_tokens(tokenized_lines):
    """
    Count the total number of tokens.
    :param tokenized_lines:
    :return:
    """
    count = 0
    for line in tokenized_lines:
        count += len(line)
    return count


def get_token_to_count_dictionary(tokenized_lines):
    """
    Get the dictionary of tokens to its number of occurrences.

    For example, the dictionary looks like:
    'a' -> 3
    'b' -> 6
    ...

    :param tokenized_lines:
    :return:
    """
    token_to_count = dict()
    for line in tokenized_lines:
        for token in line:
            if token in token_to_count:
                token_to_count[token] += 1
            else:
                token_to_count[token] = 1
    return token_to_count


def write_token_frequency_to_file(file, token_to_count_dict):
    """
    Write the token frequencies to text file.
    :param file:
    :param token_to_count_dict:
    :return:
    """
    utils.append_dict_to_text_file(file, token_to_count_dict)


def count_tokens_appearing_only_once(token_to_count_dict):
    frequencies = token_to_count_dict.values()
    count = sum(1 for x in frequencies if x == 1)
    return count