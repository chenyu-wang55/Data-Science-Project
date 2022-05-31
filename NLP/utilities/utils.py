import os
import utilities.string_utils as string_utils


def create_file(root, file_name):
    if string_utils.is_not_blank(root) and not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    file_path = os.path.join(root, file_name)
    open(file_path, "a").close()


def clear_text_file(path_to_file):
    """
    Clear a text file.
    :param path_to_file: path to the text file.
    """
    open(path_to_file, 'w').close()


def read_text_file(file):
    with open(file, 'r') as f:
        all_lines = f.read().splitlines()
    return all_lines


def append_to_text_file(file, text):
    """
    Write text to the end of file.
    :param file:
    :param text:
    :return:
    """
    if not os.path.exists(file):
        f = open(file, "w")
    else:
        f = open(file, "a")
    f.write(str(text) + '\n')
    f.close()


def append_to_text_file_and_print_line(file, text):
    """
    Write text to the end of file. And also print out the text in the console.
    :param file:
    :param text:
    :return:
    """
    append_to_text_file(file, text)
    print(text)


def append_list_to_text_file(file, lis):
    """
    Append list to text file. Each element in a separate line.
    :param file:
    :param lis:
    :return:
    """
    f = open(file, "a")
    for x in lis:
        append_to_text_file(file, x)
    f.close()


def append_dict_to_text_file(file, dictionary):
    """
    Append list to text file. Each key and element in a separate line.

    For example:
    'a' 3
    'b' 5
    ...

    :param file:
    :param dictionary:
    :return:
    """
    f = open(file, "a")
    for k, v in dictionary.items():
        append_to_text_file(file, '{} {}'.format(k, v))
    f.close()


def print_first_k_in_list(lis, k=None):
    """
    Print the first k elements in the list.
    :param lis:
    :param k:
    """
    if k is None:
        k = len(lis)
    for i, x in enumerate(lis[:min(k, len(lis))]):
        if '\n' in x[-2:]:
            print(x, end='')
        else:
            print(x)


def append_dict_to_file_first_n(file, dictionary, n=None):
    """
    Append the first n elements in the dictionary to file.
    :param dictionary:
    :param n:
    :return:
    """
    f = open(file, "a")
    if n is None:
        n = len(dictionary)
    keys = list(dictionary.keys())
    for i, key in enumerate(keys[:min(n, len(keys))]):
        value = dictionary[key]
        append_to_text_file(file, '{} {}'.format(key, value))
    f.close()


def print_dict_first_n(dictionary, n=None):
    """
    Print the first n elements in the dictionary.
    :param dictionary:
    :param n:
    :return:
    """
    if n is None:
        n = len(dictionary)
    keys = list(dictionary.keys())
    for i, key in enumerate(keys[:min(n, len(keys))]):
        value = dictionary[key]
        if isinstance(value, str) and '\n' in value[-2:]:
            print('{} {}'.format(key, value), end='')
        else:
            print('{} {}'.format(key, value))


def remove_values_from_list(lis, value):
    """
    Remove all occurrences of val from list.
    :param lis:
    :param value:
    :return:
    """
    return list(filter(lambda x: x != value, lis))


def create_folders(root, *new_folders):
    """
    Create new folders all under the root folder.
    :param root:
    :param new_folders:
    :return:
    """
    if string_utils.is_not_blank(root) and not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    for new_folder in new_folders:
        new_path = os.path.join(root, new_folder)
        if string_utils.is_not_blank(new_path) and not os.path.exists(new_path):
            os.mkdir(new_path)
