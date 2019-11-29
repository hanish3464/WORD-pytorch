import sys

def flush(buffer, cnt):
    buffer = ''
    cnt = 0
    return buffer, cnt

def DISLPLAY_STDOUT(chars=None, img_name=None, space=None):

    str_buffer = ''
    word = 0
    cnt = 0
    bubble = 0
    for k, char in enumerate(chars):
        cnt += 1
        str_buffer += char
        if cnt == space[bubble][word]:
            str_buffer += ' '
            word += 1
            cnt = 0
        if word == len(space[bubble]):
            print(str_buffer)
            bubble += 1
            word = 0
            str_buffer, cnt = flush(str_buffer, cnt)
