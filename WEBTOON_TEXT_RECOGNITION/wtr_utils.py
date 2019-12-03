import sys
import codecs
import config

def flush(buffer, cnt):
    buffer = ''
    cnt = 0
    return buffer, cnt

def DISLPLAY_STDOUT(chars=None, space=None, img_name=None, MODE=None, Net=None):

    str_buffer = ''
    word = 0
    cnt = 0
    img_idx = 0
    bubble = 0
    with codecs.open(config.TEST_PREDICTION_PATH + Net + '.txt', 'w') as res:
        for k, char in enumerate(chars):
            cnt += 1
            str_buffer += char
            if cnt == space[bubble][word]:
                str_buffer += ' '
                word += 1
                cnt = 0
            if word == len(space[bubble]):
                if MODE == 'stdout': print(str_buffer)
                elif MODE == 'file': res.write(str_buffer + '\n')
                else:
                    print(str_buffer)
                    res.write(str_buffer + '\n')

                bubble += 1
                word = 0
                str_buffer, cnt = flush(str_buffer, cnt)

            if img_name[img_idx] != img_name[img_idx+1]:
                if MODE == 'stdout': print('')
                elif MODE == 'file': res.write('\n')
                else:
                    print('')
                    res.write('\n')

            img_idx += 1
            if len(chars) -1 == img_idx: return

