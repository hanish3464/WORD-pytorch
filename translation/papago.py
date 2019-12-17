import urllib.request
import json


def translation(load_from=None, save_to=None, id=None, pw=None):
    client_id = id
    client_secret = pw

    with open(load_from,'r',encoding='utf8') as f:
        srcText = f.read()

        encText = urllib.parse.quote(srcText)
        data = "source=ko&target=en&text=" + encText
        url = "https://openapi.naver.com/v1/papago/n2mt"
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            res = json.loads(response_body.decode('utf-8'))
            with open(save_to, 'w', encoding='utf8') as f:
                f.write(res['message']['result']['translatedText'])

        else:
            print("Error Code:" + rescode)
