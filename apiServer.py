import json
import urllib
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

import predictData


# %%
class HttpHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        print('do_get')
        o = urllib.parse.urlparse(self.path)
        print(o)
        self._response(o.path, o.query)

    def do_POST(self):
        print('do_post')
        args = self.rfile.read(int(self.headers['content-length'])).decode('utf-8')
        self._response(self.path, args)

    def _response(self, path, args):
        print('_response')
        rtv = {'c': 0, 'm': '', 'v': ''}
        print(path)
        print(args)
        args_json = {}
        if path == '/predict':
            try:
                if args:
                    if args[0] == '{' or args[0] == '[':
                        args = args.replace('\r', '\\r').replace('\n', '\\n').replace('\'', '\"')
                        args_json = json.loads(args)
                    else:
                        args = urllib.parse.parse_qs(args).items()
                        args_json = dict([(k, v[0]) for k, v in args])
            except Exception as e:
                rtv['m'] = 'Parse data error'
                rtv['c'] = 1

            if args_json.get('cmd') == 'predict' and args_json.get('data') != None:
                if args_json.get('cmd') == 'predict':
                    rtv['m'] = 'Success'
                    data = args_json.get('data')
                    print(len(data))
                    label = args_json.get('label')
                    layer = args_json.get('layer')
                    if layer == None:
                        preVal = predictData.predict_batch(data)
                    elif(label != None):
                        preVal = predictData.predict(data, layer, label)
                    else:
                        preVal = predictData.predict(data, layer)
                    rtv['v'] = preVal
                else:
                    rtv['m'] = 'Missing parameter("cmd":"predict")'
                    rtv['c'] = 3
            else:
                rtv['m'] = 'Missing parameter("cmd":"predict","data":[[float,float],...])'
                rtv['c'] = 4
        try:
            rtv = json.dumps(rtv, ensure_ascii=False)
        except Exception as e:
            rtv = {'c': 2, 'm': 'Response error', 'v': ''}
            rtv = json.dumps(rtv, ensure_ascii=False)

        code = 200
        print(rtv)
        self.send_response(code)
        self.send_header('Content-type', 'text/json;charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(rtv.encode())


if __name__ == '__main__':
    port = 4985
    address = '0.0.0.0'
    predictData.init()
    httpd = HTTPServer((address, port), HttpHandler)
    print('HTTPServer started => {}:{}'.format(address, port))
    httpd.serve_forever()
    print('HTTPServer had shutdown...')
