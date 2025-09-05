from flask import Flask, request, Response, redirect
from PIL import Image
import threading
import re
import base64
from screens import start_ui, set_image, screens_reset
import json

# Globals
remote_imgs = []

def global_reset():
    screens_reset()
    global remote_imgs
    remote_imgs = []

# sid = "L{id}" -> Local screen
# sid = "R{id}" -> Remote screen
def global_set_screen(sid: str, img: Image):
    pattern = r'(L|R)(\d+)'
    match = re.search(pattern, sid)
    if match:
        type = match.group(1)
        id = int(match.group(2))
        # print(type, id)
        if (type == "L"):
            set_image(id, img)
        if (type == "R"):
            remote_imgs[id] = img
        return
    raise

def event_stream(i):
    def send(type, data=None):
        return f'data: {json.dumps({"t": type, "d": data})}\n\n'
    yield send('hello', i)
    while True:
        if remote_imgs[i] is not None:
            img = remote_imgs[i]
            remote_imgs[i] = None
            yield send('img', 'data:image/png;base64,' + base64.b64encode(img).decode('utf-8'))
                 
# Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return redirect('/static/index.html')

@app.route('/reset', methods=['POST'])
def clear_all_screens():
    global_reset()
    return "All reset"

@app.route('/inform')
def sse():
    # id = int(request.args.get('id'))
    id = len(remote_imgs)
    remote_imgs.append(None)
    return Response(event_stream(id), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload_file():
    ret_str = 'Image received:'
    for sid_str in request.files:
        try:
            global_set_screen(sid_str, request.files[sid_str].read())
            ret_str = ret_str + ' ' + sid_str
        except:
            pass
    return ret_str

if __name__ == '__main__':
    threading.Thread(target=start_ui, daemon=True).start()
    app.run(debug=False, host='0.0.0.0', port=23333)
