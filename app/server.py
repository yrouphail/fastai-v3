import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import numpy as np

export_file_url = 'https://www.googleapis.com/drive/v3/files/1k2hEbEsQYPyXC3WwRxSJBHzp9q192rNj?alt=media&key=AIzaSyDypQ3rlE3w6bF8D3MpXWsmdtjffNYzfTE'
export_file_name = 'mini-CheXpert-se101.pkl'

classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    pred_class,pred_idx,outputs = learn.predict(img)
    atel = outputs[0]
    atel = atel.numpy()*100
    atel_final = round(atel, 2)
    card = outputs[1]
    card = card.numpy()*100
    card_final = round(card, 2)
    cons = outputs[2]
    cons = cons.numpy()*100
    cons_final = round(cons, 2)
    edem = outputs[3]
    edem = edem.numpy()*100
    edem_final = round(edem, 2)
    pleu = outputs[4]
    pleu = pleu.numpy()*100
    pleu_final = round(pleu, 2)
    return JSONResponse({'result': 'Atelectasis:' +  str(atel_final) + '%'
                         \n
                         'Cardiomegaly:' +  str(card_final) + '%'
                         \n
                         'Consolidation:' +  str(cons_final) + '%'
                         \n
                         'Edema:' +  str(edem_final) + '%'
                         \n
                         'Pleural Effusion:' +  str(pleu_final) + '%'})
                         
                         
                         
                        


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
