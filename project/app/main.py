from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api import predict, viz, ocr_reader

app = FastAPI(
    title='STORYSQUAD TEAM C DS API (LABS26)',
    description='DS RESTful API for the StorySquad Team C hosted in AWS Elastic Beanstalk',
    version='0.1',
    docs_url='/',
)

app.include_router(predict.router)
app.include_router(viz.router)
app.include_router(ocr_reader.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    uvicorn.run(app)
