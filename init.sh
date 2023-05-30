# create a virtual env
python3 -m venv ml-api
# init virtual env
source ml-api/bin/activate
# install fastapi
pip install fastapi uvicorn
# use kaniko to build the image and send it 
kaniko --dockerfile=Dockerfile --context=. --destination=ml-fastapi:latest


