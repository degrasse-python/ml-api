# create a virtual env
python3 -m venv ml-api
# init virtual env
source ml-api/bin/activate
# install fastapi
pip install fastapi uvicorn
# login
docker login
# use Docker to build images
docker image build -t adsaunde/ml-fastapi:0.1 .
# use kaniko to build the image and send it 
# kaniko --dockerfile=Dockerfile --context=. --destination=ml-fastapi:latest


