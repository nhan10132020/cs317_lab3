echo "Testing the prediction endpoint with 10 concurrent requests..."

seq 100 | xargs -n1 -P8 curl -s -o /dev/null -X POST 'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"alcohol":0,"malic_acid":0,"ash":0,"alcalinity_of_ash":0,"magnesium":0,"total_phenols":0,"flavanoids":0,"nonflavanoid_phenols":0,"proanthocyanins":0,"color_intensity":0,"hue":0,"od280/od315_of_diluted_wines":0,"proline":0}'

# GET /home 10 lần
seq 35 | xargs -n1 -I{} curl -s -X GET 'http://localhost:8000/home' -H 'accept: application/json'

# DELETE /delete 10 lần
seq 17 | xargs -n1 -I{} curl -s -X DELETE 'http://localhost:8000/delete' -H 'accept: application/json'

# PUT /update 10 lần
seq 50 | xargs -n1 -I{} curl -s -X PUT 'http://localhost:8000/update' -H 'accept: application/json'

# PATCH /modify 10 lần
seq 10 | xargs -n1 -I{} curl -s -X PATCH 'http://localhost:8000/modify' -H 'accept: application/json'
