

# Setup
```
docker login -u foundationmodels
```
and use access token as password

# Troubleshooting

if you get error
```Error saving credentials: error storing credentials - err: exit status 1, out: `error getting credentials - err: exit status 1```
Try the following
```
service docker stop
rm ~/.docker/config.json
service docker start
```