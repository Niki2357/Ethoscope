import requests

# url = "https://api.x.com/2/users/by/username/annvandersteel"


headers = {"Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAAFGRyQEAAAAAxpcD7CHn1NusMPOH5AImAN%2FEGVs%3DWZvvOt5QyPOR27CRJ87rB6QQeRGFMi7BUDGLy3osbXzoPMA9L9"}

# response = requests.request("GET", url, headers=headers)

# print(response.text)


url = "https://api.x.com/2/users/1052225335318712320/tweets"

# headers = {"Authorization": "Bearer <token>"}

response = requests.request("GET", url, headers=headers)

print(response.text)