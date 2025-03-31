from linkedin_api import Linkedin

api = Linkedin('niki.hu04@gmail.com', 'niki4317')

# profile = input("username: ")

print(api.get_profile_posts("mklikushin"))