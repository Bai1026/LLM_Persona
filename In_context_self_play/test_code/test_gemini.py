from google import genai

client = genai.Client(api_key="AIzaSyCtE0PMaJyjeZLHfIp4s3kzStj0q0vg8vU")

response = client.models.generate_content(model="gemini-2.0-flash", contents=["How does AI work?"])
print(response.text)
