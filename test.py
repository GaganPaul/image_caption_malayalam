from transformers import pipeline

translator = pipeline("translation", model="kurianbenoy/kde_en_ml_translation_model")
result = translator("Hello, how are you?")
print(result[0]['translation_text'])
